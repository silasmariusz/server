"""
Music-structure awareness for the Hue Lights Sync analyzer.

Owns musical time (BPM + bar phase from the beat stream) and a section state
machine (NORMAL / RISE / DROP / SUSTAIN / BREAK) derived from the *shape* of the
energy over bars, plus the legacy climax score used by the manual strobe gate.

Everything is relative (fast-vs-slow envelope ratios), genre-agnostic, and cheap
(band sums + EMAs). It degrades gracefully: with no beats it estimates tempo from
onset intervals and free-runs the bar phase; with no onsets the spectral envelopes
alone still drive RISE/DROP/BREAK - which matters because the Sendspin beat/onset
streams can be empty or late for some tracks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

# Section state names (module constants to avoid string divergence across files).
SECTION_NORMAL = "normal"
SECTION_RISE = "rise"
SECTION_DROP = "drop"
SECTION_SUSTAIN = "sustain"
SECTION_BREAK = "break"

# Onset-density window and the count (per second) that counts as "full" density.
_ONSET_WINDOW_US = 1_000_000
_ONSET_DENSITY_FULL = 10.0

# Legacy climax-score weights (still feeds the manual strobe gate).
_DENSITY_WEIGHT = 1.2
_BRIGHT_WEIGHT = 0.6

# Band split over the mel spectrum (17 bins): bass 0-2, mid 2-10, high 10+.
_BASS_HI = 2
_HIGH_LO = 10

# Envelope coefficients (per render, ~30 Hz).
_FAST_A = 0.30  # ~100 ms
_SLOW_A = 0.015  # ~2 s
_DELAY_A = 0.04  # ~0.8 s at 30 Hz (delayed reference for the DROP step / RISE slope)
_CEIL_DECAY = 0.0008  # ceiling tracks the peak (~25 s) - holds the loud reference
# long enough to tell a verse from a drop, while still adapting across a set.

# Beat tracking.
_IBI_A = 0.18  # inter-beat-interval EMA (tempo settles in ~6 beats, octave-stable)
_ANCHOR_FALLBACK_BEATS = 8  # anchor the bar phase after N beats if no downbeat seen

# State-machine thresholds. Classification is driven by the smoothed, track-
# relative loudness level = broad_delayed/ceiling plus brightness, NOT by fragile
# instantaneous fast/slow ratios (which never spike on sustained-loud club music
# so a drop/sustain never classifies). Edges (DROP) use the jump in intensity.
_LOUD_FLOOR = 0.12  # absolute energy below which nothing counts as loud/sustain
_SUSTAIN_ENTER = 0.78  # smoothed intensity (broad_delayed/ceiling) to enter SUSTAIN
_SUSTAIN_EXIT = 0.62  # hysteresis: leave SUSTAIN below this
_DROP_STEP = 0.18  # intensity jump vs a delayed reference = a slam edge
_DROP_INT_MIN = 0.78  # a slam must also arrive loud
_DROP_REENTRY = 0.70  # intensity to call DROP when re-entering from a breakdown
_BREAK_LEVEL = 0.35  # smoothed-intensity collapse (near silence)
_BREAK_BRIGHT = 0.05  # brightness collapse (filter-down: highs gone, kick stays)
_BREAK_BRIGHT_EXIT = 0.12  # brightness must recover past this to leave a break
_BREAK_DEBOUNCE_US = 400_000
_RISE_BRIGHT_SLOPE = 0.006  # brightness climbing over its delayed ref = a build
_RISE_EXPECTED_BEATS = 16.0  # nominal build length for rise_progress 0..1
# Per-state minimum dwell (in beats; wall-clock fallback when no tempo) to debounce.
_MIN_DWELL_BEATS = 1
_SUSTAIN_ENTER_BEATS = 2
_SUSTAIN_EXIT_BEATS = 3
_DROP_HOLD_BEATS = 4
_RISE_EXIT_BEATS = 2  # hold RISE through a momentary slope dip so the build doesn't flicker


@dataclass(frozen=True, slots=True)
class SectionState:
    """Read-only snapshot of the current musical section + timing."""

    state: str = SECTION_NORMAL
    bpm: float = 0.0
    bar_phase: float = 0.0  # 0 at the downbeat, -> 1 at the bar end
    beat_in_bar: int = 0
    intensity: float = 0.0  # 0..1 instantaneous broad_fast/ceiling (noisier than the
    # smoothed `level` the state machine actually classifies on)
    rise_progress: float = 0.0  # 0..1 while RISE, else 0
    beats_since_change: int = 0


class StructureDetector:
    """Owns BPM, bar phase, the section state machine, and the climax score."""

    def __init__(self) -> None:
        """Initialize the detector."""
        self._onsets: deque[int] = deque()
        # beat tracker
        self._last_beat_ts = -1
        self._ibi_ema = 0.0
        self._beat_count = 0
        self._bar_anchor_ts = -1
        self._beats_per_bar = 4
        self._unanchored = 0
        self._anchored = False
        # energy envelopes
        self._broad_fast = self._broad_slow = self._broad_delayed = 0.0
        self._bass_fast = self._bass_slow = 0.0
        self._broad_ceiling = 1e-6
        self._brightness = 0.0
        self._int_ref = 0.0  # delayed EMA of intensity (for the DROP step edge)
        self._bright_ref = 0.0  # delayed EMA of brightness (for the RISE slope)
        self._initialized = False
        # state machine
        self._state = SECTION_NORMAL
        self._state_since_beat = 0
        self._state_since_us = 0
        self._below_since: int | None = None
        self._cached = SectionState()

    # -- inputs --

    def note_onset(self, timestamp_us: int) -> None:
        """Record an onset/transient at ``timestamp_us`` (server clock)."""
        self._onsets.append(timestamp_us)

    def note_beat(self, timestamp_us: int, is_downbeat: bool) -> None:
        """Feed a scheduled beat into the tempo + bar-phase tracker."""
        if self._last_beat_ts > 0:
            ibi = timestamp_us - self._last_beat_ts
            if ibi > 0:
                if self._ibi_ema <= 0:
                    self._ibi_ema = ibi
                elif 0.5 * self._ibi_ema <= ibi <= 2.0 * self._ibi_ema:
                    self._ibi_ema += _IBI_A * (ibi - self._ibi_ema)
        self._last_beat_ts = timestamp_us
        self._beat_count += 1
        if is_downbeat:
            self._bar_anchor_ts = timestamp_us
            self._anchored = True
            self._unanchored = 0
        elif not self._anchored:
            self._unanchored += 1
            if self._unanchored >= _ANCHOR_FALLBACK_BEATS:
                self._bar_anchor_ts = timestamp_us
                self._anchored = True

    def reset(self) -> None:
        """Clear all state (used on stream restart / track change)."""
        self._onsets.clear()
        self._last_beat_ts = -1
        self._ibi_ema = 0.0
        self._beat_count = 0
        self._bar_anchor_ts = -1
        self._unanchored = 0
        self._anchored = False
        self._initialized = False
        self._int_ref = 0.0
        self._bright_ref = 0.0
        self._state = SECTION_NORMAL
        # Clear the dwell anchors too: otherwise dwell_beats = beat_count(0 after reset)
        # - stale _state_since_beat goes negative on the next track and stalls every
        # transition. The warm-start in update() re-anchors them to live time.
        self._state_since_beat = 0
        self._state_since_us = 0
        self._below_since = None
        self._cached = SectionState()

    # -- per render --

    def update(self, now_us: int, spectrum: list[float]) -> None:
        """Advance the envelopes + section state machine one render tick."""
        if not spectrum:
            return
        count = len(spectrum)
        bass = sum(spectrum[0:_BASS_HI]) / _BASS_HI if count >= _BASS_HI else 0.0
        mid = sum(spectrum[_BASS_HI:_HIGH_LO]) / (_HIGH_LO - _BASS_HI) if count >= _HIGH_LO else 0.0
        high = sum(spectrum[_HIGH_LO:count]) / max(1, count - _HIGH_LO) if count > _HIGH_LO else 0.0
        broad = sum(spectrum) / count
        self._brightness = high / (bass + mid + high + 1e-6)

        if not self._initialized:
            self._initialized = True
            self._broad_fast = self._broad_slow = self._broad_delayed = broad
            self._bass_fast = self._bass_slow = bass
            self._broad_ceiling = max(1e-6, broad)
            # Warm-start the step/slope references so the first frame is not seen
            # as a huge upward jump (which would false-trigger a DROP at track start).
            self._int_ref = 1.0  # intensity = broad_fast/ceiling = 1.0 at init
            self._bright_ref = self._brightness
            # Anchor the dwell timer to live time (post-reset _state_since_us is 0,
            # so wall-clock elapsed would otherwise be enormous and skip min-dwell).
            self._state_since_us = now_us
            self._state_since_beat = self._beat_count
        else:
            self._broad_fast += _FAST_A * (broad - self._broad_fast)
            self._broad_slow += _SLOW_A * (broad - self._broad_slow)
            self._broad_delayed += _DELAY_A * (broad - self._broad_delayed)
            self._bass_fast += _FAST_A * (bass - self._bass_fast)
            self._bass_slow += _SLOW_A * (bass - self._bass_slow)
            if self._broad_fast > self._broad_ceiling:
                self._broad_ceiling = self._broad_fast
            else:
                self._broad_ceiling += _CEIL_DECAY * (self._broad_fast - self._broad_ceiling)

        self._step_state(now_us)

    def section(self) -> SectionState:
        """Return the latest section snapshot (computed in update())."""
        return self._cached

    @property
    def bpm(self) -> float:
        """Current tempo estimate (0 if unknown)."""
        return 60_000_000.0 / self._ibi_ema if self._ibi_ema > 0 else 0.0

    def bar_phase(self, now_us: int) -> float:
        """Position within the current bar: 0 at the downbeat, -> 1 at the bar end."""
        if self._ibi_ema <= 0 or self._bar_anchor_ts < 0:
            return 0.0
        bars = (now_us - self._bar_anchor_ts) / (self._ibi_ema * self._beats_per_bar)
        return bars - int(bars) if bars >= 0 else 0.0

    def beat_in_bar(self, now_us: int) -> int:
        """Which beat of the bar we are on (0-based), 0 if unknown."""
        if self._ibi_ema <= 0 or self._bar_anchor_ts < 0:
            return 0
        beats = int((now_us - self._bar_anchor_ts) / self._ibi_ema)
        return beats % self._beats_per_bar if beats >= 0 else 0

    def onset_density(self, now_us: int) -> float:
        """Return onsets observed in the trailing one-second window."""
        cutoff = now_us - _ONSET_WINDOW_US
        onsets = self._onsets
        while onsets and onsets[0] < cutoff:
            onsets.popleft()
        return float(len(onsets))

    def climax_score(self, now_us: int, spectrum: list[float]) -> float:
        """Legacy climax score for the manual strobe gate (loud + dense + bright)."""
        if not spectrum:
            return 0.0
        count = len(spectrum)
        broad = sum(spectrum) / count
        high_lo = min(_HIGH_LO, count)
        high = sum(spectrum[high_lo:]) / max(1, count - high_lo) if count > high_lo else 0.0
        density_factor = min(1.0, self.onset_density(now_us) / _ONSET_DENSITY_FULL)
        return (
            broad * (1.0 + _DENSITY_WEIGHT * density_factor)
            + _BRIGHT_WEIGHT * high * density_factor
        )

    # -- internals --

    def _step_state(self, now_us: int) -> None:  # noqa: PLR0915
        ceiling = self._broad_ceiling + 1e-6
        intensity = max(0.0, min(1.0, self._broad_fast / ceiling))  # instantaneous
        level = max(0.0, min(1.0, self._broad_delayed / ceiling))  # ~0.8 s smoothed tier
        brightness = self._brightness
        int_step = intensity - self._int_ref
        bright_slope = brightness - self._bright_ref
        loud = self._broad_fast >= _LOUD_FLOOR
        # Without a downbeat anchor we cannot place the "1", so allow drops anytime.
        on_downbeat = not self._anchored or self.bar_phase(now_us) < 0.18
        have_beats = self._ibi_ema > 0
        beat_dur = self._ibi_ema if have_beats else 500_000
        dwell_beats = self._beat_count - self._state_since_beat
        elapsed = now_us - self._state_since_us
        # BREAK is bass-AGNOSTIC: a true silence (smoothed-intensity collapse with no
        # loudness) OR a filter-down where the kick keeps going but the highs vanish
        # (brightness collapse) - the latter must NOT require energy to drop.
        is_break = (level < _BREAK_LEVEL and not loud) or brightness < _BREAK_BRIGHT
        state = self._state

        def change(new_state: str) -> None:
            self._state = new_state
            self._state_since_beat = self._beat_count
            self._state_since_us = now_us

        def held(beats: int) -> bool:
            # Min-dwell satisfied by beat count when we have a tempo, else wall-clock.
            return dwell_beats >= beats if have_beats else elapsed >= beats * beat_dur

        # Debounce the collapse so a one-frame dip doesn't trip a breakdown.
        self._below_since = now_us if is_break and self._below_since is None else self._below_since
        if not is_break:
            self._below_since = None
        break_ready = (
            self._below_since is not None and now_us - self._below_since >= _BREAK_DEBOUNCE_US
        )

        step_drop = int_step >= _DROP_STEP and intensity >= _DROP_INT_MIN and on_downbeat
        reentry_drop = state == SECTION_BREAK and intensity >= _DROP_REENTRY and on_downbeat

        if state != SECTION_DROP and (step_drop or reentry_drop):
            change(SECTION_DROP)  # a slam interrupts anything (highest priority)
        elif state == SECTION_DROP:
            if held(_DROP_HOLD_BEATS):
                if is_break:
                    change(SECTION_BREAK)
                elif level >= _SUSTAIN_EXIT and loud:
                    change(SECTION_SUSTAIN)
                else:
                    change(SECTION_NORMAL)
        elif state == SECTION_BREAK:
            if (
                level >= _BREAK_LEVEL + 0.10
                and brightness >= _BREAK_BRIGHT_EXIT
                and held(_MIN_DWELL_BEATS)
            ):
                change(SECTION_SUSTAIN if level >= _SUSTAIN_ENTER and loud else SECTION_NORMAL)
        elif break_ready:
            change(SECTION_BREAK)
        elif state == SECTION_SUSTAIN:
            if level < _SUSTAIN_EXIT and held(_SUSTAIN_EXIT_BEATS):
                change(SECTION_NORMAL)
        elif level >= _SUSTAIN_ENTER and loud and held(_SUSTAIN_ENTER_BEATS):
            change(SECTION_SUSTAIN)  # held-high energy -> SUSTAIN directly (the 3:43 fix)
        elif (
            state != SECTION_RISE
            and bright_slope > _RISE_BRIGHT_SLOPE
            and int_step > 0
            and level < _SUSTAIN_ENTER
            and held(_MIN_DWELL_BEATS)
        ):
            change(SECTION_RISE)
        elif (
            state == SECTION_RISE and bright_slope < -_RISE_BRIGHT_SLOPE and held(_RISE_EXIT_BEATS)
        ):
            change(SECTION_NORMAL)

        # Delayed references for the next tick's step/slope (read above, update now).
        self._int_ref += _DELAY_A * (intensity - self._int_ref)
        self._bright_ref += _DELAY_A * (brightness - self._bright_ref)

        rise_progress = (
            max(0.0, min(1.0, (now_us - self._state_since_us) / (_RISE_EXPECTED_BEATS * beat_dur)))
            if self._state == SECTION_RISE
            else 0.0
        )
        self._cached = SectionState(
            state=self._state,
            bpm=self.bpm,
            bar_phase=self.bar_phase(now_us),
            beat_in_bar=self.beat_in_bar(now_us),
            intensity=intensity,
            rise_progress=rise_progress,
            beats_since_change=self._beat_count - self._state_since_beat,
        )
