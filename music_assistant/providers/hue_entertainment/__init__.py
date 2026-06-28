"""
Hue Lights Sync Plugin for Music Assistant.

Syncs Philips Hue lights in Entertainment Areas to music using the
Sendspin visualization pipeline and the Hue Entertainment API (DTLS streaming).

Each Entertainment Area on a paired Hue bridge appears as a virtual player
in Music Assistant. Playing music to the player activates entertainment mode
and makes the lights react to the music in real time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hue_entertainment import HueEntertainmentAPI
from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import ConfigEntryType, ProviderFeature
from music_assistant_models.errors import LoginFailed, SetupFailedError

from .constants import (
    COLOR_MODES,
    CONF_ACTION_PAIR,
    CONF_BRIDGE_HOST,
    CONF_BRIDGE_ID,
    CONF_BRIGHTNESS,
    CONF_CLIENTKEY,
    CONF_COLOR_MODE,
    CONF_HUE_LATENCY_MS,
    CONF_PALETTE,
    CONF_PALETTE_ROTATE,
    CONF_PALETTE_ROTATE_BEATS,
    CONF_PALETTE_ROTATE_LIST,
    CONF_PALETTE_ROTATE_SMOOTH,
    CONF_PERLIGHT_BRIGHTNESS_DATA,
    CONF_PULSE_DECAY,
    CONF_PULSE_DOWNBEAT,
    CONF_PULSE_FLOOR,
    CONF_PULSE_SELECT,
    CONF_STROBE_AUTO,
    CONF_STROBE_BEAT_SYNC,
    CONF_STROBE_BLACKOUT,
    CONF_STROBE_BRIGHTNESS,
    CONF_STROBE_COLOR,
    CONF_STROBE_COVERAGE,
    CONF_STROBE_DUTY,
    CONF_STROBE_ENABLED,
    CONF_STROBE_FLASH_HZ,
    CONF_STROBE_LIGHTS,
    CONF_STROBE_MIN_HOLD_MS,
    CONF_STROBE_RELEASE_MS,
    CONF_STROBE_SENSITIVITY,
    CONF_USERNAME,
    DEFAULT_COLOR_MODE,
    DEFAULT_HUE_LATENCY_MS,
    DEFAULT_PALETTE,
    DEFAULT_PALETTE_ROTATE,
    DEFAULT_PALETTE_ROTATE_BEATS,
    DEFAULT_PALETTE_ROTATE_SMOOTH,
    DEFAULT_PERLIGHT_BRIGHTNESS_DATA,
    DEFAULT_PULSE_DECAY,
    DEFAULT_PULSE_DOWNBEAT,
    DEFAULT_PULSE_FLOOR,
    DEFAULT_PULSE_SELECT,
    DEFAULT_STROBE_AUTO,
    DEFAULT_STROBE_BEAT_SYNC,
    DEFAULT_STROBE_BLACKOUT,
    DEFAULT_STROBE_BRIGHTNESS,
    DEFAULT_STROBE_COLOR,
    DEFAULT_STROBE_COVERAGE,
    DEFAULT_STROBE_DUTY,
    DEFAULT_STROBE_ENABLED,
    DEFAULT_STROBE_FLASH_HZ,
    DEFAULT_STROBE_MIN_HOLD_MS,
    DEFAULT_STROBE_RELEASE_MS,
    DEFAULT_STROBE_SENSITIVITY,
    HUE_DEVICE_TYPE,
    PULSE_SELECT_OPTIONS,
    STROBE_COLOR_OPTIONS,
)
from .palettes import palette_names

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from hue_entertainment import EntertainmentArea
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

SUPPORTED_FEATURES: set[ProviderFeature] = set()


async def setup(
    mass: MusicAssistant,
    manifest: ProviderManifest,
    config: ProviderConfig,
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    from .provider import HueEntertainmentProvider  # noqa: PLC0415

    if not config.get_value(CONF_BRIDGE_HOST) or not config.get_value(CONF_USERNAME):
        msg = "Hue bridge not paired. Please configure the bridge host and complete pairing."
        raise SetupFailedError(msg)

    return HueEntertainmentProvider(mass, manifest, config, SUPPORTED_FEATURES)


async def _handle_pair_action(values: dict[str, ConfigValueType]) -> None:
    """Execute the pairing flow when the pair button is clicked."""
    host = str(values.get(CONF_BRIDGE_HOST, "")).strip()
    if not host:
        msg = "Bridge host/IP address is required for pairing"
        raise LoginFailed(msg)

    LOGGER.info("Starting Hue bridge pairing with %s ...", host)
    api = HueEntertainmentAPI(host)
    try:
        credentials = await api.pair(device_type=HUE_DEVICE_TYPE)
        values[CONF_USERNAME] = credentials["username"]
        values[CONF_CLIENTKEY] = credentials["clientkey"]
        # Fetch and store bridge ID for mDNS IP tracking
        api_authed = HueEntertainmentAPI(host, credentials["username"])
        try:
            bridge_id = await api_authed.get_bridge_id()
            if bridge_id:
                values[CONF_BRIDGE_ID] = bridge_id
        finally:
            await api_authed.close()
        LOGGER.info("Hue bridge pairing successful!")
    except TimeoutError as err:
        LOGGER.warning("Hue bridge pairing timed out: %s", err)
        raise LoginFailed(str(err)) from err
    except Exception as err:
        LOGGER.warning("Hue bridge pairing failed: %s", err)
        msg = f"Failed to connect to Hue bridge at {host}: {err}"
        raise LoginFailed(
            msg,
            translation_key="connect_failed",
            translation_args=[host],
        ) from err
    finally:
        await api.close()


def _areas_to_options(areas: list[EntertainmentArea]) -> list[ConfigValueOption]:
    """
    Build a selectable option for every channel across the given areas.

    Each option value is "<area_id>:<channel_id>" so a selection survives even
    though channel ids restart at 0 per area. Channels that share a device name
    within an area (the segments of a gradient lightstrip) are numbered
    "<name> - section N" in channel order so each segment is individually
    selectable.
    """
    options: list[ConfigValueOption] = []
    for area in areas:
        name_counts: dict[str, int] = {}
        for channel in area.channels:
            name_counts[channel.name] = name_counts.get(channel.name, 0) + 1
        seen: dict[str, int] = {}
        for channel in area.channels:
            base = channel.name or f"Light {channel.channel_id}"
            if name_counts.get(channel.name, 0) > 1:
                seen[channel.name] = seen.get(channel.name, 0) + 1
                title = f"{area.name} - {base} - section {seen[channel.name]}"
            else:
                title = f"{area.name} - {base}"
            options.append(ConfigValueOption(value=f"{area.id}:{channel.channel_id}", title=title))
    return options


async def _fetch_strobe_light_options(
    values: dict[str, ConfigValueType] | None,
) -> list[ConfigValueOption]:
    """
    Fetch entertainment areas from the bridge and build channel options.

    Fallback used during initial setup, before the provider instance (with its
    already-discovered areas) exists.
    """
    if not values:
        return []
    host = str(values.get(CONF_BRIDGE_HOST, "")).strip()
    username = str(values.get(CONF_USERNAME, "")).strip()
    if not host or not username:
        return []
    api = HueEntertainmentAPI(host, username)
    try:
        areas = await api.get_entertainment_areas()
    except Exception as err:
        LOGGER.debug("Could not list channels for strobe options: %s", err)
        return []
    finally:
        await api.close()
    return _areas_to_options(areas)


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """
    Return Config entries to setup this provider.

    :param action: Action key called from config entries UI.
    :param values: Intermediate raw values for config entries.
    """
    if action == CONF_ACTION_PAIR and values:
        await _handle_pair_action(values)

    # Determine pairing state from current values
    paired = bool((values or {}).get(CONF_USERNAME) not in (None, ""))

    # Build status label based on pairing state
    if action == CONF_ACTION_PAIR and paired:
        status_text = "Successfully paired with Hue bridge! Click save to complete setup."
    elif paired:
        status_text = "Paired with Hue bridge."
    else:
        status_text = (
            "Press the physical button on your Hue bridge, "
            "then click the pair button below within 30 seconds."
        )

    # Prefer the channels already discovered by the loaded provider instance
    # (no network round-trip); fall back to a fresh fetch during initial setup.
    strobe_options: list[ConfigValueOption] = []
    if instance_id and (prov := mass.get_provider(instance_id)) is not None:
        strobe_options = _areas_to_options(getattr(prov, "entertainment_areas", []))
    if not strobe_options:
        strobe_options = await _fetch_strobe_light_options(values)

    # Live preview URL - only resolvable once the instance exists and the webserver
    # knows its base url. Surfaced as a clickable help link inside the settings.
    preview_url = ""
    if instance_id and mass.webserver.base_url:
        preview_url = f"{mass.webserver.base_url}/hue_preview"

    entries: list[ConfigEntry] = [
        ConfigEntry(
            key=CONF_BRIDGE_HOST,
            type=ConfigEntryType.STRING,
            required=True,
            value=values.get(CONF_BRIDGE_HOST) if values else None,
        ),
        ConfigEntry(
            key=CONF_BRIDGE_ID,
            type=ConfigEntryType.STRING,
            required=False,
            hidden=True,
            value=values.get(CONF_BRIDGE_ID) if values else None,
        ),
        ConfigEntry(
            key="status_label",
            type=ConfigEntryType.LABEL if paired else ConfigEntryType.ALERT,
            label=status_text,
            depends_on=CONF_BRIDGE_HOST,
        ),
        ConfigEntry(
            key=CONF_ACTION_PAIR,
            type=ConfigEntryType.ACTION,
            translation_key="repair_bridge" if paired else None,
            action=CONF_ACTION_PAIR,
            depends_on=CONF_BRIDGE_HOST,
            required=False,
            hidden=action == CONF_ACTION_PAIR and paired,
        ),
        ConfigEntry(
            key=CONF_USERNAME,
            type=ConfigEntryType.SECURE_STRING,
            hidden=True,
            required=False,
            default_value="",
            value=values.get(CONF_USERNAME, "") if values else "",
        ),
        ConfigEntry(
            key=CONF_CLIENTKEY,
            type=ConfigEntryType.SECURE_STRING,
            hidden=True,
            required=False,
            default_value="",
            value=values.get(CONF_CLIENTKEY, "") if values else "",
        ),
        ConfigEntry(
            key=CONF_PERLIGHT_BRIGHTNESS_DATA,
            type=ConfigEntryType.STRING,
            hidden=True,
            required=False,
            default_value=DEFAULT_PERLIGHT_BRIGHTNESS_DATA,
        ),
        ConfigEntry(
            key=CONF_PALETTE_ROTATE,
            type=ConfigEntryType.BOOLEAN,
            hidden=True,
            required=False,
            default_value=DEFAULT_PALETTE_ROTATE,
        ),
        ConfigEntry(
            key=CONF_PALETTE_ROTATE_LIST,
            type=ConfigEntryType.STRING,
            multi_value=True,
            hidden=True,
            required=False,
            default_value=[],
        ),
        ConfigEntry(
            key=CONF_PALETTE_ROTATE_BEATS,
            type=ConfigEntryType.INTEGER,
            hidden=True,
            required=False,
            default_value=DEFAULT_PALETTE_ROTATE_BEATS,
        ),
        # ---- Base effects ----
        ConfigEntry(
            key="divider_base",
            type=ConfigEntryType.DIVIDER,
            category="settings",
        ),
        ConfigEntry(
            key=CONF_BRIGHTNESS,
            type=ConfigEntryType.INTEGER,
            default_value=100,
            range=(0, 100),
            immediate_apply=True,
            category="settings",
        ),
        ConfigEntry(
            key=CONF_COLOR_MODE,
            type=ConfigEntryType.STRING,
            default_value=DEFAULT_COLOR_MODE,
            options=[ConfigValueOption(mode, title=mode.capitalize()) for mode in COLOR_MODES],
            immediate_apply=True,
            category="settings",
        ),
        ConfigEntry(
            key=CONF_PALETTE,
            type=ConfigEntryType.STRING,
            default_value=DEFAULT_PALETTE,
            options=[
                ConfigValueOption(""),
                *(ConfigValueOption(name, title=name) for name in palette_names()),
            ],
            immediate_apply=True,
            category="settings",
        ),
        ConfigEntry(
            key=CONF_PALETTE_ROTATE_SMOOTH,
            type=ConfigEntryType.BOOLEAN,
            default_value=DEFAULT_PALETTE_ROTATE_SMOOTH,
            immediate_apply=True,
            category="settings",
        ),
        # ---- Pulse / Club fire engine ----
        ConfigEntry(
            key="divider_pulse",
            type=ConfigEntryType.DIVIDER,
            category="settings",
        ),
        ConfigEntry(
            key=CONF_PULSE_SELECT,
            type=ConfigEntryType.STRING,
            default_value=DEFAULT_PULSE_SELECT,
            options=[ConfigValueOption(value) for value, _ in PULSE_SELECT_OPTIONS],
            immediate_apply=True,
            category="settings",
        ),
        ConfigEntry(
            key=CONF_PULSE_DOWNBEAT,
            type=ConfigEntryType.BOOLEAN,
            default_value=DEFAULT_PULSE_DOWNBEAT,
            immediate_apply=True,
            category="settings",
        ),
        ConfigEntry(
            key=CONF_PULSE_DECAY,
            type=ConfigEntryType.INTEGER,
            default_value=DEFAULT_PULSE_DECAY,
            range=(50, 100),
            immediate_apply=True,
            category="settings",
        ),
        ConfigEntry(
            key=CONF_PULSE_FLOOR,
            type=ConfigEntryType.INTEGER,
            default_value=DEFAULT_PULSE_FLOOR,
            range=(0, 40),
            immediate_apply=True,
            category="settings",
        ),
        # ---- Strobe overlay ----
        ConfigEntry(
            key="divider_strobe",
            type=ConfigEntryType.DIVIDER,
            category="settings",
        ),
        ConfigEntry(
            key=CONF_STROBE_LIGHTS,
            type=ConfigEntryType.STRING,
            multi_value=True,
            default_value=[],
            required=False,
            options=strobe_options,
            category="settings",
        ),
        ConfigEntry(
            key=CONF_STROBE_COVERAGE,
            type=ConfigEntryType.INTEGER,
            default_value=DEFAULT_STROBE_COVERAGE,
            range=(5, 100),
            immediate_apply=True,
            category="settings",
            depends_on=CONF_STROBE_LIGHTS,
        ),
        ConfigEntry(
            key=CONF_STROBE_SENSITIVITY,
            type=ConfigEntryType.INTEGER,
            default_value=DEFAULT_STROBE_SENSITIVITY,
            range=(0, 100),
            immediate_apply=True,
            category="settings",
            depends_on=CONF_STROBE_LIGHTS,
        ),
        ConfigEntry(
            key=CONF_STROBE_ENABLED,
            type=ConfigEntryType.BOOLEAN,
            default_value=DEFAULT_STROBE_ENABLED,
            immediate_apply=True,
            category="settings",
            depends_on=CONF_STROBE_LIGHTS,
        ),
        ConfigEntry(
            key=CONF_STROBE_AUTO,
            type=ConfigEntryType.BOOLEAN,
            default_value=DEFAULT_STROBE_AUTO,
            immediate_apply=True,
            category="settings",
            depends_on=CONF_STROBE_LIGHTS,
        ),
        ConfigEntry(
            key=CONF_STROBE_BLACKOUT,
            type=ConfigEntryType.BOOLEAN,
            default_value=DEFAULT_STROBE_BLACKOUT,
            immediate_apply=True,
            category="settings",
            depends_on=CONF_STROBE_LIGHTS,
        ),
        ConfigEntry(
            key=CONF_STROBE_COLOR,
            type=ConfigEntryType.STRING,
            default_value=DEFAULT_STROBE_COLOR,
            options=[
                ConfigValueOption(hex_value, title=name) for hex_value, name in STROBE_COLOR_OPTIONS
            ],
            immediate_apply=True,
            category="settings",
            depends_on=CONF_STROBE_LIGHTS,
        ),
        ConfigEntry(
            key=CONF_STROBE_BRIGHTNESS,
            type=ConfigEntryType.INTEGER,
            default_value=DEFAULT_STROBE_BRIGHTNESS,
            range=(1, 100),
            immediate_apply=True,
            category="settings",
            depends_on=CONF_STROBE_LIGHTS,
        ),
        ConfigEntry(
            key=CONF_STROBE_FLASH_HZ,
            type=ConfigEntryType.INTEGER,
            default_value=DEFAULT_STROBE_FLASH_HZ,
            range=(1, 25),
            immediate_apply=True,
            category="settings",
            depends_on=CONF_STROBE_LIGHTS,
        ),
        ConfigEntry(
            key=CONF_STROBE_DUTY,
            type=ConfigEntryType.INTEGER,
            default_value=DEFAULT_STROBE_DUTY,
            range=(5, 90),
            immediate_apply=True,
            category="settings",
            depends_on=CONF_STROBE_LIGHTS,
        ),
        ConfigEntry(
            key=CONF_STROBE_MIN_HOLD_MS,
            type=ConfigEntryType.INTEGER,
            default_value=DEFAULT_STROBE_MIN_HOLD_MS,
            range=(0, 3000),
            immediate_apply=True,
            category="settings",
            depends_on=CONF_STROBE_LIGHTS,
        ),
        ConfigEntry(
            key=CONF_STROBE_RELEASE_MS,
            type=ConfigEntryType.INTEGER,
            default_value=DEFAULT_STROBE_RELEASE_MS,
            range=(0, 3000),
            immediate_apply=True,
            category="settings",
            depends_on=CONF_STROBE_LIGHTS,
        ),
        ConfigEntry(
            key=CONF_STROBE_BEAT_SYNC,
            type=ConfigEntryType.BOOLEAN,
            default_value=DEFAULT_STROBE_BEAT_SYNC,
            immediate_apply=True,
            category="settings",
            depends_on=CONF_STROBE_LIGHTS,
        ),
        # ---- Debug & tuning ----
        ConfigEntry(
            key="divider_debug",
            type=ConfigEntryType.DIVIDER,
            category="settings",
        ),
        ConfigEntry(
            key=CONF_HUE_LATENCY_MS,
            type=ConfigEntryType.INTEGER,
            default_value=DEFAULT_HUE_LATENCY_MS,
            range=(0, 3000),
            immediate_apply=True,
            category="settings",
        ),
    ]

    if preview_url:
        entries.append(
            ConfigEntry(
                key="live_preview",
                type=ConfigEntryType.LABEL,
                help_link=preview_url,
                category="settings",
            )
        )

    return tuple(entries)
