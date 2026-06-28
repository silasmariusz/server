"""
Hue Lights Sync provider for Music Assistant.

Discovers entertainment areas on a paired Hue bridge and creates
virtual Sendspin players for each. When music plays to a virtual player,
the lights in that entertainment area react to the music.
"""

from __future__ import annotations

import hmac
import json
import logging
import secrets
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from aiohttp import web
from hue_entertainment import HueEntertainmentAPI
from zeroconf import ServiceStateChange

from music_assistant.models.plugin import PluginProvider

from .analyzer import PulseSettings
from .bridge import HueEntertainmentBridgeManager
from .calibration import Calibrator
from .constants import (
    COLOR_MODES,
    CONF_BRIDGE_HOST,
    CONF_BRIDGE_ID,
    CONF_BRIGHTNESS,
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
    DEFAULT_PALETTE_ROTATE_BEATS,
    DEFAULT_PALETTE_ROTATE_SMOOTH,
)
from .palettes import palettes_with_colors
from .preview import PreviewServer
from .strobe_overlay import StrobeSettings

if TYPE_CHECKING:
    from hue_entertainment import EntertainmentArea, LightChannel, LightColorCommand
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.enums import ProviderFeature
    from music_assistant_models.provider import ProviderManifest
    from zeroconf.asyncio import AsyncServiceInfo

    from music_assistant.mass import MusicAssistant

LOGGER = logging.getLogger(__name__)


class HueEntertainmentProvider(PluginProvider):
    """Provider that syncs Hue lights to music via Sendspin."""

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
        supported_features: set[ProviderFeature],
    ) -> None:
        """Initialize the provider."""
        super().__init__(mass, manifest, config, supported_features)
        self._hue_api: HueEntertainmentAPI | None = None
        self._bridge_manager: HueEntertainmentBridgeManager | None = None
        self._preview = PreviewServer(self)
        self._calibrator = Calibrator(self)
        self._preview_routes: list[Callable[[], None]] = []
        # Unguessable capability token embedded in the served preview page; the
        # state-changing preview endpoints require it so they can't be driven by
        # an arbitrary client that can merely reach the webserver.
        self._preview_token = secrets.token_urlsafe(24)

    @property
    def hue_api(self) -> HueEntertainmentAPI | None:
        """Return the Hue API client."""
        return self._hue_api

    @property
    def entertainment_areas(self) -> list[EntertainmentArea]:
        """Entertainment areas with a live bridge, for building config selectors."""
        return self._bridge_manager.areas if self._bridge_manager else []

    @property
    def bridge_manager(self) -> HueEntertainmentBridgeManager | None:
        """The active bridge manager, or None until the provider is loaded."""
        return self._bridge_manager

    def preview_event(self, payload: dict[str, Any]) -> None:
        """Push a custom event (e.g. a calibration tick) to live preview clients."""
        self._preview.publish_event(payload)

    @property
    def preview_token(self) -> str:
        """Capability token embedded in the preview page (gates state changes)."""
        return self._preview_token

    def authorize_preview(self, request: web.Request) -> bool:
        """
        Check the preview capability token on a state-changing preview request.

        :param request: The incoming preview HTTP request.
        """
        sent = request.headers.get("x-hue-preview-token", "")
        return hmac.compare_digest(sent, self._preview_token)

    async def serve_perlight(self, request: web.Request) -> web.Response:
        """Get or set the per-light base-brightness map (driven by the preview sliders)."""
        try:
            data = await request.json()
        except ValueError, json.JSONDecodeError:
            data = {}
        raw = str(self.config.get_value(CONF_PERLIGHT_BRIGHTNESS_DATA) or "")
        try:
            mapping = json.loads(raw) if raw else {}
        except ValueError, json.JSONDecodeError:
            mapping = {}
        if not isinstance(mapping, dict):
            mapping = {}
        if str(data.get("action", "")) == "set":
            if not self.authorize_preview(request):
                return web.json_response({"ok": False, "error": "unauthorized"}, status=403)
            area = str(data.get("area", ""))
            channel = data.get("channel")
            pct = data.get("pct")
            if area and channel is not None and pct is not None:
                try:
                    key = f"{area}:{int(channel)}"
                    value = max(0, min(100, int(pct)))
                except TypeError, ValueError:
                    return web.json_response({"ok": False}, status=400)
                # 100% is the default, so don't persist it (keeps the blob small).
                if value >= 100:
                    mapping.pop(key, None)
                else:
                    mapping[key] = value
                new_raw = json.dumps(mapping)
                self._update_config_value(CONF_PERLIGHT_BRIGHTNESS_DATA, new_raw)
                if self._bridge_manager:
                    self._bridge_manager.update_settings(per_light_data=new_raw)
        return web.json_response({"perlight": mapping})

    async def serve_palette(self, request: web.Request) -> web.Response:
        """List palettes / report the selection, or set the active palette (preview dropdown)."""
        try:
            data = await request.json()
        except ValueError, json.JSONDecodeError:
            data = {}
        action = str(data.get("action", ""))
        if action in ("set", "rotate") and not self.authorize_preview(request):
            return web.json_response({"ok": False, "error": "unauthorized"}, status=403)
        if action == "set":
            name = str(data.get("palette", ""))
            self._update_config_value(CONF_PALETTE, name)
            if self._bridge_manager:
                self._bridge_manager.update_settings(palette=name)
        elif action == "rotate":
            try:
                enabled = bool(data.get("enabled"))
                names = [str(n) for n in (data.get("list") or [])]
                beats = max(4, int(data.get("beats", DEFAULT_PALETTE_ROTATE_BEATS)))
            except TypeError, ValueError:
                return web.json_response({"ok": False}, status=400)
            smooth_val = self.config.get_value(CONF_PALETTE_ROTATE_SMOOTH)
            smooth = DEFAULT_PALETTE_ROTATE_SMOOTH if smooth_val is None else bool(smooth_val)
            self._update_config_value(CONF_PALETTE_ROTATE, enabled)
            self._update_config_value(CONF_PALETTE_ROTATE_LIST, names)
            self._update_config_value(CONF_PALETTE_ROTATE_BEATS, beats)
            if self._bridge_manager:
                self._bridge_manager.set_rotation(enabled, names, beats, smooth)
        return web.json_response(
            {
                "palettes": palettes_with_colors(),
                "selected": str(self.config.get_value(CONF_PALETTE) or ""),
                "rotate": {
                    "enabled": bool(self.config.get_value(CONF_PALETTE_ROTATE)),
                    "list": cast(
                        "list[str]", self.config.get_value(CONF_PALETTE_ROTATE_LIST) or []
                    ),
                    "beats": int(
                        float(
                            str(
                                self.config.get_value(CONF_PALETTE_ROTATE_BEATS)
                                or DEFAULT_PALETTE_ROTATE_BEATS
                            )
                        )
                    ),
                },
            }
        )

    async def loaded_in_mass(self) -> None:
        """Initialize Hue bridge connection and set up entertainment area bridges."""
        # Register the live preview web routes. Pre-unregister makes this safe to
        # re-run across provider reloads (register raises if a key already exists).
        try:
            base = "/hue_preview"
            for path, handler in (
                (base, self._preview.serve_page),
                (f"{base}/stream", self._preview.serve_stream),
                (f"{base}/calib.mp3", self._calibrator.serve_track),
                (f"{base}/control", self._calibrator.serve_control),
                (f"{base}/perlight", self.serve_perlight),
                (f"{base}/palette", self.serve_palette),
            ):
                self.mass.webserver.unregister_dynamic_route(path)
                self._preview_routes.append(
                    self.mass.webserver.register_dynamic_route(path, handler)
                )
            self.logger.info(
                "Hue live preview available at %s%s", self.mass.webserver.base_url, base
            )
        except Exception as err:
            self.logger.warning("Could not register Hue live preview routes: %s", err)

        # Migrate orphaned color_mode values from older versions to the default
        # so the settings dropdown shows a valid option.
        stored_mode = self.config.get_value(CONF_COLOR_MODE)
        if stored_mode is not None and str(stored_mode) not in COLOR_MODES:
            self._update_config_value(CONF_COLOR_MODE, DEFAULT_COLOR_MODE)

        host = self.config.get_value(CONF_BRIDGE_HOST)
        username = self.config.get_value(CONF_USERNAME)

        if not host or not username:
            self.logger.warning("Hue bridge not configured, provider inactive")
            self.available = False
            return

        self._hue_api = HueEntertainmentAPI(str(host), str(username))
        self._bridge_manager = HueEntertainmentBridgeManager(self)

        # Fetch entertainment areas and set up bridges
        try:
            areas = await self._hue_api.get_entertainment_areas()
            if not areas:
                self.logger.warning("No entertainment areas found on Hue bridge at %s", host)
            else:
                self.logger.info(
                    "Found %d entertainment area(s) on Hue bridge: %s",
                    len(areas),
                    ", ".join(a.name for a in areas),
                )
            await self._bridge_manager.setup_bridges(areas)
            self.available = True
        except Exception as err:
            self.logger.error("Failed to initialize Hue bridge at %s: %s", host, err)
            self.available = False

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/close of the provider."""
        for remove_route in self._preview_routes:
            remove_route()
        self._preview_routes.clear()
        self._preview.close()
        self._calibrator.close()
        if self._bridge_manager:
            await self._bridge_manager.stop_all()
            self._bridge_manager = None
        if self._hue_api:
            await self._hue_api.close()
            self._hue_api = None

    def preview_register_area(self, area_id: str, name: str, channels: list[LightChannel]) -> None:
        """
        Register an entertainment area with the live preview legend.

        :param area_id: The Hue entertainment area id.
        :param name: Human readable area name.
        :param channels: The ordered light channels in the area.
        """
        self._preview.register_area(area_id, name, channels)

    def preview_publish(self, area_id: str, commands: list[LightColorCommand]) -> None:
        """
        Mirror a rendered frame to any connected live preview clients.

        :param area_id: The area the frame belongs to.
        :param commands: The exact color commands sent to the bridge this tick.
        """
        self._preview.publish(area_id, commands)

    async def on_mdns_service_state_change(
        self, name: str, state_change: ServiceStateChange, info: AsyncServiceInfo | None
    ) -> None:
        """
        Handle mDNS service discovery for Hue bridges.

        Updates the bridge IP address if it changes (e.g. DHCP renewal).
        """
        if info is None:
            return

        # Extract the bridge ID from the mDNS service name
        raw_bridge_id = info.properties.get(b"bridgeid", b"")
        bridge_id = raw_bridge_id.decode("utf-8", errors="ignore") if raw_bridge_id else ""
        if not bridge_id:
            return

        configured_bridge_id = self.config.get_value(CONF_BRIDGE_ID) or ""

        if state_change == ServiceStateChange.Removed:
            if bridge_id == configured_bridge_id:
                self.logger.info("Hue bridge %s removed from network", bridge_id)
                self.available = False
            return

        # Extract IP address from service info
        addresses = info.parsed_addresses()
        if not addresses:
            return
        new_host = addresses[0]

        if state_change == ServiceStateChange.Added:
            # If we don't have a bridge ID configured yet, and no host is set,
            # this is likely the initial discovery during setup
            if not configured_bridge_id:
                self.logger.debug(
                    "Discovered Hue bridge %s at %s (not yet configured)", bridge_id, new_host
                )
                return

        if bridge_id != configured_bridge_id:
            return

        # Update the host if it changed
        current_host = self.config.get_value(CONF_BRIDGE_HOST) or ""
        if new_host != current_host:
            self.logger.info(
                "Hue bridge %s IP changed from %s to %s",
                bridge_id,
                current_host,
                new_host,
            )
            if self._hue_api:
                self._hue_api.host = new_host
            # Persist the new IP
            self._update_config_value(CONF_BRIDGE_HOST, new_host)

        if not self.available:
            # Re-initialize bridges if we were previously unavailable, and only flip
            # back to available once setup actually succeeds (mirrors loaded()).
            if self._hue_api and self._bridge_manager:
                try:
                    areas = await self._hue_api.get_entertainment_areas()
                    await self._bridge_manager.setup_bridges(areas)
                    self.available = True
                except Exception as err:
                    self.logger.warning("Failed to reinitialize bridges: %s", err)
            else:
                self.available = True

    async def update_config(self, config: ProviderConfig, changed_keys: set[str]) -> None:
        """
        Handle config changes.

        :param config: The updated provider configuration.
        :param changed_keys: Changed config keys; value keys are ``values/<key>`` prefixed.
        """
        self.config = config

        # Keys that can be pushed to the running bridges without a full reload.
        # The config controller emits changed_keys prefixed with "values/", so the
        # match set must use the same prefix (a bare CONF_* set never intersects).
        live_keys = {
            f"values/{key}"
            for key in (
                CONF_BRIGHTNESS,
                CONF_COLOR_MODE,
                CONF_HUE_LATENCY_MS,
                CONF_PALETTE,
                CONF_PERLIGHT_BRIGHTNESS_DATA,
                CONF_STROBE_LIGHTS,
                CONF_STROBE_COVERAGE,
                CONF_STROBE_SENSITIVITY,
                CONF_STROBE_ENABLED,
                CONF_STROBE_BLACKOUT,
                CONF_STROBE_COLOR,
                CONF_STROBE_BRIGHTNESS,
                CONF_STROBE_FLASH_HZ,
                CONF_STROBE_DUTY,
                CONF_STROBE_MIN_HOLD_MS,
                CONF_STROBE_RELEASE_MS,
                CONF_STROBE_BEAT_SYNC,
                CONF_STROBE_AUTO,
                CONF_PALETTE_ROTATE,
                CONF_PALETTE_ROTATE_LIST,
                CONF_PALETTE_ROTATE_BEATS,
                CONF_PALETTE_ROTATE_SMOOTH,
                CONF_PULSE_FLOOR,
                CONF_PULSE_DECAY,
                CONF_PULSE_SELECT,
                CONF_PULSE_DOWNBEAT,
            )
        }
        if self._bridge_manager and (changed_keys & live_keys):
            self.logger.debug("Applying live settings update: %s", changed_keys & live_keys)
            self._bridge_manager.update_settings(
                color_mode=str(config.get_value(CONF_COLOR_MODE) or "smooth"),
                brightness=int(float(str(config.get_value(CONF_BRIGHTNESS) or 100))),
                hue_latency_ms=int(
                    float(str(config.get_value(CONF_HUE_LATENCY_MS) or DEFAULT_HUE_LATENCY_MS))
                ),
                strobe_selection=cast("list[str]", config.get_value(CONF_STROBE_LIGHTS) or []),
                strobe=StrobeSettings.from_config(config),
                palette=str(config.get_value(CONF_PALETTE) or ""),
                per_light_data=str(config.get_value(CONF_PERLIGHT_BRIGHTNESS_DATA) or ""),
                pulse=PulseSettings.from_config(config),
            )

        # Palette rotation feeds set_rotation (not update_settings), so re-push it on
        # any rotation key change. These keys are in live_keys so a preview-driven
        # change applies here without falling through to a full provider reload.
        rotation_keys = {
            f"values/{key}"
            for key in (
                CONF_PALETTE_ROTATE,
                CONF_PALETTE_ROTATE_LIST,
                CONF_PALETTE_ROTATE_BEATS,
                CONF_PALETTE_ROTATE_SMOOTH,
            )
        }
        if self._bridge_manager and (changed_keys & rotation_keys):
            rotate_smooth = config.get_value(CONF_PALETTE_ROTATE_SMOOTH)
            self._bridge_manager.set_rotation(
                bool(config.get_value(CONF_PALETTE_ROTATE)),
                cast("list[str]", config.get_value(CONF_PALETTE_ROTATE_LIST) or []),
                int(
                    float(
                        str(
                            config.get_value(CONF_PALETTE_ROTATE_BEATS)
                            or DEFAULT_PALETTE_ROTATE_BEATS
                        )
                    )
                ),
                DEFAULT_PALETTE_ROTATE_SMOOTH if rotate_smooth is None else bool(rotate_smooth),
            )

        # Defer remaining keys (host, credentials, log level, name) to the base
        # implementation, which reloads the provider only when actually required.
        remaining = changed_keys - live_keys
        if remaining:
            await super().update_config(config, remaining)
