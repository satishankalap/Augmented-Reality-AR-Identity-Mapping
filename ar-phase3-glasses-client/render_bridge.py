"""
Render bridge — the seam between this pipeline and whatever glasses SDK is
actually driving the display.

Per the doc's hardware compatibility notes:
  - Android XR (Samsung Galaxy Glasses, XREAL): Jetpack Compose XR's
    Glanceable Information Architecture APIs render the card and expose
    gaze-duration triggers natively — an AndroidXRRenderBridge would call
    into those instead of doing the dwell timing itself.
  - Vuzix / Magic Leap 2 / Brilliant Labs Frame: open Android-based SDKs,
    draw directly via their display APIs.
  - Meta Ray-Ban / Display glasses: mwdat-display for rendering,
    mwdat-camera's single-frame-trigger mode instead of continuous capture.

This file ships a ConsoleRenderBridge so the pipeline is fully testable
without any hardware attached. Swapping hardware means writing one new
class that implements the same three methods below — nothing in
pipeline.py, state_machine.py, or matching_api.py changes.
"""
from __future__ import annotations

from typing import Protocol


class RenderBridge(Protocol):
    def show_targeting_ring(self, target_id: str, progress_pct: float) -> None: ...
    def show_matching_indicator(self, target_id: str) -> None: ...
    def show_anchored_card(self, target_id: str, name: str, role: str | None, company: str | None) -> None: ...
    def clear(self, target_id: str) -> None: ...


class ConsoleRenderBridge:
    """Reference implementation for local dev/demo — logs what would be drawn."""

    def show_targeting_ring(self, target_id: str, progress_pct: float) -> None:
        print(f"[HUD] targeting ring on {target_id}: {progress_pct:.0f}%")

    def show_matching_indicator(self, target_id: str) -> None:
        print(f"[HUD] pulsing match indicator on {target_id}")

    def show_anchored_card(self, target_id: str, name: str, role: str | None, company: str | None) -> None:
        line2 = " · ".join(x for x in (role, company) if x)
        print(f"[HUD] card anchored on {target_id}: {name}" + (f" — {line2}" if line2 else ""))

    def clear(self, target_id: str) -> None:
        print(f"[HUD] cleared {target_id}")
