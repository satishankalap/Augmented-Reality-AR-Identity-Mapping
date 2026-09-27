"""
Wires state_machine + embedding + matching_api + render_bridge into the
actual on-demand identification loop described in the spec:

  gaze hit -> dwell -> single frame capture -> vector match -> anchored card

This is the class the real glasses app instantiates once per session. Feed it
gaze events from whatever sensor API the hardware provides; it calls the
render bridge at each step and the matching API exactly once per successful
dwell — never a continuous stream.
"""
from __future__ import annotations

from .embedding import extract as extract_embedding
from .matching_api import MatchingAPIClient, MatchingAPIError
from .render_bridge import RenderBridge
from .state_machine import GazeStateMachine, State


class GazePipeline:
    def __init__(
        self,
        event_id: str,
        matching_client: MatchingAPIClient,
        render_bridge: RenderBridge,
    ):
        self.event_id = event_id
        self.matching_client = matching_client
        self.render = render_bridge
        self.sm = GazeStateMachine(on_state_change=self._on_state_change)
        self._current_target: str | None = None

    def _on_state_change(self, old: State, new: State) -> None:
        if new == State.STANDBY and self._current_target:
            self.render.clear(self._current_target)
            self._current_target = None

    def on_gaze_hit(self, target_id: str, frame_id: str) -> None:
        """Called every sensor tick the gaze ray intersects target_id's bounding box."""
        self._current_target = target_id
        prev_state = self.sm.state
        self.sm.gaze_hit(target_id)

        if self.sm.state == State.TARGETING:
            # progress is cosmetic here — a real client tracks dwell start
            # time itself to compute this; kept simple for the demo.
            self.render.show_targeting_ring(target_id, progress_pct=50.0)

        if prev_state != State.MATCHING and self.sm.state == State.MATCHING:
            self._run_match(target_id, frame_id)

    def on_gaze_lost(self) -> None:
        self.sm.gaze_lost()

    def on_target_walked_away(self, target_id: str) -> None:
        if target_id == self._current_target:
            self.sm.target_walked_away()

    def _run_match(self, target_id: str, frame_id: str) -> None:
        # Single frame, single embedding call, single API call — this is the
        # only point in the whole loop where the camera and network are used.
        self.render.show_matching_indicator(target_id)
        embedding = extract_embedding(frame_id)
        try:
            results = self.matching_client.match(self.event_id, embedding, top_k=1)
        except MatchingAPIError:
            self.sm.match_failed_or_timed_out()
            return

        if not results:
            # Spec: fails silently, no error UI.
            self.sm.match_failed_or_timed_out()
            return

        top = results[0]
        self.sm.match_succeeded()
        self.render.show_anchored_card(target_id, top.name, top.role, top.company)
