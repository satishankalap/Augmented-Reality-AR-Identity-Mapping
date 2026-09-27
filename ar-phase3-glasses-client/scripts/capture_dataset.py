"""
Captures a small labeled face dataset from your webcam, for validating
embedding_v2.py's detection + matching accuracy against real photos instead
of synthetic test images.

Run this on your own machine, not in Docker — it needs direct access to a
camera device, which containers don't have by default.

Mirrors the product's own consent rules on purpose: this script won't touch
the camera until you've explicitly agreed to the same three things the
onboarding flow asks attendees to agree to. Testing tools that skip the
consent step they're testing tend to normalize skipping it in production too.

Usage:
    python3 scripts/capture_dataset.py --label "sarah" --out dataset/
    (press 's' to save a frame, 'q' to quit)

Requires the glasses_client package on the path — this script assumes the
layout from the architecture doc:
    project/
      ar-phase3-glasses-client/glasses_client/
      ar-test-infra/scripts/capture_dataset.py   <- you are here
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Make the sibling glasses_client package importable without installing it.
_SIBLING = Path(__file__).resolve().parents[2] / "ar-phase3-glasses-client"
if _SIBLING.exists():
    sys.path.insert(0, str(_SIBLING))

try:
    from glasses_client.embedding_v2 import _get_detector
except ImportError:
    print(
        "Couldn't import glasses_client.embedding_v2 — make sure "
        "ar-phase3-glasses-client/ is unpacked next to this test-infra folder."
    )
    sys.exit(1)


CONSENT_STATEMENTS = [
    "This capture is for MY OWN testing/validation use only — not a live event with real attendees.",
    "Anyone whose face is captured (including me) has explicitly agreed to it, the same way the onboarding flow requires.",
    "I will delete this dataset when I'm done validating — it's test data, not something to keep indefinitely.",
]


def get_consent() -> bool:
    print("\nBefore this touches your camera:\n")
    for i, stmt in enumerate(CONSENT_STATEMENTS, 1):
        answer = input(f"  {i}. {stmt}\n     Type 'yes' to agree: ").strip().lower()
        if answer != "yes":
            print("\nConsent not given — exiting without touching the camera.")
            return False
    return True


def capture(label: str, out_dir: Path, max_frames: int) -> None:
    detector = _get_detector()
    out_dir = out_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.png")))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Couldn't open the webcam (device 0). Check camera permissions.")
        return

    print(f"\nCamera open. Press 's' to save a frame (need a detected face), 'q' to quit.")
    saved = existing
    while saved < existing + max_frames:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed — stopping.")
            break

        face = detector.detect_largest(frame)
        display = frame.copy()
        if face is not None:
            cv2.rectangle(display, (face.x, face.y), (face.x + face.w, face.y + face.h), (0, 255, 0), 2)
            cv2.putText(display, "face detected - press 's' to save", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "no face detected", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("capture_dataset.py — press s to save, q to quit", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and face is not None:
            path = out_dir / f"{saved:03d}.png"
            cv2.imwrite(str(path), face.crop_gray)
            saved += 1
            print(f"  saved {path} ({saved - existing}/{max_frames})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {saved - existing} new frames saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Name/ID for this person's folder under --out")
    parser.add_argument("--out", default="dataset", help="Dataset root directory")
    parser.add_argument("--max-frames", type=int, default=15, help="Frames to capture for this label")
    args = parser.parse_args()

    if not get_consent():
        sys.exit(0)

    capture(args.label, Path(args.out), args.max_frames)
