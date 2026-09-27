"""
Computes real false-accept-rate (FAR) / false-reject-rate (FRR) numbers from
a dataset captured with capture_dataset.py — this is the number that was
genuinely unverifiable in the sandbox (no camera, no real faces).

Genuine pairs = two images of the same label (should match: high similarity).
Impostor pairs = two images of different labels (should NOT match: low similarity).

Sweeps candidate `min_similarity` thresholds and reports FAR/FRR at each, plus
the threshold closest to equal-error-rate (FAR == FRR) as a starting
recommendation for Phase 2's /match `min_similarity` parameter.

Usage:
    python3 scripts/evaluate_embeddings.py --dataset dataset/
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

_SIBLING = Path(__file__).resolve().parents[2] / "ar-phase3-glasses-client"
if _SIBLING.exists():
    sys.path.insert(0, str(_SIBLING))

try:
    from glasses_client.embedding_v2 import lbp_histogram
except ImportError:
    print(
        "Couldn't import glasses_client.embedding_v2 — make sure "
        "ar-phase3-glasses-client/ is unpacked next to this test-infra folder."
    )
    sys.exit(1)


def load_embeddings(dataset_dir: Path) -> Dict[str, List[List[float]]]:
    """dataset_dir/<label>/*.png -> {label: [embedding, embedding, ...]}"""
    by_label: Dict[str, List[List[float]]] = {}
    for label_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        embeddings = []
        for img_path in sorted(label_dir.glob("*.png")):
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            embeddings.append(lbp_histogram(gray))
        if embeddings:
            by_label[label_dir.name] = embeddings
    return by_label


def cosine(a: List[float], b: List[float]) -> float:
    a, b = np.array(a), np.array(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def build_pairs(by_label: Dict[str, List[List[float]]]) -> Tuple[List[float], List[float]]:
    genuine, impostor = [], []
    labels = list(by_label.keys())

    for label, embs in by_label.items():
        for a, b in itertools.combinations(embs, 2):
            genuine.append(cosine(a, b))

    for l1, l2 in itertools.combinations(labels, 2):
        for a in by_label[l1]:
            for b in by_label[l2]:
                impostor.append(cosine(a, b))

    return genuine, impostor


def sweep_thresholds(genuine: List[float], impostor: List[float]) -> None:
    print(f"\n{len(genuine)} genuine pairs, {len(impostor)} impostor pairs\n")
    print(f"{'threshold':>10} {'FAR':>8} {'FRR':>8}")
    best_t, best_gap = None, float("inf")
    for t in np.arange(0.50, 1.001, 0.02):
        far = sum(1 for s in impostor if s >= t) / len(impostor) if impostor else 0.0
        frr = sum(1 for s in genuine if s < t) / len(genuine) if genuine else 0.0
        print(f"{t:>10.2f} {far:>8.2%} {frr:>8.2%}")
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap, best_t = gap, t

    print(f"\nRecommended starting min_similarity (closest to equal-error-rate): {best_t:.2f}")
    print("Use this as a starting point for Phase 2's /match min_similarity, not a final answer —")
    print("re-run this with a larger, more diverse dataset before trusting it for a real pilot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset", help="Dataset root (same --out used in capture_dataset.py)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"No dataset found at {dataset_dir}. Run capture_dataset.py first.")
        sys.exit(1)

    by_label = load_embeddings(dataset_dir)
    if len(by_label) < 2:
        print(f"Found {len(by_label)} labeled person(s) — need at least 2 different people "
              f"to compute impostor pairs. Capture at least one more person.")
        sys.exit(1)

    genuine, impostor = build_pairs(by_label)
    sweep_thresholds(genuine, impostor)
