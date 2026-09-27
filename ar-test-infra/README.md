# Test Environment

The two things the sandbox this project was built in genuinely couldn't do:
install FAISS (no network) and validate face detection on real photos (no
camera). This folder does both, on a machine that has them.

## Expected layout

```
project/
  ar-phase2-backend/          <- from ar-phase2-backend.zip
  ar-phase3-glasses-client/   <- from ar-phase3-glasses-client.zip
  ar-test-infra/              <- this bundle
    Dockerfile
    docker-compose.yml
    scripts/
      capture_dataset.py
      evaluate_embeddings.py
```

Copy `Dockerfile` and `docker-compose.yml` into `ar-phase2-backend/` (they
expect to sit next to that project's own `app/` folder), or adjust the build
context path in `docker-compose.yml` if you'd rather leave them here.

## 1. Bring the backend up with real network access

```bash
cd ar-phase2-backend
docker compose up --build
```

This installs everything in `requirements.txt` plus `faiss-cpu` — the one
package that couldn't install in the sandbox. Backend comes up on
`localhost:8000`.

## 2. Switch to the FAISS backend

In `app/vector_store.py`:

```python
# before
from .vector_index_v2 import NumpyIndex
IndexBackend = NumpyIndex

# after
from .vector_index_v2 import FaissIndex
IndexBackend = FaissIndex
```

Docker's live-reload volume mount picks this up without a rebuild.

## 3. Capture a real face dataset

Run this on your host machine, not in Docker — it needs direct webcam access:

```bash
cd ar-test-infra
python3 scripts/capture_dataset.py --label sarah --max-frames 15
python3 scripts/capture_dataset.py --label marcus --max-frames 15
```

It won't touch the camera until you've explicitly agreed to three consent
statements in the terminal — deliberately mirroring the product's own
onboarding rules. Capture at least 2 different people so there's something
to compute impostor (different-person) pairs against.

## 4. Get real FAR/FRR numbers

```bash
python3 scripts/evaluate_embeddings.py --dataset dataset/
```

This is the number that was flagged as unverified in every prior README —
actual false-accept-rate and false-reject-rate at a sweep of thresholds,
computed from real captured faces, with a suggested starting
`min_similarity` at the equal-error-rate point.

**Sanity-checked before shipping:** the FAR/FRR math itself was tested
against fabricated clustered embeddings (tight same-person clusters, wide
separation between different people) and behaves correctly — FAR and FRR
both hit 0% in the comfortable middle of the threshold range and FRR
correctly spikes to 100% only at an impossible threshold of 1.0. What
still depends on you running it: whether *real* LBP embeddings from *real*
faces cluster this cleanly, which is exactly what this script measures.

## 5. Feed the result back into the product

Whatever threshold `evaluate_embeddings.py` recommends becomes the
`min_similarity` default in:
- Phase 2's `POST /match` endpoint (`app/models.py`'s `MatchQuery.min_similarity`)
- Phase 3's pipeline, wherever it calls `matching_client.match(...)`

Re-run this evaluation any time the embedding method changes (e.g. after
swapping in a real deep-learning model via `OnnxEmbeddingExtractor`) — the
right threshold for LBP histograms and the right threshold for a deep
embedding are not the same number.
