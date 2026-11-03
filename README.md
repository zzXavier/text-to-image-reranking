# Text-to-image preference reranking

This project compares four ways of choosing between two generated images for a given prompt: always picking the first image, whole-prompt cosine, the mean score over natural-language clauses, and a preference head trained on the frozen scores. Human votes from Pick-a-Pic are the target. See `HANDOFF.md` for the latest verification status.

## Run

```shell
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python run.py reproduce --device cpu
python run.py evaluate
```

`reproduce` downloads the pinned 3.95 GB PickScore checkpoint and a 36-prompt public preference subset, extracts real image/text embeddings, trains a four-feature pairwise logistic head, reloads it and evaluates the 12 held-out human decisions. Use `--cache` to share the model/data across repos and `--out` for a fresh output directory. When a slow connection interrupts the weights download, `python tools/range_model.py .cache/PickScore` resumes the large tensor file in independently cached ranges. `demo` refits the head from the shipped anonymized feature records once a full extraction has been recorded.

## Experiment

The training set uses 24 unique records from `validation_unique`, and the held-out set uses 12 records from `test_unique`. Both come from the public anonymous Pick-a-Pic v1 release, whose original training split is excluded from this experiment. Sampling removes ties, duplicate captions and reused candidate IDs across both arms. Image bytes remain in the local cache. The data manifest contains ranking IDs, split assignments, prompts and checksums, with user identities omitted.

`src/scoring.py` reuses the normalized frozen image and text similarity from PickScore. New code splits on comma, semicolon or simple conjunctions and computes the whole-prompt cosine plus mean, minimum and standard deviation of clause scores. These are text-clause heuristics. Each query has two candidates and one human preference. Per-object counting, exact spatial localization and a general preference win rate need separate evidence.

The fitted linear weights can be consumed by P12. The formal run writes a reusable head, held-out decisions, summary CSV and comparison plot into `results/`. `evaluate` reconstructs the summary and figure from the saved decisions.
