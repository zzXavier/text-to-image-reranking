# Handoff

Formal experiment completed on CPU using FP32. The official PickScore v1 safetensors digest was verified. The processor is the authors' pinned LAION CLIP ViT-H/14 processor. The text tower was loaded first, then released before loading the vision tower. Unit tests prove this schedule agrees numerically with the full CLIP model and exercise attribute extraction and reversed pair labels.

24 disjoint public human-rated prompts trained the logistic head and 12 held-out prompts supplied independent human votes. In original order, "first" agreed with 11/12. Mirroring all 12 pairs makes its decision agree on 12/24 presentations. Whole prompt agreed on 10/12 human pairs, and decomposed / fitted head on 8/12 each. The separate known-answer probe ranked 5 candidates per prompt; whole prompt ranked the correct one first in 3/4, attribute mean in 1/4 and fitted head in 0/4. The unfavorable comparisons are preserved.

Formal outputs: results/head.json (canonical SHA-256 b9cd5ee6aa4ca6cb749a291ddfbf3c3da56e39559b38f54fe9d0418f2fd6a597), anonymized frozen features, decisions, summary CSV, order-balanced CSV, controlled probe scores and the generated comparison images. The same head is shipped in P12 and is covered by results/head.LICENSE.

Last verified: three unit tests, real 36-pair feature extraction, actual head training/save/reload, separately run five-candidate probes, and numeric smoke demo. Next independent check from a fresh GitHub clone: run the demo, then run reproduce against the pinned model/image cache into a new output directory and compare all three summaries. The same 12 human labels underlie the mirrored-order analysis.
