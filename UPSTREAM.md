# Sources

[PickScore authors](https://github.com/yuvalkirstain/PickScore/tree/f79607ff68609c1e6dfa78cba38afeea4c248df1) (MIT). The CLIPModel image/text cosine recipe is reproduced using the official Transformers loader. This repository adds frozen clause aggregation, training-pair differences, logistic fitting, data selection and independent held-out evaluation. No upstream scoring code files or weights are redistributed.

[Official PickScore v1 model](https://huggingface.co/yuvalkirstain/PickScore_v1/tree/a4e4367c6dfa7288a00c550414478f865b875800). The fixed checkpoint is fetched into the caller's cache and checked against its Hugging Face LFS SHA-256.

[Pick-a-Pic anonymous v1 data](https://huggingface.co/datasets/pickapic-anonymous/pickapic_v1/tree/8e69b58d729339cb7771133c70ee18b527fd4ae3). Only label and derived numeric results are intended for shipment. Original photographs, model weights and user identifiers remain outside the repo. A fixed viewer split, offsets, checksums and the dataset revision identify the experiment.
