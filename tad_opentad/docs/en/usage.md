# Usage (Beiyu)

```bash
cd OpenTAD-main
bash run_prepare.sh
bash run_extract_features.sh
CUDA_VISIBLE_DEVICES=0 bash run_train.sh configs/beiyu/actionformer_rgb_xsub.py
bash run_test.sh configs/beiyu/actionformer_rgb_xsub.py
```

See [README](../../README.md) for experiment configs and feature backends.
