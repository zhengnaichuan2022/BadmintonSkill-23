# CoM-M2U 

## 
- RGB: r2plus1d full+Kinetics
- Skeleton: protogcn 
- IMU: ispl_inception full_w256
- :  `.md`（leave-one-out + conf fusion + gain weight + stopgrad）

## 
- `multimodal/com_m2u/`：loss / export / finetune
- ：`multimodal/run_com_m2u.sh`

## 
|  |  | Baseline | CoM-M2U |  |
|------|------|----------|---------|------|
| xsub | IMU | 58.27% | **59.11%** | +0.84 |
| xsub | RGB | 55.96% |  | |
| xsub | Skeleton 4s | 67.84% |  | |
| xacq | * | | logits  | |

## 
```bash
# IMU xacq（ rgb logits）
PYTHONPATH=multimodal CUDA_VISIBLE_DEVICES=2 \
  python -m com_m2u.finetune_imu_simple --protocol xacq --mode com_m2u \
  --epochs 12 --lambda_kd 0.3 --gamma 2.0 --lr 3e-5 --gpu 0

# Skeleton 
PYTHONPATH=multimodal:skeleton/pyskl-main/pyskl-main \
  python -m com_m2u.finetune_skeleton --protocol xsub --mode com_m2u \
  --epochs 6 --lambda_kd 0.3 --gamma 2.0 --streams j,b,jm,bm

# RGB
PYTHONPATH=multimodal:video:$MMACTION2_ROOT \
  $MMACTION_PYTHON -m com_m2u.finetune_video --protocol xsub --auto_search
```
