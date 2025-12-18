export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 main.py validate \
   -c configs/dinov3/coco/instance/eomt_large_640_validate_on_ft_training.yaml \
   --ckpt_path '/workspace/eomt/checkpoints/FT1_3/epoch=018-mAP=0.88.ckpt' \
   --data.path /workspace/data/Datasets/fine_tuning_2025_12_16/ \
   --data.batch_size 1

# Fine-tuned model.
# --ckpt_path '/workspace/eomt/checkpoints/FT_3/epoch=018-mAP=0.88.ckpt' \

# Base model
# --ckpt_path '/workspace/eomt/checkpoints/Run5_Oregon/epoch=067-mAP=0.83.ckpt' \

# OLD
# --ckpt_path '/workspace/eomt/checkpoints/Run2/epoch=055-mAP=0.81.ckpt' \
