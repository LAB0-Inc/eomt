export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 main.py validate \
   -c configs/dinov3/coco/instance/eomt_large_640_our_202408.yaml \
   --ckpt_path '/workspace/eomt/checkpoints/FT_3/epoch=018-mAP=0.88.ckpt' \
   --data.path /workspace/data/Datasets/202507_fine_tuning_1/ \
   --data.batch_size 1

# Fine-tuned model.
# --ckpt_path '/workspace/eomt/checkpoints/FT_3/epoch=018-mAP=0.88.ckpt' \

# Base model
# --ckpt_path '/workspace/eomt/checkpoints/Run5_Oregon/epoch=067-mAP=0.83.ckpt' \

# OLD
# --ckpt_path '/workspace/eomt/checkpoints/Run2/epoch=055-mAP=0.81.ckpt' \
