export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 main.py validate \
   -c configs/dinov3/coco/instance/eomt_large_640_our_202408.yaml \
   --ckpt_path '/workspace/eomt/checkpoints/Run5_Oregon/epoch=067-mAP=0.83.ckpt' \
   --data.path /workspace/data/Datasets/202408/ \
   --data.batch_size 1


# --ckpt_path '/workspace/eomt/checkpoints/Run2/epoch=055-mAP=0.81.ckpt' \
