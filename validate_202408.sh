export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 main.py validate \
   -c configs/dinov3/coco/instance/eomt_large_640.yaml \
   --ckpt_path '/workspace/eomt/checkpoints/Run2/epoch=055-mAP=0.81.ckpt' \
   --data.path /workspace/data/Datasets/202408/ \
   --data.batch_size 1
