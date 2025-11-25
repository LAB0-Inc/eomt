export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 main.py fit \
  -c configs/dinov3/coco/instance/eomt_large_640.yaml \
  --trainer.devices 1 \
  --trainer.accumulate_grad_batches 2 \
  --data.batch_size 1 \
  --data.path /workspace/data/Datasets/DV3_same_t_and_v/
