export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 main.py fit \
  -c configs/dinov3/coco/instance/eomt_large_640_fine_tuning.yaml \
  --trainer.devices 1 \
  --trainer.accumulate_grad_batches 16 \
  --data.batch_size 1 \
  --data.path /workspace/data/Datasets/202507_fine_tuning_1/
