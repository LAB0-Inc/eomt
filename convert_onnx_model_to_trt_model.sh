/usr/src/tensorrt/bin/trtexec \
  --onnx=checkpoints/ONNX/eomt_large_640.onnx \
  --saveEngine=checkpoints/TensorRT_EoMT/model_fp16.engine \
  --fp16 \
  --stronglyTyped  # This is necessary, otherwise we get NaNs in the output.

# /usr/src/tensorrt/bin/trtexec \
#   --onnx=eomt_large_640.onnx \
#   --saveEngine=model_int8.engine \
#   --int8 \
#   --best