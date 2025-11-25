import tensorrt as trt
import numpy as np
import torch
import yaml
import importlib
import pycuda.driver as cuda
import pycuda.autoinit

onnx_model_file = '/workspace/eomt/checkpoints/ONNX/eomt_large_640.onnx'
trt_model_file = '/workspace/eomt/checkpoints/TensorRT_EoMT/model_fp16_new.engine'

device = 0  # GPU[0].

config_path = "/workspace/eomt/configs/dinov3/coco/instance/eomt_large_640.yaml"
data_path = "/workspace/data/Datasets/SCD_for_EoMT/"
masked_attn_enabled = False  # If it's True, the conversion to ONNX fails, we would need to look into that.
my_checkpoint = '/workspace/data/EOMT/Checkpoints/Run2/epoch=055-mAP=0.81.ckpt'  # Trained with the proper masked attention setting.

#####################
# Prepare the model #
#####################
# 1. Load the configuration.
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


# 2. Set up the data loader.
data_module_name, class_name = config["data"]["class_path"].rsplit(".", 1)
data_module = getattr(importlib.import_module(data_module_name), class_name)
data_module_kwargs = config["data"].get("init_args", {})
data = data_module(
    path=data_path,
    batch_size=1,
    num_workers=0,
    check_empty_targets=False,
    **data_module_kwargs
).setup()


# 3. Set up the model, we are basically building the Lightning module by hand.
# 3.1. Load the encoder (it is given as input when creating the network).
encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
encoder = encoder_cls(img_size=data.img_size, **encoder_cfg.get("init_args", {}))

# 3.2. Load the network.
network_cfg = config["model"]["init_args"]["network"]
network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
network = network_cls(
    masked_attn_enabled=masked_attn_enabled,
    num_classes=data.num_classes,
    encoder=encoder,
    **network_kwargs)

# 3.3. Load the Lightning module.
lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
if "stuff_classes" in config["data"].get("init_args", {}):
    model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

# 3.4. Build the model using Lightning, while loading the weights from a checkpoint.
model = lit_cls.load_from_checkpoint(
    my_checkpoint,
    img_size=data.img_size,
    num_classes=data.num_classes,
    network=network,
    **model_kwargs).eval().to(device)


############################
# 4. Get one input sample. #
############################
val_dataset = data.val_dataloader().dataset
img, target = val_dataset[0]
imgs = [img.to(device)]
img_sizes = [img.shape[-2:] for img in imgs]
transformed_imgs = model.resize_and_pad_imgs_instance_panoptic(imgs)
input_sample = transformed_imgs.cpu().numpy().astype(np.uint8)


###########################
# 5. Convert ONNX to TRT. #
###########################
import tensorrt as trt
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# Parse ONNX
print(f"Parsing ONNX from: {onnx_model_file}")
with open(onnx_model_file, "rb") as model_file:
    if not parser.parse(model_file.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError("Failed to parse ONNX file")

print("ONNX parsed successfully!")

# Configure builder
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

# CRITICAL: Keep input as UINT8, don't quantize it
input_tensor = network.get_input(0)
print(f"Input tensor: name={input_tensor.name}, dtype={input_tensor.dtype}, shape={input_tensor.shape}")

# Explicitly set input to stay as INT8/UINT8
if input_tensor.dtype == trt.int8 or input_tensor.dtype == trt.uint8:
    print("Input is already INT8/UINT8 - preserving precision")
    # Don't convert input layer to FP16
    input_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
else:
    print(f"Warning: Input tensor is {input_tensor.dtype}, expected INT8/UINT8")

# Build engine
print("Building engine... (this may take several minutes)")
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    raise RuntimeError("Failed to build engine!")

# Save engine
with open(trt_model_file, "wb") as f:
    f.write(serialized_engine)

print(f"Engine built and saved to: {trt_model_file}")