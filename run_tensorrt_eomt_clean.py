import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

import yaml
from lightning import seed_everything
import numpy as np
import warnings
import importlib
import os


############
# Settings #
############

# Generic settings.
device = 0  # GPU[0].
seed_everything(0, verbose=False)
# Filter some warnings.
warnings.filterwarnings(
    "ignore", message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
)

# Deep Learning settings.
config_path = "configs/dinov3/coco/instance/eomt_large_640.yaml"
data_path = "/workspace/data/Datasets/SCD_for_EoMT/"
masked_attn_enabled = False
my_checkpoint = '/workspace/data/EOMT/Checkpoints/Run2/epoch=055-mAP=0.81.ckpt'  # Should the mAP not reach 86?

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


#######################################
# 4. Run the inference with TensorRT. #
#######################################
# Engines computed with TRT 10.7.
# ENGINE_PATH = '/workspace/eomt/trt/TensorRT/model_fp16.engine'  # DOES NOT WORK. OUTPUTS NANS.
ENGINE_PATH = '/workspace/eomt/trt/TensorRT/model_fp16_correct.engine'
# ENGINE_PATH = '/workspace/eomt/trt/TensorRT/model_fp32.engine'  # Works.
# ENGINE_PATH = '/workspace/eomt/trt/TensorRT/model_int8.engine'

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
# Load the engine.
with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

output_dir = '/workspace/eomt/output/tensorrt_eomt/'
os.makedirs(output_dir, exist_ok=True)

# Create the execution context and the CUDA stream.
context = engine.create_execution_context()
stream = cuda.Stream()

# List the IO tensors, for inspection.
tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
print('')
print('Tensors:', tensor_names)

val_dataset = data.val_dataloader().dataset
step = 1

for img_idx in [0]:
    # Get the sample.
    img, target = val_dataset[img_idx]
    output_path_base =  os.path.join(output_dir, f"image_{img_idx:04d}")

    # Create dictionaries to store buffers.
    buffers = {}
    host_inputs = {}
    host_outputs = {}

    # Allocate buffers and set device pointers for all IO tensors.
    for name in tensor_names:
        shape = context.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        np_dtype = trt.nptype(dtype)
        elem_size = np.dtype(trt.nptype(dtype)).itemsize
        size_bytes = int(np.prod(shape) * elem_size)

        print(f"\t{name}: shape={shape}, dtype={dtype}, bytes={size_bytes}")

        # Allocate device memory, store reference in buffers.
        d_buffer = cuda.mem_alloc(size_bytes)
        buffers[name] = d_buffer

        # Set the tensor address.
        context.set_tensor_address(name, int(d_buffer))

        # Prepare input data.
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            # Format the input image.
            if dtype == trt.DataType.UINT8:
                imgs = [img.to(device)]
                img_sizes = [img.shape[-2:] for img in imgs]
                transformed_imgs = model.resize_and_pad_imgs_instance_panoptic(imgs)

                # TODO: is this the correct way to feed the engine? Why do I have to do the cpu part?
                h_input = transformed_imgs.cpu().contiguous().numpy().astype(np_dtype)

                # This is how PyTorch processing works.
                # mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs)

            else:
                raise Exception('WEIRD')

            host_inputs[name] = h_input
            # Copy to device.
            cuda.memcpy_htod_async(d_buffer, h_input, stream)
        else:
            # Prepare host buffer for output.
            host_outputs[name] = np.zeros(shape, dtype=trt.nptype(dtype))


    context.set_input_shape("input", (1, 3, 640, 640))
    actual_shape = context.get_tensor_shape("input")
    print(f"Context input shape: {actual_shape}")

    # Synchronize the stream before inference to ensure inputs are copied.
    stream.synchronize()

    # Run the inference.
    print('Running inference.')
    success = context.execute_async_v3(stream_handle=stream.handle)

    if not success:
        print("ERROR: execute_async_v3 returned False!")
    else:
        print("Inference launched successfully.")

    # Synchronize the stream before copying the output values.
    stream.synchronize()

    print('Inference complete.')

    # Copy the outputs back to the host.
    for name, h_output in host_outputs.items():
        cuda.memcpy_dtoh(h_output, buffers[name])
        print(f'{name} output shape: {h_output.shape}')
        print(f'{name}: {h_output}')
    pass