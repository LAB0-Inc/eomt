import yaml
import torch
from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
import matplotlib.pyplot as plt
import warnings
import importlib

import onnx
import onnxruntime as ort
import numpy as np


warnings.filterwarnings(
    "ignore",
    message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
)


def to_numpy(output):
    if isinstance(output, torch.Tensor):
        return output.detach().cpu().numpy() if output.requires_grad else output.cpu().numpy()
    elif isinstance(output, (tuple, list)):
        return [to_numpy(x) for x in output]
    else:
        raise TypeError(f"Unsupported type: {type(output)}")


def draw_black_border(sem, inst, mapping):
    h, w = sem.shape
    out = np.zeros((h, w, 3))
    for s in np.unique(sem):
        out[sem == s] = mapping[s]

    combined = sem.astype(np.int64) * 100000 + inst.astype(np.int64)
    border = np.zeros((h, w), dtype=bool)
    border[1:, :] |= combined[1:, :] != combined[:-1, :]
    border[:-1, :] |= combined[1:, :] != combined[:-1, :]
    border[:, 1:] |= combined[:, 1:] != combined[:, :-1]
    border[:, :-1] |= combined[:, 1:] != combined[:, :-1]
    out[border] = 0
    return out


def plot_panoptic_results(img, sem_pred, inst_pred, sem_target, inst_target, output_path_base):
    all_ids = np.union1d(np.unique(sem_pred), np.unique(sem_target))
    mapping = {
        s: (
            [0, 0, 0]
            if s == -1 or s == model.num_classes
            else plt.cm.hsv(i / len(all_ids))[:3]
        )
        for i, s in enumerate(all_ids)
    }

    vis_pred = draw_black_border(sem_pred, inst_pred, mapping)
    vis_target = draw_black_border(sem_target, inst_target, mapping)

    img_np = (
        img.cpu().numpy().transpose(1, 2, 0) if img.dim() == 3 else img.cpu().numpy()
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Input")
    axes[1].imshow(vis_pred)
    axes[1].set_title("Prediction")
    axes[2].imshow(vis_target)
    axes[2].set_title("Target")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_path_base}.jpg")
    plt.close()


def infer_panoptic(img, target):
    with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
        imgs = [img.to(device)]
        img_sizes = [img.shape[-2:] for img in imgs]
        transformed_imgs = model.resize_and_pad_imgs_instance_panoptic(imgs)
        # print("Original shape:", imgs[0].shape)
        # print("Transformed shape:", transformed_imgs.shape)
        del img, imgs  # Free memory.

        mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs)
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], model.img_size, mode="bilinear"
        )
        mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
            mask_logits, img_sizes
        )

        preds = model.to_per_pixel_preds_panoptic(
            mask_logits,
            class_logits_per_layer[-1],
            model.stuff_classes,
            model.mask_thresh,
            model.overlap_thresh,
        )[0].cpu()

    pred = preds.numpy()
    sem_pred, inst_pred = pred[..., 0], pred[..., 1]

    target_seg = model.to_per_pixel_targets_panoptic([target])[0].cpu().numpy()
    sem_target, inst_target = target_seg[..., 0], target_seg[..., 1]

    return sem_pred, inst_pred, sem_target, inst_target


def infer_panoptic_onnx(img, target):
    with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
        imgs = [img.to(device)]
        img_sizes = [img.shape[-2:] for img in imgs]
        transformed_imgs = model.resize_and_pad_imgs_instance_panoptic(imgs)
        # print("Original shape:", imgs[0].shape)
        # print("Transformed shape:", transformed_imgs.shape)
        del img, imgs  # Free memory.

        mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs)
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], model.img_size, mode="bilinear"
        )
        mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
            mask_logits, img_sizes
        )

        preds = model.to_per_pixel_preds_panoptic(
            mask_logits,
            class_logits_per_layer[-1],
            model.stuff_classes,
            model.mask_thresh,
            model.overlap_thresh,
        )[0].cpu()

    pred = preds.numpy()
    sem_pred, inst_pred = pred[..., 0], pred[..., 1]

    target_seg = model.to_per_pixel_targets_panoptic([target])[0].cpu().numpy()
    sem_target, inst_target = target_seg[..., 0], target_seg[..., 1]

    return sem_pred, inst_pred, sem_target, inst_target

########
# MAIN #
########

# Setup
device = 0  # GPU[0].
config_path = "configs/dinov3/coco/instance/eomt_large_640.yaml"
data_path = "/workspace/data/Datasets/SCD_for_EoMT/"
# my_checkpoint = '/workspace/data/EOMT/Checkpoints/Run1/epoch=026-mAP=0.81.ckpt'  # Run 1.
masked_attn_enabled = False  # If it's True, the conversion to ONNX fails, we would need to look into that.
my_checkpoint = '/workspace/data/EOMT/Checkpoints/Run2/epoch=055-mAP=0.81.ckpt'  # Run 2.



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
input_sample = model.resize_and_pad_imgs_instance_panoptic(imgs)
# The inference can be run like this: mask_logits_per_layer, class_logits_per_layer = model(input_sample)


############################################
# 5. Create the ONNX version of the model. #
############################################
print('Exporting scripted model to ONNX.')
model.eval()
torch.onnx.export(
    model,                              # PyTorch model
    input_sample,                       # Example input tensor
    "checkpoints/eomt_large_640.onnx",  # Output file
    input_names=['input'],              # Optional: name model inputs
    output_names=['output'],            # Optional: name model outputs
    opset_version=18,                   # ONNX operator set version
    do_constant_folding=True,           # Optimize constants (recommended)
    dynamo=True                         # Not sure what this does, but it seems to be important.

    # dynamic_axes={                    # Optional: mark variable-sized dimensions
    #     'input': {0: 'batch_size'},
    #     'output': {0: 'batch_size'}
    # }
)


###############################
# 6. Validate the ONNX model. #
###############################
print('Validating the ONNX model.')
# Load and verify structure
onnx_model = onnx.load("checkpoints/eomt_large_640.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")


##############################################
# 7. Compare PyTorch output with ONNX ouput. #
##############################################
# 7.1 Perform inference on PyTorch.
torch_mask_logits_per_layer, torch_class_logits_per_layer = model(input_sample)
torch_out_0 = torch_mask_logits_per_layer[0].detach().cpu().numpy()
torch_out_1 = torch_class_logits_per_layer[0].detach().cpu().numpy()

# 7.2 Perform inference on ONNX.
ort_session = ort.InferenceSession("checkpoints/eomt_large_640.onnx", providers=["CUDAExecutionProvider"])
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_sample)}
ort_outs = ort_session.run(None, ort_inputs)
ort_out_0 = ort_outs[0]
ort_out_1 = ort_outs[1]

# 7.3 Check the numerical difference.
tolerance = 6e-2
np.testing.assert_allclose(torch_out_0, ort_out_0, rtol=tolerance, atol=tolerance)
np.testing.assert_allclose(torch_out_1, ort_out_1, rtol=tolerance, atol=tolerance)

print(f"Outputs match between PyTorch and ONNX (tolerance = {tolerance}).")

# 7.4 Check the difference visually.
# PyTorch.
out_file_name = '/workspace/eomt/output/compare_onnx/pytorch'  # The extension is added inside the function.
sem_pred, inst_pred, sem_target, inst_target = infer_panoptic(img, target)
plot_panoptic_results(img, sem_pred, inst_pred, sem_target, inst_target, out_file_name)

out_file_name = '/workspace/eomt/output/compare_onnx/onnx'  # The extension is added inside the function.
sem_pred, inst_pred, sem_target, inst_target = infer_panoptic_onnx(img, target)
plot_panoptic_results(img, sem_pred, inst_pred, sem_target, inst_target, out_file_name)

pass