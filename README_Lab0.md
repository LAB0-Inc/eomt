# General info
### Docker image
The Docker image can be created like this:
```
docker run --gpus all -v ~/.lab0/data/:/workspace/data/ --shm-size=2g -it --rm nvcr.io/nvidia/pytorch:24.09-py3
```

Then, the Docker image can be save to disk as `eomt:latest`. The following times, the image with the code can be opened like this:
```
docker run --gpus all -v ~/.lab0/data/:/workspace/data/ --volume /etc/localtime:/etc/localtime:ro --volume /etc/timezone:/etc/timezone:ro   -v ~/.gitconfig:/root/.gitconfig:ro -v $SSH_AUTH_SOCK:/ssh-agent -e SSH_AUTH_SOCK=/ssh-agent  --privileged --shm-size=2g --name EoMT -it eomt:latest
```

### Git repo
We have a [clone of the original repo](https://github.com/LAB0-Inc/eomt).
You can clone that repo inside the Docker image, and install the code as follows:
```
git clone git@github.com:LAB0-Inc/eomt
python3 -m pip install -r requirements.txt
```

### Data
The dataset folder resides in the `~/.lab0/data/Datasets/SCD_for_EoMT` folder on the host file system.
The `data` folder is mounted automatically because it is inside `~/.lab0/data/`.
The dataset can be found on the NAS, it is called `Clean_SCD`, there.

### WANDB
Training and validation data is logged on [Weights and biases](https://wandb.ai/eleramp-lab0/eomt) (W&B, wandb.ai).

### Folder structure
The following folders have to be created in the project root, as symbolic links to folders from `/workspace/data`, which is mounted from the host file system:
* `checkpoints`
* `output`
* `trt`

### Extra stuff
Scripts that are not currectly in use are stored in `extra/`.


## Files
| File | Usage |
|------|-------|
| configs/dinov3/coco/instance/eomt_large_640.yaml | Defines the training. |
| training/mask_classification_instance.py | Defines, among other things, eval_step(). At the end of that you can enable saving segm. images. |
| datasets/coco_instance.py | Contains the hard-coded extension for the input images, defines how train and val datasets lists are loaded. |

# Training
The script is: `train_SCD.sh`.
It accumulates the gradient of 16 samples before applying Gradient Descent.
The best three checkpoints are saved for the latest training in `checkpoints`. **Warning, this will overwrite existing checkpoints** so, before starting a new training, you should move old checkpoints in subdirectories like: `checkpoints/Run2`.

# PyTorch inference on one image (or more)
The original authors provide the `inference.ipynb` notebook. [Not sure if that is currently working]

The `inference.py` script can run inference on a single image or on the full validation set.

# Validation
The script is `validate_SCD.sh`.
The dataloader, `dataset/coco_instance.py`, applies a scaling to the images before feeding them to the segmentor. I think setting the scale range to (1, 1) before running a validation makes sense: `scale_range=(1.0, 1.0)`.

By default, the validation process does not save the segmentation images. If you want to save them, you need to set `save_visualizations = True` in `training/mask_classification_instance.py`. You should also select a new output folder, to avoid overwriting existing data.
When the images are saved, the `iou_log.txt` file is saved in the same folder. It contains the original file name of the input image, the value of the Semantic Segmentation IoU (so it's about the area on the image, not single segments) between the GT and the estimates, the number of segments in the estimate, that in the GT, and the difference between them.
It is possible that saving the images only works with a batch size of 1.

In case you want to only save the images, without computing the validation statistics, you can comment out this line in `mask_classification_instance.py`: `self.update_metrics_instance(preds, targets_, i)`.

In case you want to run the validation on the training set (it takes ~2h30), you should change the definition of the validation set in `datasets/dataset.py` to be the same as the training set.

In case you want to run the validation on a specific sample (for debug purposes), you can inject the index of the sample in `datasets/dataset.py`, so that it would repeatedly work on the same sample (or list of samples).

# Converting to TensorRT


