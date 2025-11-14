# General info
## Docker image
The Docker image can be created like this:
```
docker run --gpus all -v ~/.lab0/data/:/workspace/data/ --shm-size=2g -it --rm nvcr.io/nvidia/pytorch:24.09-py3
```

Then, the Docker image can be save to disk as `eomt:latest`. The following times, the image with the code can be opened like this:
```
docker run --gpus all -v ~/.lab0/data/:/workspace/data/ --volume /etc/localtime:/etc/localtime:ro --volume /etc/timezone:/etc/timezone:ro --privileged --shm-size=2g --name EoMT -it  -v ~/.ssh:/root/.ssh:ro -v ~/.gitconfig:/root/.gitconfig:ro eomt:latest
```

## Git repo
We have a [clone of the original repo](https://github.com/LAB0-Inc/eomt).
You can clone that repo inside the Docker image, and install the code as follows:
```
git clone git@github.com:LAB0-Inc/eomt
python3 -m pip install -r requirements.txt
```

## Data
The dataset folder resides in the `~/.lab0/data/Datasets/SCD_for_EoMT` folder on the host file system.
The `data` folder is mounted automatically because it is inside `~/.lab0/data/`.
The dataset can be found on the NAS, it is called `Clean_SCD`, there.

## WANDB
Training and validation data is logged on [Weights and biases](https://wandb.ai/eleramp-lab0/eomt) (W&B, wandb.ai).

## Folder structure
The following folders have to be created in the project root, as symbolic links to folders from `/workspace/data`, which is mounted from the host file system:
* `checkpoints`
* `output`
* `trt`

### Extra stuff
Scripts that are not currectly in use are stored in `extra/`.


# Training
The script is: `train_SCD.sh`.
It accumulates the gradient of 16 samples before applying Gradient Descent.
The best three checkpoints are saved for the latest training in `checkpoints`. **Warning, this will overwrite existing checkpoints** so, before starting a new training, you should move old checkpoints in subdirectories like: `checkpoints/Run2`.

# Validation (on the whole validation set)
The script is `validate_SCD.sh`. TODO: I need to check what it produces in output.

# PyTorch inference on one image (or more)
The original authors provide the `inference.ipynb` notebook.

# TensorRT engine



