# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from pycocotools import mask as coco_mask
import torch
import numpy as np

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms
from datasets.dataset import Dataset

CLASS_MAPPING = {
    0: 0,
    1: 1,
    # 3: 2,
    # 4: 3,
}


class COCOInstance(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 1,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 2,
        color_jitter_enabled=False,
        scale_range=(1.0, 1.0),
        check_empty_targets=True,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(
        polygons_by_id: dict[int, list[list[float]]],
        labels_by_id: dict[int, int],
        is_crowd_by_id: dict[int, bool],
        width: int,
        height: int,
        **kwargs
    ):
        masks, labels, is_crowd = [], [], []

        for label_id, cls_id in labels_by_id.items():
            if cls_id not in CLASS_MAPPING:
                print(f'Ignoring label with class ID = {cls_id}')
                continue

            # segmentation = polygons_by_id[label_id]
            segmentation = (np.array(polygons_by_id[label_id], dtype=np.float32)).tolist()  # Ensure values are float.
            rles = coco_mask.frPyObjects(segmentation, height, width)
            rle = coco_mask.merge(rles) if isinstance(rles, list) else rles

            masks.append(tv_tensors.Mask(coco_mask.decode(rle), dtype=torch.bool))
            labels.append(CLASS_MAPPING[cls_id])
            is_crowd.append(is_crowd_by_id[label_id])

        return masks, labels, is_crowd

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_parser": self.target_parser,
            "only_annotations_json": True,
            "check_empty_targets": self.check_empty_targets,
        }
        # TODO: Train dataset is set on validation data.
        self.train_dataset = Dataset(
            transforms=self.transforms,
            #
            zip_path=Path(self.path, "validation_images.zip"),
            img_folder_path_in_zip=Path("./validation_images/"),
            target_zip_path=Path(self.path, "validation_annotations.zip"),
            annotations_json_path_in_zip=Path("./validation_annotations/annotations.json"),
            **dataset_kwargs,
        )
        self.val_dataset = Dataset(
            zip_path=Path(self.path, "validation_images.zip"),
            img_folder_path_in_zip=Path("./validation_images/"),
            target_zip_path=Path(self.path, "validation_annotations.zip"),
            annotations_json_path_in_zip=Path("./validation_annotations/annotations.json"),
            **dataset_kwargs,
        )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            # prefetch_factor=2,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            # prefetch_factor=2,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
