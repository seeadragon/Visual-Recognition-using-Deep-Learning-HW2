import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image


class DetectionDataset(Dataset):
    """
    handle train and valid dataset
    """
    def __init__(self, root_dir, file_name, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(file_name, 'r') as file:
            self.data = json.load(file)

        self.coco = COCO(file_name)
        self.coco.dataset = self.data

        self.image = {image['id']: image for image in self.data['images']}

        self.annotation = {}
        for annotation in self.data['annotations']:
            image_id = annotation['image_id']
            if image_id not in self.annotation:
                self.annotation[image_id] = []
            self.annotation[image_id].append(annotation)

        self.image_ids = list(self.image.keys())

        self.cat_id_to_idx = {cat['id']: i for i, cat in enumerate(self.data['categories'])}

        self.categories = {cat['id']: cat['name'] for cat in self.data['categories']}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image[image_id]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        ann_list = self.annotation.get(image_id, [])
        target = {
            'boxes': [],
            'labels': [],
            'image_id': torch.tensor([image_id]),
            'area': [],
            'iscrowd': []
        }

        for ann in ann_list:
            x, y, w, h = ann['bbox']
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            target['boxes'].append([x1, y1, x2, y2])

            cat_idx = self.cat_id_to_idx[ann['category_id']] + 1 # 0 for background
            target['labels'].append(cat_idx)

            target['area'].append(ann['area'])
            target['iscrowd'].append(ann['iscrowd'])

        if len(target['boxes']) > 0:
            target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
            target['area'] = torch.tensor(target['area'], dtype=torch.float32)
            target['iscrowd'] = torch.tensor(target['iscrowd'], dtype=torch.int64)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['area'] = torch.zeros((0,), dtype=torch.float32)
            target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)

        return image, target


class TestDataset(Dataset):
    """
    handle test dataset
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in list(os.listdir(root_dir))]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image_id = int(os.path.splitext(image_name)[0])

        if self.transform is not None:
            image = self.transform(image)

        target = {
            'image_id': torch.tensor([image_id]),
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64),
            'area': torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.zeros((0,), dtype=torch.int64)
        }

        return image, target


class GetLoader():
    """
    Get data loader for train, valid and test dataset
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.batch_size = 2
        self.num_workers = 4

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def train_loader(self):
        """
        load train data
        """
        train_dataset = DetectionDataset(
            root_dir=os.path.join(self.data_dir, 'train'),
            file_name=os.path.join(self.data_dir, 'train.json'),
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

        return train_loader, train_dataset

    def valid_loader(self):
        """
        load valid data
        """
        valid_dataset = DetectionDataset(
            root_dir=os.path.join(self.data_dir, 'valid'),
            file_name=os.path.join(self.data_dir, 'valid.json'),
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

        return valid_loader, valid_dataset

    def test_loader(self):
        """
        load test data
        """
        test_dataset = TestDataset(
            root_dir=os.path.join(self.data_dir, 'test'),
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

        return test_loader, test_dataset
