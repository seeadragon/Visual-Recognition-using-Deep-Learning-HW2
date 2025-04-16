import os
import json
import csv
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval

from dataloader import GetLoader


class FasterRCNN(nn.Module):
    """
    backbone:
    region proposal network (RPN)
    roi heads (RoI heads)
    """
    def __init__(self, config):
        """
        FasterRCNN initialization
        """
        super().__init__()
        self.config = config

        self.num_classes = config['num_classes']
        self.pretrained = config['pretrained']
        self.device = config['device']
        self.epochs = config['num_epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.optimizer = config['optimizer']
        self.log_dir = config['log_dir']
        self.weight_decay = config['weight_decay']

        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        anchor_generator = AnchorGenerator(
            sizes=((4,), (8,), (14,), (22,), (32,), (44),),
            aspect_ratios=((0.5, 1.0, 1.5),) * 5
        )
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )
        if self.pretrained:
            self.model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler
            )
        else:
            self.model = fasterrcnn_resnet50_fpn(
                weights=None,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler
            )

        self.model.roi_heads.nms_thresh = 0.2
        self.model.roi_heads.score_thresh = 0.5 # v1 = 0.05 v2 = 0.5 v3 = 0.4 v4 = 0.6
        self.model.roi_heads.batch_size_per_image = 256
        self.model.roi_heads.positive_fraction = 0.25

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)


    def forward(self, x, targets=None):
        """
        
        """
        if self.training and targets is not None:
            return self.model(x, targets)
        return self.model(x)


    def predict(self, x):
        """
        
        """
        self.eval()
        with torch.no_grad():
            predictions = self.model(x)
        return predictions

    ### train function
    ### train epoch
    ### valid epoch

    def train(self):
        """
        train the model
        """
        self.model.to(self.device)

        data_loader = GetLoader(data_dir='data')
        data_loader.batch_size = self.batch_size
        train_loader, _ = data_loader.train_loader()
        valid_loader, valid_dataset = data_loader.valid_loader()

        optimizer = self.optimizer(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )

        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(train_loader, optimizer)
            val_loss, map_score = self._eval_epoch(valid_loader, valid_dataset)

            log_stats = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'map_score': map_score,
            }
            lr_scheduler.step()

            self.write_log(log_stats)
            self.save_model(epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(self.log_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), best_model_path)

        self.writer.close()


    def _train_epoch(self, train_loader, optimizer):
        """
        train the model on the training set 1 epoch
        """
        self.model.train()
        train_loss = 0.0
        counter = 0
        with tqdm(train_loader, desc='Train') as pbar:
            for images, targets in pbar:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                train_loss += loss_value
                counter += 1

                pbar.set_postfix(loss=loss_value)


        avg_loss = train_loss / counter if counter > 0 else 0
        return avg_loss


    def _eval_epoch(self, val_loader, val_dataset):
        """
        evaluate the model on the validation set 1 epoch
        """
        self.model.eval()
        val_loss = 0.0
        counter = 0

        coco_results = []
        coco_gt = val_dataset.coco if val_dataset is not None else None

        with torch.no_grad():
            with tqdm(val_loader, desc='Valid') as pbar:
                for images, targets in pbar:
                    images = [image.to(self.device) for image in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    predictions = self.model(images)

                    if coco_gt is not None:
                        for pred, target in zip(predictions, targets):
                            image_id = target['image_id'].item()

                            boxes = pred['boxes'].cpu().numpy()
                            scores = pred['scores'].cpu().numpy()
                            labels = pred['labels'].cpu().numpy()

                            for box, score, label in zip(boxes, scores, labels):
                                x1, y1, x2, y2 = box
                                width = x2 - x1
                                height = y2 - y1
                                coco_results.append({
                                    'image_id': image_id,
                                    'category_id': int(label),
                                    'bbox': [float(x1), float(y1), float(width), float(height)],
                                    'score': float(score)
                                })
                    self.model.train()
                    loss_dict = self.model(images, targets)
                    self.model.eval()

                    if isinstance(loss_dict, dict):
                        losses = sum(loss for loss in loss_dict.values())
                        loss_value = losses.item()

                        val_loss += loss_value
                        counter += 1

                        pbar.set_postfix(loss=loss_value)

                    del images, targets, predictions, loss_dict, losses

        avg_loss = val_loss / counter if counter > 0 else 0

        map_score = None
        if coco_gt is not None and coco_results:
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(coco_results, f)
                    temp_file_name = f.name

                map_score = self._calculate_map(temp_file_name, coco_gt)
                os.remove(temp_file_name)
            except Exception as e:
                print(f"Error calculating mAP: {e}")


        return avg_loss, map_score


    def _calculate_map(self, coco_results, coco_gt):
        """
        using COCOeval to calculate mAP
        """
        try:
            coco_dt = coco_gt.loadRes(coco_results)

            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            return coco_eval.stats[0]  # mAP@0.5
        except Exception as e:
            print(f"Error calculating mAP: {e}")
            return None

    ### test function
    ### detection json file
    ### recognition csv file

    def test(self):
        """
        Generate task1 and task2 result for test dataset
        """
        self.model.to(self.device)

        data_loader = GetLoader(data_dir='data')
        data_loader.batch_size = self.batch_size
        test_loader, _ = data_loader.test_loader()

        self._generate_task1(test_loader)
        print('task1!')
        self._generate_task2()
        print('task2!')


    def _generate_task1(self, test_loader, output_path='pred.json'):
        """
        Generate results in json format for task 1
        """
        self.model.eval()
        results = []
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc="Detection"):
                images = [img.to(self.device) for img in images]
                predictions = self.model(images)

                for pred, target in zip(predictions, targets):
                    image_id = target['image_id'].item()

                    boxes = pred['boxes'].cpu().numpy()
                    scores = pred['scores'].cpu().numpy()
                    labels = pred['labels'].cpu().numpy()

                    for box, score, label in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = box
                        width = x2 - x1
                        height = y2 - y1

                        results.append({
                            'image_id': int(image_id),
                            'bbox': [float(x1), float(y1), float(width), float(height)],
                            'score': float(score),
                            'category_id': int(label)
                        })

        with open(output_path, 'w') as f:
            json.dump(results, f)

        return results


    def _generate_task2(self, json_path='pred.json', output_path='pred.csv'):
        """
        Convert the detection results to a CSV file for task 2
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            detections = json.load(f)

        results_by_image = {}
        for detection in detections:
            image_id = detection['image_id']
            if image_id not in results_by_image:
                results_by_image[image_id] = []

            results_by_image[image_id].append({
                'box': detection['bbox'],
                'score': detection['score'],
                'category_id': detection['category_id']
            })

        results = []
        for image_id, detections in results_by_image.items():
            if not detections:
                pred_label = -1
            else:
                detections.sort(key=lambda x: x['box'][0])

                digit_values = [str(int(d['category_id'])-1) for d in detections]
                try:
                    pred_label = int(''.join(digit_values))
                except ValueError:
                    pred_label = -1

            results.append([int(image_id), pred_label])

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_id', 'pred_label'])
            writer.writerows(results)

        return results


    def write_log(self, log_stats):
        """
        print log
        write log to tensorboard
        """
        print(f'Epoch: {log_stats["epoch"]}, '
              +f'Train Loss: {log_stats["train_loss"]}, '
              +f'Val Loss: {log_stats["val_loss"]}')

        self.writer.add_scalar('Loss/train', log_stats['train_loss'], log_stats['epoch'])
        self.writer.add_scalar('Loss/val', log_stats['val_loss'], log_stats['epoch'])
        if log_stats['map_score'] is not None:
            print(f'mAP@0.5: {log_stats["map_score"]}')
            self.writer.add_scalar('mAP', log_stats['map_score'], log_stats['epoch'])
        else:
            print('mAP@0.5: No predictions for mAP evaluation.')


    def save_model(self, epoch):
        """
        save model checkpoint
        """
        model_path = os.path.join(self.log_dir, f'model_epoch_{epoch}.pth')
        torch.save(self.model.state_dict(), model_path)


    def load_model(self, model_path):
        """
        load model checkpoint
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)


    def count_params(self):
        """
        Returns:
            total params: int
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params


    def _normalize(self, image):
        image = image.cpu()
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

        image_normalized = inv_normalize(image)
        image_np = image_normalized.permute(1, 2, 0).numpy()
        return np.clip(image_np, 0, 1)

    def _setup_figure(self):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        blue_patch = patches.Patch(color='blue', label='Ground Truth')
        red_patch = patches.Patch(color='red', label='Prediction')

        ax.legend(handles=[blue_patch, red_patch], loc='upper right')
        return fig, ax

    def _draw_boxes(self, ax, boxes, labels, scores=None, is_prediction=False):
        color = 'red' if is_prediction else 'blue'
        prefix = 'Predicted' if is_prediction else 'Ground Truth'

        for i, (box, label) in enumerate(zip(boxes, labels)):
            if is_prediction and scores is not None and scores[i] < 0.5:
                continue

            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)

            text_y = y2+5 if is_prediction else y1-5
            label_text = f'{prefix}: {label-1}'

            if is_prediction and scores is not None:
                label_text += f' ({scores[i]:.2f})'

            ax.text(
                x1, text_y,
                label_text,
                fontsize=9,
                color=color,
                bbox={"facecolor": 'white', "alpha": 0.7}
            )

    def visual(self):
        """
        Visualize the predictions and ground truth on the validation set.
        """
        os.makedirs("visual", exist_ok=True)
        valid_loader, _ = GetLoader(data_dir='data').valid_loader()

        self.model.eval()
        with torch.no_grad():
            with tqdm(valid_loader, desc='Visual') as pbar:
                for images,targets in pbar:
                    images = [image.to(self.device) for image in images]

                    predictions = self.model(images)

                    for (image, target, preidction) in enumerate(zip(images, targets, predictions)):
                        image_id = target['image_id'].item()
                        pbar.set_postfix(image_id=image_id)

                        # denormalize
                        image_np = self._normalize(image)

                        fig, ax = self._setup_figure()
                        ax.imshow(image_np)
                        ax.set_title(f"Image ID: {image_id}")
                        ax.axis('off')

                        true_boxes = target['boxes'].cpu().numpy()
                        true_labels = target['labels'].cpu().numpy()
                        self._draw_boxes(ax, true_boxes, true_labels, is_prediction=False)

                        pred_boxes = preidction['boxes'].cpu().numpy()
                        pred_labels = preidction['labels'].cpu().numpy()
                        pred_scores = preidction['scores'].cpu().numpy()
                        self._draw_boxes(ax, pred_boxes, pred_labels, pred_scores, is_prediction=True)

                        plt.tight_layout()
                        plt.savefig(f"visual/image_{image_id}.png")
                        plt.close(fig)
                        pbar.set_postfix(image_id=image_id)
