'''transforms和数据增强'''
import torch
import random
import torchvision.transforms as t


#自己写的模块
from src.utils import dboxes300_coco, Encoder, calc_iou_tensor

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target=None):
        for trans in self.transforms:
            image, target = trans(image, target)
        return image, target


class ToTensor(): # 仅仅对图像进行操作即可
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = t.functional.to_tensor(image).contiguous()
        return image, target


class RandomHorizontalFlip():
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            # height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target['boxes']
            bbox[:, [0, 2]] = 1.0 - bbox[:, [2, 0]]
            target['boxes'] = bbox
        return image, target


class SSDCropping():
    '''
    根据原文，对图像进行裁剪，该方法应该放在ToTensor前
    Cropping for SSD, according to original paper
    Choose between following 3 conditions:
    1. Preserve the original image
    2. Random crop minimum IoU is among 0.1, 0.3, 0.5, 0.7, 0.9
    3. Random crop
    Reference to https://github.com/chauhan-utk/ssd.DomainAdaptation.
    '''
    def __init__(self):
        self.sample_options = (
            None, # Do nothing
            (0.1, None), # min IoU, max IoU
            (0.3, None), # min IoU, max IoU
            (0.5, None), # min IoU, max IoU
            (0.7, None), # min IoU, max IoU
            (0.9, None), # min IoU, max IoU
            (None, None), # no IoU requirements
            )
        self.dboxes = dboxes300_coco()
    
    def __call__(self, image, target):
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, target
            htruth, wtruth = target['height_width'] # 真实高宽

            min_iou, max_iou = mode
            min_iou = float('-inf') if min_iou is None else min_iou
            max_iou = float('+inf') if max_iou is None else max_iou

            # 迭代5次找到可能的候选
            for _ in range(5):
                # 截取的高宽都在0.3到1之间
                w = random.uniform(0.3, 1.0)
                h = random.uniform(0.3, 1.0)

                if w/h < 0.5 or w/h > 2: # 保证宽高比例在0.5-2之间
                    continue

                # left top xmin:0 ~ (1.0 - w), ymin:0 ~ (1.0 - h)
                # 用来确保剪切的块完全在原图内
                xmin = random.uniform(0, 1.0 - w)
                ymin = random.uniform(0, 1.0 - h)

                xmax = xmin + w
                ymax = ymin + h


                # boxes的坐标是在0-1之间的，在数据加载时以及处理了
                bboxes = target['boxes']
                ious = calc_iou_tensor(bboxes, torch.tensor([[xmin, ymin, xmax, ymax]]))
                #torch.tensor([[left, top, right, bottom]])

                # tailor all the bboxes and return
                # all(): retrun true if all the elements in the tensor are True, False otherwisr
                if not ((ious > min_iou) & (ious < max_iou)).all():
                    continue

                # 确定有GT的中心都在剪切的图中
                xc = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
                yc = 0.5 * (bboxes[:, 1] + bboxes[:, 3])
                masks = (xc > xmin) & (xc < xmax) & (yc > ymin) & (yc < ymax)

                # 如果所有的gt box的中心点都不在采样的图中，则重新找
                if not masks.any():
                    continue

                # 修改采样图中所有的gt box的坐标（防止出现越界的情况）
                bboxes[bboxes[:, 0] < xmin, 0] = xmin
                bboxes[bboxes[:, 1] < ymin, 1] = ymin
                bboxes[bboxes[:, 2] > xmax, 2] = xmax
                bboxes[bboxes[:, 3] > ymax, 3] = ymax

                # 滤除不在采样patch中的gt box
                bboxes = bboxes[masks, :]
                # 获取在采样patch中的gt box的标签
                labels = target['labels']
                labels = labels[masks]

                # 裁剪patch
                xmin_truth = int(xmin * wtruth)
                ymin_truth = int(ymin * htruth)
                xmax_truth = int(xmax * wtruth)
                ymax_truth = int(ymax * htruth)

                image = image.crop((xmin_truth, ymin_truth, xmax_truth, ymax_truth))

                # 调整裁剪后的bboxes信息
                bboxes[:, 0] = (bboxes[:, 0] - xmin) / w
                bboxes[:, 1] = (bboxes[:, 1] - ymin) / h
                bboxes[:, 2] = (bboxes[:, 2] - xmax) / w
                bboxes[:, 3] = (bboxes[:, 3] - ymax) / h

                # 更新crop后的gt box的坐标信息及标签信息
                target['boxes'] = bboxes
                target['labels'] = labels
                return image, target


class Resize():
    def __init__(self, size=(300, 300)):
        self.resize = t.Resize(size)

    def __call__(self, image, target):
        image = self.resize(image)
        return image, target
    

class ColorJitter():
    '''
    对图像颜色进行随机调整，该方法应该放在ToTensor前
    亮度=0.125，对比度=0.5，饱和度=0.5，色调=0.05
    '''
    def __init__(self, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.5):
        self.trans = t.ColorJitter(brightness, contrast, saturation, hue)
    
    def __call__(self, image, target):
        image = self.trans(image)
        return image, target


class Normalization():
    def __init__(self, mean=None, std=None):
        if mean == None:
            mean = [0.485, 0.456, 0.406]
        if std == None:
            std = [0.229, 0.224, 0.406]
        self.normalization = t.Normalize(mean, std)

    def __call__(self, image, target):
        image = self.normalization(image)
        return image, target


class AssignGTtoDefaultBox():
    """将DefaultBox与GT进行匹配"""
    def __init__(self):
        self.default_box = dboxes300_coco()
        self.encoder = Encoder(self.default_box)
    
    def __call__(self, image, target):
        boxes = target['boxes']
        labels = target['labels']

        # bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
        bboxes_out, labels_out = self.encoder.encode(boxes, labels)
        target['boxes'] = bboxes_out
        target['labels'] = labels_out

        return image, target
