import os
import torch
from torch.utils import data
from pathlib import Path
from PIL import Image

import numpy as np


class YOLODataset(data.Dataset):
    '''
    读取YOLO 数据格式的数据集
    数据标注：1->circle, 2->Yes, 3->No
    '''
    def __init__(self, root, transforms=None, is_train=True): # 数据集文件夹camera2/images 包含训练集合和测试集
        super().__init__()
        if is_train:
            # root = '/kaggle/input/diordata/'
            self.img_path = Path(root) / 'images'
            self.lable_path = Path(root) / 'labels'
            with open(str(Path(root) / 'ImageSets/train.txt'), 'r') as f:
                x = f.readlines()
            self.dataname = [i.strip() for i in x]


        else:
            self.img_path = Path(root) / 'images'
            self.lable_path = Path(root) / 'labels'
            with open(str(Path(root) / 'ImageSets/val.txt'), 'r') as f:
                x = f.readlines()
            self.dataname = [i.strip() for i in x]
        self.transforms = transforms

    def __len__(self):
        return len(self.dataname)
    
    def __getitem__(self, index):
        image = Image.open(str(self.img_path / (self.dataname[index] + '.jpg')))
        data_width, data_height = image.size
        height_width = [data_height, data_width]

        # print(self.dataname[index])

        # read txt
        boxes, labels = [], []
        with open(str(self.lable_path / (self.dataname[index] + '.txt')), 'r') as f:
            for line in f.readlines():
                x = [float(i) for i in line.split(' ')]
                boxes.append([x[1:]])
                labels.append(int(x[0]) + 1)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        xmin = boxes[:, 0] - 0.5 * boxes[:, 2]
        ymin = boxes[:, 1] - 0.5 * boxes[:, 3]
        xmax = boxes[:, 0] + 0.5 * boxes[:, 2]
        ymax = boxes[:, 1] + 0.5 * boxes[:, 3]
        boxes[:, 0] = xmin
        boxes[:, 1] = ymin
        boxes[:, 2] = xmax
        boxes[:, 3] = ymax
        # print(boxes)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        height_width = torch.as_tensor(height_width, dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['height_width'] = height_width

        # print(self.dataname[index])
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
    
    def coco_index(self, index):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        image = Image.open(str(self.img_path / (self.dataname[index] + '.jpg')))
        data_width, data_height = image.size
        height_width = [data_height, data_width]


        # read txt
        boxes, labels = [], []
        with open(str(self.lable_path / (self.dataname[index] + '.txt')), 'r') as f:
            for line in f.readlines():
                x = [float(i) for i in line.split(' ')]
                boxes.append([x[1:]])
                labels.append(int(x[0]) + 1)
        
        # iscrowd = torch.as_tensor([0, 0, 0], dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        xmin = boxes[:, 0] - 0.5 * boxes[:, 2]
        ymin = boxes[:, 1] - 0.5 * boxes[:, 3]
        xmax = boxes[:, 0] + 0.5 * boxes[:, 2]
        ymax = boxes[:, 1] + 0.5 * boxes[:, 3]
        boxes[:, 0] = xmin
        boxes[:, 1] = ymin
        boxes[:, 2] = xmax
        boxes[:, 3] = ymax

        iscrowd = torch.zeros(len(boxes))
        labels = torch.as_tensor(labels, dtype=torch.int64)
        height_width = torch.as_tensor(height_width, dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['height_width'] = height_width
        target['iscrowd'] = iscrowd

        return target
    
    @staticmethod
    def collate_fn(batch):
        images, targets = tuple(zip(*batch))
        # images = torch.stack(images, dim=0)
        #
        # boxes = []
        # labels = []
        # img_id = []
        # for t in targets:
        #     boxes.append(t['boxes'])
        #     labels.append(t['labels'])
        #     img_id.append(t["image_id"])
        # targets = {"boxes": torch.stack(boxes, dim=0),
        #            "labels": torch.stack(labels, dim=0),
        #            "image_id": torch.as_tensor(img_id)}

        return images, targets
    

if __name__ == '__main__':
    # import transforms
    # os.system('clear')
    # data_transform = {
    #     'train':transforms.Compose([transforms.SSDCropping(),
    #                                 transforms.Resize(),
    #                                 transforms.ColorJitter(),
    #                                 transforms.ToTensor(),
    #                                 transforms.RandomHorizontalFlip(),
    #                                 transforms.Normalization(),
    #                                 transforms.AssignGTtoDefaultBox()]),
    #     'val':transforms.Compose([transforms.Resize(),
    #                               transforms.ToTensor(),
    #                               transforms.Normalization()])}
    # x = YOLODataset('../camera', transforms=data_transform['train'])
    # for image, target in x:
    #     # image
    #     print(target)
    #     print(target['labels'].shape)
    #     print(target['labels'].max())
    #     break
    x = YOLODataset('dior')
