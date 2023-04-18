import torch
import torchvision
import numpy as np
from torch import nn, Tensor
from math import sqrt
# from torch.jit.annotations import Tuple, List
from torch.nn import functional as F
import itertools


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def calc_iou_tensor(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # 当输出形状不匹配时，返回输出量的形状遵循广播机制
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # left-top [N,M,2] 加入一个维度利用广播机制、
    rb  = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0) # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou



class Encoder():
    """
        Inspired by https://github.com/kuangliu/pytorch-src
        Transform between (bboxes, lables) <-> SSD output
        dboxes: default boxes in size 8732 x 4,
            encoder: input ltrb format, output xywh format
        encode:
            input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
            output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            criteria : IoU threshold of bboexes
    """
    def __init__(self, dboxes):
        self.dboxes = dboxes(order='ltrb')
        self.dboxes_xywh = dboxes(order='xywh').unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0) # default boxes的数量
    
    def encode(self, bboxes_in, labels_in, criteria=0.5):
        """
        encode:
            input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
            output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            criteria : IoU threshold of bboexes
        """
        # [nboxes, 8732]
        ious = calc_iou_tensor(bboxes_in, self.dboxes) # 计算每个GT与default box的iou
        # [8732,]
        best_dbox_ious, best_dbox_idxs = ious.max(dim=0) # 寻找每个default box匹配到的最大IoU
        # [nboxes,]
        best_bbox_ious, best_bbox_idxs = ious.max(dim=1) # 寻找每个GT匹配到的最大IoU
        
        # 将每个GT匹配到的最佳default box设置为正样本（对应论文中Matching strategy的第一条）
        # SET BEST IOUS 2.0，将GT匹配到的最大IOU的default box进行替换
        best_dbox_ious.index_fill(0, best_bbox_idxs, 2.0) # dim, index, value

        # 将相应default box匹配最大IOU的GT索引进行替换
        idx = torch.arange(0, best_bbox_idxs.size(0), dtype=torch.int64)
        best_dbox_idxs[best_bbox_idxs[idx]] = idx

        # filter IoU > 0.5
        # 寻找与GT iou大于0.5的default box，对应论文中Matching strategy的第二条
        # masks = best_dbox_idxs > criteria
        masks = best_dbox_ious > criteria
        # [8732,]
        labels_out = torch.zeros(self.nboxes, dtype=torch.int64)
        labels_out[masks] = labels_in[best_dbox_idxs[masks]]
        
        # 将default box匹配到正样本的位置设置为对应GT的box信息
        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idxs[masks], :] # 没匹配到的不变

        # Transform format to xywh format
        x = 0.5 * (bboxes_out[:, 0] + bboxes_out[:, 2]) # x
        y = 0.5 * (bboxes_out[:, 1] + bboxes_out[:, 3]) # y
        w = bboxes_out[:, 2] - bboxes_out[:, 0]
        h = bboxes_out[:, 3] - bboxes_out[:, 1]
        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h
        return bboxes_out, labels_out


class DefaultBoxes():
    '''self中没用的变量太多'''
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2):
        self.fig_size = fig_size  # 输入网络的图像大小 300
        self.feat_size = feat_size # 每个预测层的feature map尺寸, [38, 19, 10, 5, 3, 1]

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh # 小track，能加速收敛

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        # [8, 16, 32, 64, 100, 300] 采用2^n能够加速收敛？
        self.steps = steps # 每个特征层上的一个cell在原图上的跨度

        # [21, 45, 99, 153, 207, 261, 315]
        self.scales = scales # 每个特征层上预测的default box的scale，属于超参数

        fk = fig_size / np.array(steps) # 计算每层特征层的fk，用于求解中心点坐标
        
        # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]，为了方便没有加入1：1
        self.aspect_ratios = aspect_ratios # 每个预测特征层上预测的default box的ratios


        self.default_boxes = []
        # size of feature and number of feature
        # 遍历每层特征层，计算default box
        for idx, sfeat in enumerate(self.feat_size):
            sk1 = scales[idx] / fig_size
            sk2 = scales[idx + 1] / fig_size
            sk3 = sqrt(sk1 * sk2)

            # 先添加两个1:1比例的default box宽和高
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            # 再将剩下不同比列的default box 的宽高添加到all_sizes中
            for alpha in aspect_ratios[idx]:
                # 大小仍然是原来的scales
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))


            # 计算当前特征层对应原图上的所有的default box
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                # 计算每个default box的中心坐标（范围是在0-1之间
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))
        

        # 将default_boxes转为tensor格式
        self.dboxes = torch.as_tensor(self.default_boxes, dtype=torch.float32)
        self.dboxes.clamp_(min=0, max=1) # 将坐标（x, y, w, h）都限制在0-1之间

        # For IoU calculation
        # ltrb is left top coordinate and right bottom coordinate
        # 将(x, y, w, h)转换成(xmin, ymin, xmax, ymax)，方便后续计算IoU(匹配正负样本时)
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2] # xmin
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3] # ymin
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2] # xmax
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3] # ymax

    @property # 将方法当作属性来访问
    def scale_xy(self):
        return self.scale_xy_
    
    @property
    def scale_wh(self):
        return self.scale_wh_
    
    def __call__(self, order='ltrb'):
        # 根据要求返回default box
        if order == 'ltrb':
            return self.dboxes_ltrb
        if order == 'xywh':
            return self.dboxes


def dboxes300_coco():
    figsize = 300 # 输入网络的图像大小
    feat_size = [38, 19, 10, 5, 3, 1] # 每个预测层的feature map尺寸
    steps = [8, 16, 32, 64, 100, 300] # 每个特征层上的一个cell在原图上的跨度
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py

    scales = [21, 45, 99, 153, 207, 261, 315] # 每个特征层上预测的default box的scale
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


# def nms(boxes, scores, iou_threshold): # 实现不同于源代码
#      # type: (Tensor, Tensor, float) -> Tensor
#     """
#     Performs non-maximum suppression (NMS) on the boxes according
#     to their intersection-over-union (IoU).
#    IoU greater than iou_threshold with another (higher scoring)
#     box.   NMS iteratively removes lower scoring boxes which have an
  
#     Parameters
#     ----------
#     boxes : Tensor[N, 4])
#         boxes to perform NMS on. They
#         are expected to be in (x1, y1, x2, y2) format
#     scores : Tensor[N]
#         scores for each one of the boxes
#     iou_threshold : float
#         discards all overlapping
#         boxes with IoU < iou_threshold
#     Returns
#     -------
#     keep : Tensor
#         int64 tensor with the indices
#         of the elements that have been kept
#         by NMS, sorted in decreasing order of scores
#     """
#     return torchvision.ops.nms(boxes, scores, iou_threshold)

# def batched_nms(boxes, scores, idxs, iou_threshold): 
#     # 应该是可以直接调用torchvision.ops.boxes.batched_nms(boxes, scores, classes, nms_thresh)

#     # type: (Tensor, Tensor, Tensor, float) -> Tensor
#     """
#     Performs non-maximum suppression in a batched fashion.
#     Each index value correspond to a category, and NMS
#     will not be applied between elements of different categories.
#     Parameters
#     ----------
#     boxes : Tensor[N, 4]
#         boxes where NMS will be performed. They
#         are expected to be in (x1, y1, x2, y2) format
#     scores : Tensor[N]
#         scores for each one of the boxes
#     idxs : Tensor[N]
#         indices of the categories for each one of the boxes.
#     iou_threshold : float
#         discards all overlapping boxes
#         with IoU < iou_threshold
#     Returns
#     -------
#     keep : Tensor
#         int64 tensor with the indices of
#         the elements that have been kept by NMS, sorted
#         in decreasing order of scores
#     """
#     if boxes.numel() == 0:
#         return torch.empty((0,), dtype=torch.int64, device=boxes.device)
#     # strategy: in order to perform NMS independently per class.
#     # we add an offset to all the boxes. The offset is dependent
#     # only on the class idx, and is large enough so that boxes
#     # from different classes do not overlap
#     # 获取所有boxes中最大的坐标值（xmin, ymin, xmax, ymax）
#     max_coordinate = boxes.max()

#     # to(): Performs Tensor dtype and/or device conversion
#     # 为每一个类别生成一个很大的偏移量
#     # 这里的to只是让生成tensor的dytpe和device与boxes保持一致
#     offsets = idxs.to(boxes) * (max_coordinate + 1)
#     # boxes加上对应层的偏移量后，保证不同类别之间boxes不会有重合的现象
#     boxes_for_nms = boxes + offsets[:, None]
#     keep = nms(boxes_for_nms, scores, iou_threshold)
#     return keep
    


class PostProcess(nn.Module):
    def __init__(self, dboxes):
        super().__init__()
        # [num_anchors, 4] -> [1, num_anchors, 4]
        self.dboxes_xywh = nn.Parameter(dboxes('xywh').unsqueeze(dim=0), requires_grad=False)
        self.scale_xy = dboxes.scale_xy # 0.1
        self.scale_wh = dboxes.scale_wh # 0.2

        self.criteria = 0.5 # NMS的IoU阈值
        self.max_output = 100 # 最大有一百个目标

    def scale_back_batch(self, bboxes_in, scores_in):
        '''
            1）通过预测boxes回归参数得到最终目标
            2）将box格式从xywh转换为ltrb
            3）将预测目标score进行softmax处理
            Do scale and transform from xywh to ltrb
            suppose input N x 4 x num_bbox | N x label_num x num_bbox
            bboxes_in: [N, 4, 8732]是网络预测的xywh回归参数
            scores_in: [N, label_num, 8732]是预测的每个default box的各目标概率
        '''
        # [batch, 4, 8732] -> [batch, 8732, 4]
        bboxes_in = bboxes_in.permute(0, 2, 1)
        # [batch, label_num, 8732] -> [batch, 8732, label_num]
        scores_in = scores_in.permute(0, 2, 1)

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2] # 预测的x, y回归参数
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]

        # 将预测的回归参数叠加到default box上得到最终的预测边界框
        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]    

        l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l # xmin
        bboxes_in[:, :, 1] = t # ymin
        bboxes_in[:, :, 2] = r # xmax
        bboxes_in[:, :, 3] = b # ymax

        return bboxes_in, F.softmax(scores_in, dim=-1) # 对类别那一维度操作
    

    def batched_nms(self, boxes, scores, idxs, iou_threshold): 
        return torchvision.ops.boxes.batched_nms(boxes, scores, idxs, iou_threshold)

    def decode_single_new(self, bboxes_in, scores_in, criteria, num_output):
        ## type: (Tensor, Tensor, float, int) -> Tuple[Tensor, Tensor, Tensor]
        """
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
        """
        device = bboxes_in.device
        num_classes = scores_in.shape[-1]

        # 对越界的bbox进行裁剪
        bboxes_in = bboxes_in.clamp(min=0, max=1)

        # [8732, 4] -> [8732, 21, 4] 在维度为1的地方复制21次
        bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

        # create lables for each prediction
        labels = torch.arange(num_classes, device=device)
        # [num_classes] -> [8732, num_classes]
        labels = labels.view(1, -1).expand_as(scores_in)

        # remove prediction with the background label
        # 移除归为背景类别的概率信息
        bboxes_in = bboxes_in[:, 1:, :] # [8732, 21, 4] -> [8732, 20, 4]
        scores_in = scores_in[:, 1:] # [8732, 21] -> [8732, 20]
        labels = labels[:, 1:] # [8732, 21] -> [8732, 20]

        # batch everything, by making every class prediction be a separate instance
        # 通过将每个类预测作为一个单独的实例，对所有内容进行批处理
        bboxes_in = bboxes_in.reshape(-1, 4) # [8732, 20, 4] -> [8732x20, 4]
        scores_in = scores_in.reshape(-1) # [8732, 20] -> [8732x20]
        labels = labels.reshape(-1) # [8732, 20] -> [8732x20]

        # remove low scoring boxes
        # 移除低概率目标，self.scores_thresh=0.05
        ids = torch.where(torch.gt(scores_in, 0.05))[0]
        bboxes_in, scores_in, labels = bboxes_in[ids, :], scores_in[ids], labels[ids]

        # remove empty boxes
        ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
        keep = (ws >= 1 / 300) & (hs >= 1 / 300)
        keep = torch.where(keep)[0]
        # 多维数组只填一个。默认取第0维
        bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]

        # NMS
        keep = self.batched_nms(bboxes_in, scores_in, labels, criteria)
        
        # keep only topk scoring predictions
        keep = keep[:num_output]
        bboxes_out = bboxes_in[keep, :]
        scores_out = scores_in[keep]
        labels_out = labels[keep]
        return bboxes_out, labels_out, scores_out


    def forward(self, bboxes_in, scores_in):
        # 通过预测的boxes回归参数得到最终预测目标，将预测目标score通过softmax处理
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in) # probs是预测概率

        # outputs = torch.jit.annotate(List[Tuple[Tensor, Tensor, Tensor]], []) 是否不用这么麻烦
        outputs = []
        
        # 遍历一个batch中的每张image数据
        # bboxes: [batch, 8732, 4]
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            # bbox: [1, 8732, 4]
            bbox = bbox.squeeze(0)      
            prob = prob.squeeze(0)
            outputs.append(self.decode_single_new(bbox, prob, self.criteria, self.max_output))
        return outputs   
