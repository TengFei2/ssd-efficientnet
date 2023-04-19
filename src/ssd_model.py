import torch
import math
from torch import nn 
from torch.jit.annotations import List

# 自定义的模块 https://www.jb51.net/article/237102.htm
# from .resnet import resnet50
from .efficientnet import efficientnet_b0
from .utils import dboxes300_coco, PostProcess

class Backbone(nn.Module):
    def __init__(self, pretrain_path=None) -> None:
        super().__init__()
        net = efficientnet_b0()
        self.out_channels = [40, 512, 512, 256, 256, 256] # 六个feature map的通道
        # self.out_channels = [40, 512]

        if pretrain_path:
            net.load_state_dict(torch.load(pretrain_path))
        
        self.feature_extractor = nn.Sequential(*list(net.features.children())[:6])


    def forward(self, x):
        return self.feature_extractor(x)

    
class SSD300(nn.Module):
    def __init__(self, backbone=None, num_classes=21):
        super().__init__()
        if backbone is None:
            raise Exception('backbone is None')
        if not hasattr(backbone, 'out_channels'):
            raise Exception("the backbone don't have attribute: out_channels")
        self.feature_extractor = backbone

        self.num_classes = num_classes

        # out_channels = [1024, 512, 512, 256, 256, 256] for resnet50
        self.build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]


        '''添加位置和置信度处理'''
        location_extractors = []
        confidence_extractors = []
        
        for oc, nd in zip(self.feature_extractor.out_channels, self.num_defaults):
            # nd is the number of default boxes, oc is the output channel
            location_extractors.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            confidence_extractors.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))
        self.loc = nn.ModuleList(location_extractors)
        self.conf = nn.ModuleList(confidence_extractors)
        self.__init_weights()   

        default_box = dboxes300_coco()
        self.compute_loss = Loss(default_box)
        # self.encoder = Encoder(default_box) 
        self.postprocess = PostProcess(default_box)

    def build_additional_features(self, input_size):
        '''为backbone添加额外的一系列卷积层，得到相应的一系列的特征提取器'''
        additional_blocks = []

        # input_size = [1024, 512, 512, 256, 256, 256] for resnet50
        middle_channels = [256, 256, 128, 128, 128]

        for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
            padding, stride = (1, 2) if i <3 else (0, 1)
            layer = nn.Sequential(
                nn.Conv2d(input_ch, middle_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_ch, output_ch, kernel_size=3, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(output_ch),
                nn.ReLU(inplace=True))
            additional_blocks.append(layer)
        self.additional_blocks = nn.ModuleList(additional_blocks)
    
    def __init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
    


    def bbox_view(self, features, loc_extractor, conf_extractor):
        '''与代码中不同，此处改用了reshape'''
        locs = []
        confs = []
        for f, l, c in zip(features, loc_extractor, conf_extractor):
            # [batch, n*4, feat_size, feat_size] -> [batch, 4, -1]
            locs.append(l(f).reshape(f.shape[0], 4, -1))
            # [batch, n*classes, feat_size, feat_size] -> [batch, classes, -1]
            confs.append(c(f).reshape(f.shape[0], self.num_classes, -1))

        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, image, targets=None):
        x = self.feature_extractor(image)

        # Feature Map Map 38x38x1024, 19x19x512, 10x10x512, 5x5x256, 3x3x256, 1x1x256
        # detection_features = torch.jit.annotate(List[torch.Tensor], []) # 该函数是一个传递函数，返回值为the_value，用于提示TorchScript编译器the_value的类型。当该函数运行在TorchScript以外时，为空操作。
        detection_features = []
        detection_features.append(x)
        for layer in self.additional_blocks:
            x = layer(x)
            detection_features.append(x)

        locs, confs = self.bbox_view(detection_features, self.loc, self.conf)
        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        # 38x38x4 + 19x19x6 + 10x10x6 + 5x5x6 + 3x3x4 + 1x1x4 = 8732

        if self.training:
            if targets is None:
                raise ValueError('In training mode, targets should be passed')
            # bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            bboxes_out = targets['boxes']
            bboxes_out = bboxes_out.transpose(1, 2).contiguous()
            label_out = targets['labels']

            # pred_loc, pred_label, gloc, glabel
            # pred_loc是预测的回归参数，并不是真实的坐标
            loss = self.compute_loss(locs, confs, bboxes_out, label_out)
            return {"total_losses": loss}
        # 将预测回归参数叠加到default box上得到最终预测box，并执行NMS抑制滤除重叠框
        results = self.postprocess(locs, confs)
        return results


class Loss(nn.Module):
    '''
     Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    '''
    def __init__(self, dboxes):
        super().__init__()
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.scale_xy = 1.0 / dboxes.scale_xy # 10
        self.scale_wh = 1.0 / dboxes.scale_wh # 5

        self.location_loss = nn.SmoothL1Loss(reduction='none')

        # [num_anchors, 4] -> [4, num_anchors] -> [1, 4, num_anchors]
        # https://blog.csdn.net/qq_43391414/article/details/120484239
        ''' Parameter将一个不能进行参数优化的变成一个可以进行参数优化的，并自动包含于model.parameters() '''
        self.dboxes = nn.Parameter(dboxes(order='xywh').transpose(0, 1).unsqueeze(0), requires_grad=False)
        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')

    def _location_vec(self, loc):
        '''
        Generate Location Vectors
        计算ground truth相对anchors的回归参数
        param loc: anchor匹配到的对应GTBOX Nx4x8732
        '''
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :] # Nx2x8732
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()  # Nx2x8732

        return torch.cat((gxy, gwh), dim=1).contiguous()


    def forward(self, ploc, plabel, gloc, glabel):
        """
            predicted location and labels
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                
            ground truth location and labels
            gloc, glabel: Nx4x8732, Nx8732
        """

        # h获取正样本的mask Tensor: [N, 8732]（是在transforer中处理的）
        # N为batch_size的大小
        mask = torch.gt(glabel, 0) # (gt: >)
        # 计算一个batch中每张图片的正样本个数
        pos_num = mask.sum(dim=1)
        
        # 计算gt的location回归参数 [N, 4, 8732]
        # ploc是网络预测的回归参数，vec_gd是由dbox与它所匹配的gt这两个框之间的差距（相当于这次训练中的真实回归参数）就是他们两个之间计算损失，后面再根据损失调整预测值
        vec_gd = self._location_vec(gloc)
        
        # sum on four coordinates, and mask
        # 计算定位损失，只有正样本
        loc_loss = self.location_loss(ploc, vec_gd).sum(dim=1) # Tensor: [N, 8732]
        loc_loss = (mask.float() * loc_loss).sum(dim=1) # Tensor: [N]

        # 分类所有的损失 Tenosr: [N, 8732]
        con = self.confidence_loss(plabel, glabel)


        # positive mask will never be selected
        # 获取负样本
        con_neg = con.clone()
        con_neg[mask] = 0.0 # 将正样本损失置为0

        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1) # 先降序，再升序得到需要负样本的蒙版

        # number of negative three times positive
        # 用于loss计算的负样本的个数是正样本个数的三倍，总数不超过8732
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = torch.lt(con_rank, neg_num) # (lt: <) Tensor [N, 8732]

        # confidence最终loss使用选取的正样本loss+选取的负样本loss
        con_loss = (con * (mask + neg_mask).float()).sum(dim=1) # Tensor [N]


        # avoid no boject detected 避免出现图像中没有GTBOX的情况
        # 没有GTBOX的图像的loss不计算
        total_loss = loc_loss + con_loss
        # eg. [15, 3, 5, 0] -> [1.0, 1.0, 1.0, 0.0]
        num_mask = torch.gt(pos_num, 0).float() # 统计一个batch中的每张图像中是否存在正样本
        pos_num = pos_num.float().clamp(min=1e-6) # 防止出现分母为0的情况，与原代码中不同，pos_num是一个大于等于0的数，而num_mask又将pos_num=0的给排除掉了，因而可以定一个最小值1
        ret = (total_loss * num_mask / pos_num).mean(dim=0) # 只计算存在正样本的损失
        return ret

