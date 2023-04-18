import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt


import transforms
from load_test_data import LoadImages
from src.ssd_model import SSD300, Backbone
from draw_box_utils import draw_objs




# 构建SSD模型
def create_model(num_classes):
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'predicting on the {device}')

    # create model and load weight
    num_classes = 3 + 1 # 三类物体，一类背景
    model = create_model(num_classes)

    weights_path = r'runs\train\ssd300-29.pth'
    weight_dict = torch.load(weights_path, map_location='cpu')
    weight_dict = weight_dict['model'] if 'model' in weight_dict else weight_dict
    model.load_state_dict(weight_dict)
    model.to(device)

    # read class_indict
    json_path = r'src\camera_classes.json'
    assert os.path.exists(json_path), f'file {json_path} does not exist.'
    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    num_to_class = {str(v): str(k) for k, v in class_dict.items()}
    json_file.close()


    
    # 加载数据
    trans = transforms.Compose([transforms.Resize(), transforms.ToTensor(), transforms.Normalization()])
    dataset = LoadImages(path=args.data_path, transforms=trans)
    model.eval()
    with torch.no_grad():
        init_img = torch.zeros((1, 3, 300, 300), device=device)
        model(init_img)

        time_start = time.time()
        for im0, im, file_name in dataset:
            if len(im) == 3:
                im = im[None]
            preds = model(im.to(device))[0] # bboxes_out, labels_out, scores_out
            preds_boxes = preds[0].to('cpu').numpy()
            preds_classes = preds[1].to('cpu').numpy()
            preds_scores = preds[2].to('cpu').numpy()

            if len(preds_boxes) == 0:
                print('No targets have been detected in the image!!')
                return ''
            
            # 过滤掉预测到的小概率物体
            idx = np.greater(preds_scores, args.conf_thresh)
            preds_boxes = preds_boxes[idx]
            preds_boxes[:, [0, 2]] = preds_boxes[:, [0, 2]] * im0.size[0]
            preds_boxes[:, [1, 3]] = preds_boxes[:, [1, 3]] * im0.size[1]


            preds_classes = preds_classes[idx]
            preds_scores = preds_scores[idx]

            plot_img = draw_objs(im0,
                                preds_boxes,
                                preds_classes,
                                preds_scores,
                                category_index=num_to_class,
                                box_thresh=args.conf_thresh,
                                line_thickness=3,
                                font='arial.ttf')

            # 保存预测的图片结果
            Path('./runs/detect').mkdir(exist_ok=True, parents=True)
            plot_img.save(f'runs/detect/{file_name}_result.jpg')


if __name__ == '__main__':
    os.system('cls')

    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--data_path', default='testdata/', help='dataset')
    parser.add_argument('--conf_thresh', default=0.5, type=float, help='conf_thresh')
    parser.add_argument('--show_img', default=False, type=bool, help='show images or not')
    
    args = parser.parse_args()
    print(args)
    main(args)



