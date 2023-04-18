# --------------------------------------------#
#   该部分代码用于看网络结构
# --------------------------------------------#
import torch
from thop import clever_format, profile
from torch.backends import cudnn
import time
from src.ssd_model import SSD300, Backbone


def compute_speed(model, input_size, device, iteration=500):   # 这个iteration的作用是预热cpu
    torch.cuda.set_device(device)
    cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    input = torch.randn(*input_size, device=device)

    for _ in range(50):
        model(input)

    print('=========Eval Forward Time=========')
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))
    return speed_time, fps


if __name__ == "__main__":
    with torch.no_grad():
        input_shape = [300, 300]
        num_classes = 4
        backbone = Backbone()
        model = SSD300(backbone, num_classes)


        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        m = model.cuda()
        # for i in m.children():
        #     print(i)
        #     print('==============================')

        dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
        flops, params = profile(m.to(device), (dummy_input,), verbose=False)
        # --------------------------------------------------------#
        #   flops * 2是因为profile没有将卷积作为两个operations
        #   有些论文将卷积算乘法、加法两个operations。此时乘2
        #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
        #   本代码选择乘2，参考YOLOX。
        # --------------------------------------------------------#
        # flops = flops * 2
        flops, params = clever_format([flops, params], "%.3f")
        print('Total GFLOPS: %s' % (flops))
        print('Total params: %s' % (params))

        # -------------------------计算fps------------------------ #
        # model = YoloBody(anchors_mask, num_classes, False)
        speed_time, fps = compute_speed(m, (1, 3, 300, 300), device=0)
        print(speed_time)
        print(fps)
