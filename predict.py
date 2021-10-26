import glob
import numpy as np
import torch
import os
import cv2
from model2.unet1 import Unet

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载网络，图片单通道，分类为1。
    net = Unet(1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob('data/test/image/*.jpg')
    # print(tests_path)

    # 遍历所有图片
    for test_path in tests_path:
        # 保存标签结果地址
        test_label_path = test_path.replace('image', '/result')
        save_res_path = test_label_path.split('.')[0] + '_res.jpg'
        # print(save_res_path)
        # 读取图片
        img = cv2.imread(test_path)
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # print(pred)
        # np.set_printoptions(threshold=np.inf)
        # 保存图片
        cv2.imwrite(save_res_path, pred)