import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        # __init__函数是这个类的初始化函数，根据指定的图片路径，读取所有图片数据，存放到self.imgs_path列表中。
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.jpg'))
        self.label_path = glob.glob(os.path.join(data_path, 'label/*.png'))


    def augment(self, image, flipCode):
        # augment函数是定义的数据增强函数
        # 在这个类中，你不用进行一些打乱数据集的操作，也不用管怎么按照 batchsize 读取数据。
        # 因为实例化这个类后，我们可以用 torch.utils01.data.DataLoader 方法指定 batchsize 的大小，决定是否打乱数据。

        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # __getitem__函数是数据获取函数，在这个函数里你可以写数据怎么读，怎么处理，并且可以一些数据预处理、数据增强都可以在这里进行。

        # 根据index读取图片
        image_path = self.imgs_path[index]
        label_path = self.label_path[index]
        # 根据image_path生成label_path
        # label_path = image_path.replace('image', 'label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

    
if __name__ == "__main__":
    isbi_dataset = ISBI_Loader(r"D:\pycharm\day1\torch_file\image segmentation\unet-8-10-2021\data/train/")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=1,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)