import os
import sys
import json
import pickle
import random
import math
import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images = images[:int(len(images)*0.001)]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

# 自定义 损失函数
class PrecisionLoss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_classes, labels):
        import numpy as np
        import math
        count_pre_0 = len(pred_classes) - int(sum(pred_classes)) # 预测为 flood_is（0） 的个数
        count_pre_1 = len(pred_classes) - count_pre_0 # 预测为 flood_no（1） 的个数

        temp = labels.cpu().numpy() + pred_classes.cpu().numpy() 
        count_pre_0_0 = np.where(temp==0)[0].shape[0] # flood_is（0）预测为 flood_is（0）的个数
        count_pre_1_1 = np.where(temp==2)[0].shape[0] # flood_no（1）预测为 flood_no（1）的个数
        if count_pre_0 == 0:
            precision_0 = 0.0001
        else:
            precision_0 = max(count_pre_0_0/count_pre_0, 0.0001)
        
        if count_pre_1 == 0:
            precision_1 = 0.0001
        else:
            precision_1 = max(count_pre_1_1/count_pre_1, 0.0001)
        # precision = max(b/max(a,0.001), 0.0001)
        loss_0 = torch.tensor(math.log(1/precision_0),requires_grad=True)
        loss_1 = torch.tensor(math.log(1/precision_1),requires_grad=True)

        loss = 0.7 * loss_0 + 0.3 * loss_1
        return loss

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()

    # 加权交叉熵   [flood_is权重, flood_no权重]
    loss_function_weight_cross = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.7, 0.3]).to(device))
    # 自定义损失函数
    loss_function_self = PrecisionLoss() 
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # 加权交叉熵
        loss_weight_cross = loss_function_weight_cross(pred, labels.to(device)) # 增加flood_no（1）预测为flood_is（0）的损失权重
        loss = loss_weight_cross
        # 自定义loss
        # loss_precision = loss_function_self(pred_classes, labels.to(device)) 
        # loss = 0.5 * loss_weight_cross + 0.5 * loss_precision

        # 根据精确率设置loss
        # a = len(pred_classes) - int(sum(pred_classes))
        # temp = labels.numpy() + pred_classes.cpu().numpy()
        # b = np.where(temp==0)[0].shape[0]
        # precision = b/a
        # loss = torch.tensor(math.log(1/precision),requires_grad=True).to(device)


        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    # 加权交叉熵   [flood_is权重, flood_no权重]
    loss_function_weight_cross = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.7, 0.3]).to(device))
    # 自定义损失函数
    loss_function_self = PrecisionLoss() 

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # 加权交叉熵
        loss_weight_cross = loss_function_weight_cross(pred, labels.to(device))
        loss = loss_weight_cross
        # 自定义loss
        # loss_precision = loss_function_self(pred_classes, labels.to(device)) 
        # loss = 0.5 * loss_weight_cross + 0.5 * loss_precision

        

        # 根据精确率设置loss
        # a = len(pred_classes) - int(sum(pred_classes))
        # temp = labels.numpy() + pred_classes.cpu().numpy()
        # b = np.where(temp==0)[0].shape[0]
        # precision = max(b/a, 0.0001)
        # loss = torch.tensor(math.log(1/precision),requires_grad=True).to(device)

        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
