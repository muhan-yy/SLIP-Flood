# SLIP-Flood: Soft-combination of Swin Transformer and Lightweight Language-Image Pre-training for Flood Images Classification

## 项目介绍


本项目的目标在于构建适用于洪灾监测领域多个下游任务（图文分类、图文检索）的图文跨模态预训练模型，并构建两类大规模的、适用于洪灾监测领域的开源图文基础数据集。

项目名称为**SLIP-Flood**，包括两个模块：[FICM](#FICM)、[FTIRM](#FTIRM)。其中，**FICM**负责洪灾图片分类任务，**FTIRM**负责洪灾图文检索、辅助洪灾图片分类任务以及辅助洪灾文本分类任务。

本项目最终将开源两类图文基础数据集：**FloodMulS**、**FloodIT**。数据集的详细说明见章节[数据集说明](#数据集说明)

## 数据集说明
### FloodMulS

数据集**FloodMulS**用于训练**SLIP-Flood**中的**FICM**模块。可通过[下载链接]()获取。

**FloodMulS**包括46.5万张图片，其中训练数据集有45.5万张，测试数据集有1万张。
训练数据集中数据增强的图片占比为25%，本项目采用6种数据增强方式：随机选择、调整亮度、调整对比度、颜色抖动、添加高斯噪声和添加椒盐噪声，每张图片随机采用某一种数据增强方式。示例如下图：

![FloodMulS](./imagesForReadme/FloodMulS.png)

### FloodIT

数据集**FloodIT**用于训练**SLIP-Flood**中的**FTIRM**模块。可通过[下载链接]()获取。

**FloodIT**包括23.7万个图文对数据，每个图文对数据包括：1张图片、1个中文描述标题、1个中文类别标签以及5个英文描述文本。因此，**FloodIT**可被转变成118.5万个图文对数据。此外，本项目构建了1万个图文对数据用于测试模型性能，每个测试数据包括：1张图片、1个中文描述标题以及1个中文类别标签。示例如下图：

![FloodIT](./imagesForReadme/FloodIT.png)

## 环境部署

安装项目环境前请先确保已拥有python环境、conda环境（可选）、python编译器（默认Visual Studio Code）

1. 进入项目根目录
```
    cd ./   
```
2. 新建新的环境
```
    conda create --name SLIP-Flood python==3.9
    conda activate SLIP-Flood
```
3. 安装CUDA、Pytorch等深度学习环境，参考[pytorch官网](https://pytorch.org/)
```
    # 此处示例默认CUDA版本为12.4
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
4. 安装所需第三方库
```
    pip install -r requirements.txt
```

## 项目文件说明

本项目分为两个模块**FICM**、**FTIRM**，下面将对两个模块的项目文件作出详细说明。

### FICM

该模块的所有文件都位于路径`/FICM`中，除了有特殊说明，本小节的所有路径都是以`/FICM`为根目录的。

1. `./data/`
用于存放**FICM**所需的各类数据，包括：
    - `./flood_forTrain`：训练数据集，`flood_is`用于存放类别为“与洪灾相关”的图片，`flood_no`用于存放类别为“与洪灾不相关”的图片
    - `./flood_forTest`：测试数据集，`images_all`存放所有用于测试的图片，`predict_is`存放模型预测为“与洪灾相关”的图片，`predict_no`存放模型预测为“与洪灾不相关”的图片
    - `./predict`：存放用于模型推理的相关图片，`images_all`存放所有用于推理的图片，`flood_is`存放模型预测为“与洪灾相关”的图片，`flood_no`存放模型预测为“与洪灾不相关”的图片
    - `./golden_label.txt`：图片的真实标签

2. `./models/`
用于存放SwinT各规模的预训练模型，可从此链接下载：[SwinT](https://github.com/microsoft/Swin-Transformer)

3. `./weights/`
用于存放**FICM**模块训练获得的预训练模型

4. 脚本/文件说明
   - `./do_train.py`：训练脚本 [待发布]
   - `./do_test.py`：测试脚本 [待发布]
   - `./do_predict.py`：推理脚本 [待发布]
   - `./models.py`：封装好的模型框架 [待发布]
   - `./utils.py`：相关函数 [待发布]
   - `./data_augmentation.py`：数据增强
   - `./analyse_is_no_score.py`：根据模型推理出的各测试集图片的类别概率值可视化各类指标，并采用Soft Categorization Strategy判定最有分类阈值
   - `./class_indices.json`：图片类别对应的索引

### FTIRM

该模块的所有文件都位于路径`/FTIRM`中，除了有特殊说明，本小节的所有路径都是以`/FTIRM`为根目录的。

1. `./checkpoints/`
用于存放**FTIRM**模型训练获得的预训练模型

2. `./data/`
用于存放**FTIRM**所需的各类数据，包括：
    - `./images`：用于存放用于不同功能的图片数据，`train`存放用于训练数据集，`test`存放用于测试数据集，`predict`存放用于模型测试的1万张图片，`flood_is`存放`predict`中类别为“与洪灾相关”的图片，`flood_no`存放`predict`中类别为“与洪灾不相关”的图片，`predict-is`存放被模型预测为“与洪灾相关”的图片，`predict-no`存放被模型预测为“与洪灾不相关”的图片。
      不同类型的数据集可根据'./data/'中的tsv文件以及'images_all'中所有的图片获取
    - `images_all`：存放数据集**FloodIT**对应的所有图片

1. `./data/`中的tsv文件
    - `./example_predict-label.tsv`：`predict`中1万张图片及其中文标签
    - `./example_predict-title.tsv`：`predict`中1万张图片及其中文标题
    - `./example_test-label.tsv`：`test`中所有图片及其中文标签
    - `./example_test-label.tsv`：`test`中所有图片及其中文标题
    - `./example_train-label.tsv`：`train`中所有图片及其中文标签
    - `./example_train-label.tsv`：`train`中所有图片及其中文标题
  
2. `./models/`
存放用于初始化**FTIRM**中图片编码器与文本编码器的预训练模型
    - `./chinese-roberta-wwm-ext`：初始化文本编码器，模型规模为base
    - `./chinese-roberta-wwm-ext-large`：初始化文本编码器，模型规模为large
    - `./vit-large-patch16-224`：初始化图片编码器，模型规模为large

1. `./result/`
存放各类模型测试结果

1. 脚本说明
   - `./train.py`：训练脚本[待发布]
   - `./model.py`：模型脚本[待发布]
   - `./imageClassify.py`：图片分类脚本[待发布]
   - `./imageRetrieve.py`：图文检索脚本[待发布]
   - `./textClassify.py`：辅助文本分类脚本[待发布]
   - `./analyse_is_no_score.py`：针对模型预测出的图片各类别概率值进行分析

## 预训练模型
本项目分别针对**FICM**、**FTIRM**开展不同的训练实验，最终得到不同规模的模型。
### FICM
本项目分别采用三个规模的SwinT预训练模型作为初始化模型，并开展微调工作，最终得到对应的**FICM**模型：
- SLIP-Flood_FICM_Tiny：SwinT预训练模型的规模为Tiny，[下载链接](https://huggingface.co/muhan-yy/SLIP-Flood_FICM_Tiny) [待发布]
- SLIP-Flood_FICM_Base：SwinT预训练模型的规模为Base，[下载链接](https://huggingface.co/muhan-yy/SLIP-Flood_FICM_Base) [待发布]
- SLIP-Flood_FICM_Large：SwinT预训练模型的规模为Large，[下载链接](https://huggingface.co/muhan-yy/SLIP-Flood_FICM_Large) [待发布]

### FTIRM
本项目分别采用两个规模的预训练模型用于初始化**FTIRM**的文本编码器，图片编码器只采用vit-large-patch16-224，最终得到对应的**FTIRM**模型：
- SLIP-Flood_FTIRM_Base_Large：文本编码器规模为Base，[下载链接](https://huggingface.co/muhan-yy/SLIP-Flood_FTIRM_Base_Large) [待发布]
  ```
    best_checkpoint_evalLoss.pt: 基于eval loss获取的最优模型
    best_checkpoint_score.pt: 基于score获取的最优模型
    best_checkpoint_trainLoss.pt: 基于train loss获取的最优模型
  ```
- SLIP-Flood_FTIRM_Large_Large：文本编码器规模为Large，[下载链接](https://huggingface.co/muhan-yy/SLIP-Flood_FTIRM_Large_Large) [待发布]
  ```
    best_checkpoint_evalLoss.pt: 基于eval loss获取的最优模型
    best_checkpoint_score.pt: 基于score获取的最优模型
    best_checkpoint_trainLoss.pt: 基于train loss获取的最优模型
  ```
