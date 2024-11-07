# 文件夹说明
## data
~~~
存放各类数据
~~~
### flood_forTest
~~~
测试集
~~~
#### images_all
~~~
存放所有测试集图片（1W）
~~~
#### predict_is
~~~
存放测试集中类别为flood_is的图片
~~~
#### predict_no
~~~
存放测试集中类别为flood_no的图片
~~~
### flood_forTrain
~~~
训练集
~~~
#### flood_is
~~~
存放训练集中类别为flood_is的图片
~~~
#### flood_no
~~~
存放训练集中类别为flood_no的图片
~~~
### predict
~~~
存放用于模型推理的相关图片
~~~
#### flood_is
~~~
存放训练集中类别为flood_is的图片
~~~
#### flood_no
~~~
存放训练集中类别为flood_no的图片
~~~
#### images_all
~~~
存放待推理的所有图片
~~~
### golden_label.txt
~~~
所有图片的真实标签
~~~

## models
~~~
存放Swin T-v1的各规模预训练模型
~~~

## weights
~~~
存放每次训练得到的模型
~~~

# 脚本说明
1. do_train.py: 训练脚本
2. do_test.py：测试脚本
3. do_predict.py：推理脚本
4. models.py: 封装好的模型框架
5. utils.py：相关函数
6. data_augmentation.py：数据增强
7. analyse_is_no_score.py：根据模型推理出的各测试集图片的类别概率值可视化各类指标，并采用Soft Categorization Strategy判定最有分类阈值

# 其他文件说明
1. class_indices.json：图片类别对应的索引
