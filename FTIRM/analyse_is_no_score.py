import os
import numpy as np
import matplotlib.pyplot as plt

golden_label_file = './data/golden_label.txt'
image_pre_score_file = './data/image_is_no_score_end_rl_vl_evalLoss-label-large.txt'

golden_label_aa = {} # 图片真实类别
with open(golden_label_file, 'r') as f:
    for row in f.readlines():
        row = row.strip()
        image_name, image_label = row.split('\t')
        golden_label_aa[image_name] = image_label
# print(golden_label)

thresholds = np.arange(0,1.001,0.001)

image_is_scores = [] # 所有图片预测为flood_is的得分
image_no_scores = [] # 所有图片预测为flood_no的得分
image_pre_score_data = np.loadtxt(image_pre_score_file, dtype=str, delimiter='\t')
# image_is_scores = image_pre_score_data[:,1].astype(float)
# image_no_scores = image_pre_score_data[:,2].astype(float)
image_name = image_pre_score_data[:,0]
golden_label = {}
for item in image_name:
    golden_label[item] = golden_label_aa[item]
golden_label_sorted = sorted(golden_label.items())
golden_label.clear()
golden_label = {key:value for key, value in golden_label_sorted}

x = list(range(image_pre_score_data.shape[0]))
# 绘制直方图

# plt.hist(image_is_scores, label='image_is_scores', bins=60, color='red', alpha=0.5)
# plt.hist(image_no_scores, label='image_no_scores', bins=60, color='orange', alpha=0.5)

# plt.legend()
# plt.show()

# 不同的类别判定策略：分数大的、flood_is得分大于flood_no得分超过阈值
def recall_precision(goldens, pres):
    is_is = 0
    is_no = 0
    no_no = 0
    no_is = 0
    for golden, pre in zip(goldens, pres):
        if golden == 'flood_is' and pre == 'flood_is':
            is_is += 1
        if golden == 'flood_is' and pre == 'flood_no':
            is_no += 1
        if golden == 'flood_no' and pre == 'flood_is':
            no_is += 1
        if golden == 'flood_no' and pre == 'flood_no':
            no_no += 1
    # recall_is_max = round(is_is / max((is_is + is_no), 1), 4)
    # recall_no_max = round(no_no / max((no_no + no_is), 1), 4)

    # precision_is_max = round(is_is / max((is_is + no_is), 1), 4)
    # precision_no_max = round(no_no / max((is_no + no_no), 1), 4)

    recall_is_max = round(is_is / (is_is + is_no), 4)
    recall_no_max = round(no_no / (no_no + no_is), 4)

    precision_is_max = round(is_is / (is_is + no_is), 4)
    precision_no_max = round(no_no / (is_no + no_no), 4)

    return recall_is_max, precision_is_max, recall_no_max, precision_no_max

# 策略1：分数大的
def score_max():
    
    F1_iss = []
    F1_nos = []
    recall_is_maxs = []
    precision_is_maxs = []
    recall_no_maxs = []
    precision_no_maxs = []
    for index in range(1,1002):
        recall_is_max = 0.0
        precision_is_max = 0.0
        F1_is = 0.0
        F1_no = 0.0
        recall_no_max = 0.0
        precision_no_max = 0.0
        pre_label = {}
        for item in image_pre_score_data:
            image_name = item[0]
            pre_scores = eval(item[index])
            if float(pre_scores[0]) > float(pre_scores[1]):
                pre_label[image_name] = 'flood_is'
            else:
                pre_label[image_name] = 'flood_no'
        pre_label_sorted = sorted(pre_label.items())
        pre_label.clear()
        pre_label = {key:value for key, value in pre_label_sorted}
        recall_is_max, precision_is_max, recall_no_max, precision_no_max = recall_precision(list(golden_label.values()), list(pre_label.values()))
        F1_is = round((2 * recall_is_max * precision_is_max) / (recall_is_max + precision_is_max), 4)
        F1_no = round((2 * recall_no_max * precision_no_max) / (recall_no_max + precision_no_max), 4)

        F1_iss.append(F1_is)
        F1_nos.append(F1_no)
        recall_is_maxs.append(recall_is_max)
        precision_is_maxs.append(precision_is_max)
        recall_no_maxs.append(recall_no_max)
        precision_no_maxs.append(precision_no_max)

    return F1_iss, F1_nos, recall_is_maxs, precision_is_maxs, recall_no_maxs, precision_no_maxs
# 策略2：阈值
def score_threshold():

    F1_iss = []
    F1_is_max_thresholds = []
    F1_nos = []
    # F1_no_max_thresholds = []
    recall_is_maxs = []
    precision_is_maxs = []
    recall_no_maxs = []
    precision_no_maxs = []
    for index in range(1,1002):
        recall_is_threshold = []
        precision_is_threshold = []
        F1_is = []
        
        recall_no_threshold = []
        precision_no_threshold = []
        F1_no = []

        # 阈值增加 recall递减、precision递增，没有分析的必要
        thresholds_diff = np.arange(0,1,0.01).tolist() # 0.2-0.9  0.05
        image_names = image_pre_score_data[:,0].tolist()
        temp = np.array([eval(row)[0]-eval(row)[1] for row in image_pre_score_data[:,index]])
        temp = temp.reshape(temp.shape[0],1)
        temp_new = np.concatenate([temp] * len(thresholds_diff), axis=1)
        threshold_matrix = np.repeat(np.array(thresholds_diff)[None, :], temp_new.shape[0], axis=0)
        pre_label_matrix = temp_new - threshold_matrix
        pre_labels = np.where(pre_label_matrix >= 0, "flood_is", "flood_no")
        for index_threshold in range(len(thresholds_diff)):
            pre_label = {image_names[index]:pre_labels[index,index_threshold] for index in range(len(image_names))}
            pre_label_sorted = sorted(pre_label.items())
            pre_label.clear()
            pre_label = {key:value for key, value in pre_label_sorted}

            recall_is, precision_is, recall_no, precision_no = recall_precision(list(golden_label.values()), list(pre_label.values()))
            recall_is_threshold.append(recall_is)
            precision_is_threshold.append(precision_is)
            F1_is.append(round((2 * recall_is * precision_is) / (recall_is + precision_is), 4))
            recall_no_threshold.append(recall_no)
            precision_no_threshold.append(precision_no)
            F1_no.append(round((2 * recall_no * precision_no) / (recall_no + precision_no), 4))

        # for threshold in thresholds_diff:
        #     pre_label = {}

            

        #     for item in image_pre_score_data:
        #         image_name = item[0]
        #         pre_scores = eval(item[index])

        #         if float(pre_scores[0]) - float(pre_scores[1]) >= threshold:
        #             pre_label[image_name] = 'flood_is'
        #         else:
        #             pre_label[image_name] = 'flood_no'
            

        # print(f"策略：阈值\n指标：\nrecall_is:\n{recall_is_threshold}\nprecision_is:\n{precision_is_threshold}\nrecall_no:\n{recall_no_threshold}\nprecision_no:\n{precision_no_threshold}\n")
        # print(f"F1_is:\n{F1_is}\nF1_no:\n{F1_no}")
        F1_is_max = max(F1_is) # F1_is的最大值
        F1_is_max_threhold = thresholds_diff[F1_is.index(F1_is_max)] # F1_is的最大值对应的阈值
        # F1_no_max = max(F1_no) # F1_is的最大值
        F1_no_now = F1_no[F1_is.index(F1_is_max)]
        # F1_no_max_threhold = thresholds_diff[F1_no.index(F1_no_max)] # F1_is的最大值对应的阈值
        recall_is_max = recall_is_threshold[F1_is.index(F1_is_max)]
        recall_no_max = recall_no_threshold[F1_is.index(F1_is_max)]

        precision_is_max = precision_is_threshold[F1_is.index(F1_is_max)]
        precision_no_max = precision_no_threshold[F1_is.index(F1_is_max)]

        F1_iss.append(F1_is_max)
        F1_is_max_thresholds.append(F1_is_max_threhold)
        # F1_no_max_thresholds.append(F1_no_max_threhold)
        F1_nos.append(F1_no_now)
        recall_is_maxs.append(recall_is_max)
        precision_is_maxs.append(precision_is_max)
        recall_no_maxs.append(recall_no_max)
        precision_no_maxs.append(precision_no_max)
        print(f"{index}/1002")
        
        # print("\n策略：阈值")
        # print(f"F1_is_max:\t{F1_is_max}")
        # print(f"F1_is_max_threhold:\t{F1_is_max_threhold}")
        # print(f"F1_no_max:\t{F1_no_max}")
        # print(f"F1_no_max_threhold:\t{F1_no_max_threhold}")
        # print(f"recall_is_max:\t{recall_is_max}")
        # print(f"precision_is_max:\t{precision_is_max}")
        # print(f"recall_no_max:\t{recall_no_max}")
        # print(f"precision_no_max:\t{precision_no_max}")
        
        # plt.plot(thresholds_diff, F1_is, label="F1_is", color='red', linestyle='-')
        # plt.plot(thresholds_diff, F1_no, label="F1_no", color='red', linestyle='--')

        # plt.plot(thresholds_diff, recall_is_threshold, label="recall_is_threshold", color='blue', linestyle='-')
        # plt.plot(thresholds_diff, recall_no_threshold, label="recall_no_threshold", color='blue', linestyle='--')
        
        # plt.plot(thresholds_diff, precision_is_threshold, label="precision_is_threshold", color='green', linestyle='-')
        # plt.plot(thresholds_diff, precision_no_threshold, label="precision_no_threshold", color='green', linestyle='--')

        # plt.axvline(x=thresholds_diff[F1_is.index(max(F1_is))], color='blue', linestyle='--', label=f"F1_is_max {F1_is_max}-{F1_is_max_threhold}")

        # plt.legend()
        # plt.show()
    return F1_iss, F1_is_max_thresholds, F1_nos, recall_is_maxs, precision_is_maxs, recall_no_maxs, precision_no_maxs

### 直接比较大小 ##
# F1_iss, F1_nos, recall_iss, precision_iss, recall_nos, precision_nos = score_max()
# F1_is_max = max(F1_iss)
# F1_is_max_threshold = thresholds[np.argmax(F1_iss)]
# F1_nos_max = max(F1_nos)
# F1_nos_max_threshold = thresholds[np.argmax(F1_nos)]
# recall_is_max = max(recall_iss)
# recall_is_max_threshold = thresholds[np.argmax(recall_iss)]
# precision_is_max = max(precision_iss)
# precision_is_max_threshold = thresholds[np.argmax(precision_iss)]
# recall_no_max = max(recall_nos)
# recall_no_max_threshold = thresholds[np.argmax(recall_nos)]
# precision_no_max = max(precision_nos)
# precision_no_max_threshold = thresholds[np.argmax(precision_nos)]

# print(f"F1_is:\t{F1_is_max}\tthreshold:{F1_is_max_threshold}")
# print(f"F1_no:\t{F1_nos_max}\tthreshold:{F1_nos_max_threshold}")
# print(f"recall_is:\t{recall_is_max}\tthreshold:{recall_is_max_threshold}")
# print(f"precision_is:\t{precision_is_max}\tthreshold:{precision_is_max_threshold}")
# print(f"recall_no:\t{recall_no_max}\tthreshold:{recall_no_max_threshold}")
# print(f"precision_no:\t{precision_no_max}\tthreshold:{precision_no_max_threshold}")

# plt.plot(thresholds, F1_iss, label="F1_is", color='red', linestyle='-')
# plt.plot(thresholds, F1_nos, label="F1_no", color='red', linestyle='--')

# plt.plot(thresholds, recall_iss, label="recall_is", color='blue', linestyle='-')
# plt.plot(thresholds, precision_iss, label="precision_is", color='blue', linestyle='--')

# plt.plot(thresholds, recall_nos, label="recall_no", color='green', linestyle='-')
# plt.plot(thresholds, precision_nos, label="precision_no", color='green', linestyle='--')
# plt.legend()
# plt.show()

### 根据阈值 ###
F1_iss, F1_is_max_thresholds, F1_nos, recall_is_maxs, precision_is_maxs, recall_no_maxs, precision_no_maxs = score_threshold()
with open('indicator.txt', 'a', encoding='utf-8') as fi:
    for index in range(len(F1_iss)):
        fi.write(str(F1_iss[index]) + '\t' + str(F1_is_max_thresholds[index]) + '\t' + str(F1_nos[index]) + '\t' + 
                 str(recall_is_maxs[index]) + '\t' + str(precision_is_maxs[index]) + '\t' + str(recall_no_maxs[index]) + '\t' + str(precision_no_maxs[index]) + '\n')
F1_is_max = max(F1_iss)
F1_is_max_threshold = thresholds[np.argmax(F1_iss)]
F1_is_max_threshold_diff = F1_is_max_thresholds[np.argmax(F1_iss)]
F1_nos_max = max(F1_nos)
F1_nos_max_threshold = thresholds[np.argmax(F1_nos)]
recall_is_now = recall_is_maxs[np.argmax(F1_iss)]
recall_is_max = max(recall_is_maxs)
recall_is_max_threshold = thresholds[np.argmax(recall_is_maxs)]
precision_is_now = precision_is_maxs[np.argmax(F1_iss)]
precision_is_max = max(precision_is_maxs)
precision_is_max_threshold = thresholds[np.argmax(precision_is_maxs)]
recall_no_now = recall_no_maxs[np.argmax(F1_iss)]
recall_no_max = max(recall_no_maxs)
recall_no_max_threshold = thresholds[np.argmax(recall_no_maxs)]
precision_no_now = precision_no_maxs[np.argmax(F1_iss)]
precision_no_max = max(precision_no_maxs)
precision_no_max_threshold = thresholds[np.argmax(precision_no_maxs)]

print(f"F1_is_max:\t{F1_is_max}\tthreshold:{F1_is_max_threshold}")
print(f"F1_is_max:\t{F1_is_max}\tthreshold:{F1_is_max_threshold}\tthreshold_diff:{F1_is_max_threshold_diff}")
print(f"recall_is_now:\t{recall_is_now}")
print(f"recall_is_max:\t{recall_is_max}\tthreshold:{recall_is_max_threshold}")
print(f"precision_is_now:\t{precision_is_now}")
print(f"precision_is_max:\t{precision_is_max}\tthreshold:{precision_is_max_threshold}")
print(f"recall_no_now:\t{recall_no_now}")
print(f"recall_no_max:\t{recall_no_max}\tthreshold:{recall_no_max_threshold}")
print(f"precision_no_now:\t{precision_no_now}")
print(f"precision_no_max:\t{precision_no_max}\tthreshold:{precision_no_max_threshold}")

plt.plot(thresholds, F1_iss, label="F1_is", color='red', linestyle='-')
plt.plot(thresholds, F1_nos, label="F1_no", color='red', linestyle='--')

plt.plot(thresholds, recall_is_maxs, label="recall_is", color='blue', linestyle='-')
plt.plot(thresholds, precision_is_maxs, label="precision_is", color='blue', linestyle='--')

plt.plot(thresholds, recall_no_maxs, label="recall_no", color='green', linestyle='-')
plt.plot(thresholds, precision_no_maxs, label="precision_no", color='green', linestyle='--')
plt.legend()
plt.show()