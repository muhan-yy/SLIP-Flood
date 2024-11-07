import os
import numpy as np
import matplotlib.pyplot as plt

golden_label_file = './data/golden_label.txt'
image_pre_score_file = './weights/train-15/image_is_no_score.txt'

golden_label_aa = {} # The real category of the images
with open(golden_label_file, 'r', encoding='utf-8') as f:
    for row in f.readlines():
        row = row.strip()
        image_name, image_label = row.split('\t')
        golden_label_aa[image_name] = image_label

image_is_scores = [] # Probability that the image is predicted to be flood_is
image_no_scores = [] # Probability that the image is predicted to be flood_no
image_pre_score_data = np.loadtxt(image_pre_score_file, dtype=str)
image_is_scores = image_pre_score_data[:,1].astype(float)
image_no_scores = image_pre_score_data[:,2].astype(float)
image_name = image_pre_score_data[:,0]
golden_label = {}
for item in image_name:
    golden_label[item] = golden_label_aa[item]
golden_label_sorted = sorted(golden_label.items())
golden_label.clear()
golden_label = {key:value for key, value in golden_label_sorted}

x = list(range(image_is_scores.shape[0]))
# Plotting histograms

plt.hist(image_is_scores, label='image_is_scores', bins=60, color='red', alpha=0.5)
plt.hist(image_no_scores, label='image_no_scores', bins=60, color='orange', alpha=0.5)

plt.legend()
plt.show()

# Different category determination strategies
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
    
    if is_is + no_is == 0:
        precision_is_max = 0.0
    else:
        precision_is_max = round(is_is / (is_is + no_is), 4)
    if is_no + no_no == 0:
        precision_no_max = 0.0
    else:
        precision_no_max = round(no_no / (is_no + no_no), 4)

    return recall_is_max, precision_is_max, recall_no_max, precision_no_max

# Strategy 1: HC
def score_max():
    recall_is_max = 0.0
    precision_is_max = 0.0
    F1_is = 0.0
    F1_no = 0.0
    recall_no_max = 0.0
    precision_no_max = 0.0
    pre_label = {}
    for item in image_pre_score_data:
        image_name = item[0]
        if float(item[1]) > float(item[2]):
            pre_label[image_name] = 'flood_is'
        else:
            pre_label[image_name] = 'flood_no'
    pre_label_sorted = sorted(pre_label.items())
    pre_label.clear()
    pre_label = {key:value for key, value in pre_label_sorted}

    recall_is_max, precision_is_max, recall_no_max, precision_no_max = recall_precision(list(golden_label.values()), list(pre_label.values()))
    F1_is = round((2 * recall_is_max * precision_is_max) / (recall_is_max + precision_is_max), 4)
    F1_no = round((2 * recall_no_max * precision_no_max) / (recall_no_max + precision_no_max), 4)
    print(f"HC:\nF1_is:{F1_is}\nrecall_is:\t{recall_is_max}\nprecision_is:\t{precision_is_max}\nF2_no:{F1_no}\nrecall_no:\t{recall_no_max}\nprecision_no:\t{precision_no_max}")
    # print(f"")
# Strategy 2：Soft Categorization Strategy
def score_threshold():
    recall_is_threshold = []
    precision_is_threshold = []
    F1_is = []
    # recall_iss = []
    
    recall_no_threshold = []
    precision_no_threshold = []
    F1_no = []
    # recall_nos = []

    thresholds = np.arange(0,1,0.01).tolist() 
    for threshold in thresholds:
        pre_label = {}
        for item in image_pre_score_data:
            image_name = item[0]
            if float(item[1]) - float(item[2]) >= threshold:
                pre_label[image_name] = 'flood_is'
            else:
                pre_label[image_name] = 'flood_no'
        pre_label_sorted = sorted(pre_label.items())
        pre_label.clear()
        pre_label = {key:value for key, value in pre_label_sorted}
        recall_is, precision_is, recall_no, precision_no = recall_precision(list(golden_label.values()), list(pre_label.values()))


        recall_is_threshold.append(recall_is)
        precision_is_threshold.append(precision_is)
        if recall_is + precision_is == 0.0:
            F1_is.append(0.0)
        else:
            F1_is.append(round((2 * recall_is * precision_is) / (recall_is + precision_is), 4))
        recall_no_threshold.append(recall_no)
        precision_no_threshold.append(precision_no)
        if recall_no + precision_no == 0.0:
            F1_no.append(0.0)
        else:
            F1_no.append(round((2 * recall_no * precision_no) / (recall_no + precision_no), 4))

    F1_is_max = max(F1_is) # Maximum value of F1_is
    recall_is_max = max(recall_is_threshold) # Maximum value of recall_is

    F1_is_max_threhold = thresholds[F1_is.index(F1_is_max)] # Threshold corresponding to the maximum value of F1_is
    recall_is_max_threshold = thresholds[recall_is_threshold.index(recall_is_max)]  # Threshold corresponding to the maximum value of recall_is

    recall_is_now = recall_is_threshold[F1_is.index(F1_is_max)]
    recall_no_now = recall_no_threshold[F1_is.index(F1_is_max)]

    precision_is_now = precision_is_threshold[F1_is.index(F1_is_max)]
    precision_no_now = precision_no_threshold[F1_is.index(F1_is_max)]
    
    print("\nStrategy：Soft Categorization Strategy")
    print(f"F1_is_max:\t{F1_is_max}")
    print(f"F1_is_max_threhold:\t{F1_is_max_threhold}")
    print(f"recall_is_now:\t{recall_is_now}")
    print(f"precision_is_now:\t{precision_is_now}")
    print(f"recall_no_now:\t{recall_no_now}")
    print(f"precision_no_now:\t{precision_no_now}")
    print(f"recall_is_max:\t{recall_is_max}")
    print(f"recall_is_max_threshold:\t{recall_is_max_threshold}")
    
    plt.plot(thresholds, F1_is, label="F1_is", color='red', linestyle='-')
    plt.plot(thresholds, F1_no, label="F1_no", color='red', linestyle='--')

    plt.plot(thresholds, recall_is_threshold, label="recall_is_threshold", color='blue', linestyle='-')
    plt.plot(thresholds, recall_no_threshold, label="recall_no_threshold", color='blue', linestyle='--')
    
    plt.plot(thresholds, precision_is_threshold, label="precision_is_threshold", color='green', linestyle='-')
    plt.plot(thresholds, precision_no_threshold, label="precision_no_threshold", color='green', linestyle='--')

    plt.axvline(x=thresholds[F1_is.index(max(F1_is))], color='grey', linestyle='--', label=f"F1_is_max {F1_is_max}-{F1_is_max_threhold}")
    plt.axvline(x=thresholds[recall_is_threshold.index(max(recall_is_threshold))], color='orange', linestyle='--', label=f"recall_is_max {recall_is_max}-{recall_is_max_threshold}")

    plt.legend()
    plt.show()
score_max()
score_threshold()

