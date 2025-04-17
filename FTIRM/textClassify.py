import os
import pandas as pd
import numpy as np
import random
import json
from pprint import pprint

from tqdm.autonotebook import tqdm
import torch
from model import SLIP_Flood
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import time
import transformers
from transformers import AutoFeatureExtractor, AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FTIRM:
    file = './data/example_predict.tsv'   
    golden_label_file = './data/example_predict_label.tsv'  
    text_label_file = './data/example_predict_text_label-ori.tsv'
    image_is_no_score_file = './data/image_is_no_score.txt' 
    # image_is_no_score_end_file = './data/image_is_no_score_end_r_vl_evalLoss.txt' 
    img_key = 'filepath'
    caption_key = 'title'
    max_text_len = 100
    text_ptm = "./models/chinese-roberta-wwm-ext-large/" # chinese-roberta-wwm-ext chinese-roberta-wwm-ext-large
    img_ptm = "./models/vit-large-patch16-224/" #  vit-base-patch16-224 vit-large-patch16-224
    save_path = "./checkpoints/pretrain-roberta-large_cn-vit-large-saved-title/best_checkpoint_evalLoss.pt" # The path of saved model.
    pretrained = True                 
    freeze = False
    dim = 2048
    device = 'cuda:0'
    batch_size = 256
    apex = True     

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.texts = texts
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        text_tensor = self.tokenizer(text, 
                                max_length=FTIRM.max_text_len, 
                                truncation=True, 
                                return_tensors='pt', 
                                padding="max_length",)
        for k,v in text_tensor.items():
            text_tensor[k] = v.squeeze()
        return text_tensor
        
class ImgDataset(Dataset):
    def __init__(self, img_paths, feature_extractor):
        self.feature_extractor = feature_extractor
        self.img_paths = img_paths
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        img_tensor = self.feature_extractor(Image.open(img_path).convert("RGB"), return_tensors="pt")
        for k,v in img_tensor.items():
            img_tensor[k] = v.squeeze()
        return img_tensor

class Engine:
    def __init__(self):
        self.device = torch.device(FTIRM.device)
        self.clipModel = SLIP_Flood(FTIRM.dim, FTIRM.text_ptm, FTIRM.img_ptm, self.device, pretrained=FTIRM.pretrained,freeze=FTIRM.freeze)
        self.clipModel.load_state_dict(torch.load(FTIRM.save_path))
        self.clipModel.eval()
        self.clipModel = self.clipModel.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(FTIRM.text_ptm)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(FTIRM.img_ptm)
    def encode_text(self, text_or_texts):
        if type(text_or_texts) == str:
            text = text_or_texts
            inputs = self.tokenizer(text, 
                                max_length=FTIRM.max_text_len, 
                                truncation=True, 
                                return_tensors='pt', 
                                padding="max_length",)
            for k,v in inputs.items():
                inputs[k] = v.to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=FTIRM.apex):
                    feat = self.clipModel.textencoder(inputs)
                feat = feat @ self.clipModel.text_projection
        else:
            dataset = TextDataset(text_or_texts, self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=FTIRM.batch_size, shuffle=False)
            feat = []
            for batch in tqdm(dataloader, total=len(dataloader), desc="Text encode"):
                for k,v in batch.items():
                    batch[k] = v.to(self.device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=FTIRM.apex):
                        batch_feat = self.clipModel.textencoder(batch)
                    batch_feat = batch_feat @ self.clipModel.text_projection
                feat.append(batch_feat)
            feat = torch.cat(feat)
        return feat.squeeze().cpu()
    def encode_img(self, img_or_imgs):
        if type(img_or_imgs) == str:
            img_path = img_or_imgs
            img_tensor = self.feature_extractor(Image.open(img_path).convert("RGB"), return_tensors="pt")
            for k,v in img_tensor.items():
                img_tensor[k] = v.to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=FTIRM.apex):
                    feat = self.clipModel.imgencoder(img_tensor)
                feat = feat @ self.clipModel.img_projection
        else:
            dataset = ImgDataset(img_or_imgs, self.feature_extractor)
            dataloader = DataLoader(dataset, batch_size=FTIRM.batch_size, shuffle=False, num_workers=5)
            feat = []
            for batch in tqdm(dataloader, total=len(dataloader), desc="Img encode"):
                for k,v in batch.items():
                    batch[k] = v.to(self.device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=FTIRM.apex):
                        batch_feat = self.clipModel.imgencoder(batch)
                    batch_feat = batch_feat @ self.clipModel.img_projection
                feat.append(batch_feat)
            feat = torch.cat(feat)
        return feat.squeeze().cpu()
engine = Engine()      

df = pd.read_csv(FTIRM.file, sep="\t")
texts = df[FTIRM.caption_key].values
img_paths = df[FTIRM.img_key].values

"""
# Get text and image embeddings through batch processing.
# This is a optional way.
text_feats = engine.encode_text(texts)
img_feats = engine.encode_img(img_paths)
torch.cuda.empty_cache()
text_feats.shape, img_feats.shape
"""

predict_text_image_file = {}
with open(FTIRM.file, 'r', encoding='utf-8') as f:
    for row in f.readlines()[1:]:
        row = row.strip()
        image_file, text = row.split('\t')
        predict_text_image_file[image_file] = text

image_is_no_score = {} # The probability value of each image predicted by Swin Transformer.
with open(FTIRM.image_is_no_score_file, 'r', encoding='utf-8') as f:
    for row in f.readlines():
        row = row.strip()
        image_name, is_score, no_score = row.split('\t')
        image_is_no_score[image_name] = [float(is_score), float(no_score)]

# The category probability value of the text is calculated based on the 
# image-text matching degree and the category probability value of the image.
# Used to assist text classification tasks.
text_is_no_score = {}   
count_predict = 0
for image_file, text in predict_text_image_file.items():
    text_encode = engine.encode_text(text) # Embedding of text.
    image_encode = engine.encode_img(image_file) # Embedding of image.
    image_text_suitability = torch.sigmoid(image_encode @ text_encode.t()) # Match Degree between image with text.
    image_label_p = image_is_no_score[image_name] # Image prediction probability
    text_score = torch.tensor(image_label_p) * image_text_suitability 
    text_is_no_score[image_file] = text_score.tolist()
    count_predict += 1
    print(f"{count_predict}/{10000}")



# The following code can be used to calculate the final category of the text. 
# However, the effect is not good. 
# It is recommended to rewrite it in combination with a dedicated model for text classification.
"""
# Get the true label of the text.
text_golden_label = {}
with open(FTIRM.text_label_file, 'r', encoding='utf-8') as f:
    for row in f.readlines():
        image_file, title, label = row.strip().split('\t')
        text_golden_label[image_file] = int(label)

F1_iss = []
F1_nos = []
recall_iss = []
recall_nos = []
precision_iss = []
precision_nos = []
is_count = sum(list(text_golden_label.values()))
no_count = len(text_golden_label) - is_count
thresholds_diff = np.arange(0,1,0.001).tolist()

count = 0
for threshold in thresholds_diff:
    is_is = 0
    is_no = 0
    no_no = 0
    no_is = 0
    text_pre_label = {}
    for image_file, score in text_is_no_score.items():
        if score[0] >= threshold:
            text_pre_label[image_file] = 1
        else:
            text_pre_label[image_file] = 0

    for image_file, label in text_pre_label.items():
        golden_label = text_golden_label[image_file]
        pre_label = label
        if golden_label == 1 and pre_label == 1:
            is_is += 1
        if golden_label == 1 and pre_label == 0:
            is_no += 1
        if golden_label == 0 and pre_label == 0:
            no_no += 1
        if golden_label == 0 and pre_label == 1:
            no_is += 1
    recall_is = (is_is)/(is_count)
    precision_is = is_is / max((is_is + no_is), 0.0001)
    F1_is = (2*recall_is*precision_is)/max((recall_is+precision_is), 0.0001)
    recall_no = no_no / no_count
    precision_no = no_no / max((is_no + no_no), 0.0001)
    F1_no = (2*recall_no*precision_no)/max((recall_no+precision_no), 0.0001)

    recall_iss.append(recall_is)
    precision_iss.append(precision_is)
    F1_iss.append(F1_is)
    recall_nos.append(recall_no)
    precision_nos.append(precision_no)
    F1_nos.append(F1_no)
    count += 1
    print(f"{count}/{len(thresholds_diff)}")
F1_is_max = 0.0
F1_is_max_threshold = 0.0
recall_is_now = 0.0
precision_is_now = 0.0
recall_no_now = 0.0
precision_no_now = 0.0

recall_is_max = 0.0
recall_is_max_threshold = 0.0

precision_is_max = 0.0
precision_is_max_threshold = 0.0

recall_no_max = 0.0
recall_no_max_threshold = 0.0

precision_no_max = 0.0
precision_no_max_threshold = 0.0

for F1_is, recall_is, precision_is, F1_no, recall_no, precision_no, threshold in zip(F1_iss, recall_iss, precision_iss,
                                                                          F1_nos, recall_nos, precision_nos,
                                                                          thresholds_diff):
    if F1_is > F1_is_max:
        F1_is_max = F1_is
        recall_is_now = recall_is
        precision_is_now = precision_is
        F1_is_max_threshold = threshold
        recall_no_now = recall_no
        precision_no_now = precision_no
    if recall_is > recall_is_max:
        recall_is_max = recall_is
        recall_is_max_threshold = threshold 
    if precision_is > precision_is_max:
        precision_is_max = precision_is
        precision_is_max_threshold = threshold
    if recall_no > recall_no_max:
        recall_no_max = recall_no
        recall_no_max_threshold = threshold
    if precision_no > precision_no_max:
        precision_no_max = precision_no
        precision_no_max_threshold = threshold

F1_is_max = round(F1_is_max, 4)
F1_is_max_threshold = round(F1_is_max_threshold, 4)
recall_is_now = round(recall_is_now, 4)
precision_is_now = round(precision_is_now, 4)
recall_no_now = round(recall_no_now, 4)
precision_no_now = round(precision_no_now, 4)

recall_is_max = round(recall_is_max, 4)
recall_is_max_threshold = round(recall_is_max_threshold, 4)

precision_is_max = round(precision_is_max, 4)
precision_is_max_threshold = round(precision_is_max_threshold, 4)

recall_no_max = round(recall_no_max, 4)
recall_no_max_threshold = round(recall_no_max_threshold, 4)

precision_no_max = round(precision_no_max, 4)
precision_no_max_threshold = round(precision_no_max_threshold, 4)

print(f"F1_is_max:{F1_is_max}\nF1_is_max_threshold:{F1_is_max_threshold}\nrecall_is_now:{recall_is_now}\nprecision_is_now:{precision_is_now}\nrecall_no_now:{recall_no_now}\nprecision_no_now:{precision_no_now}\n"\
      f"recall_is_max:{recall_is_max}\nrecall_is_max_threshold:{recall_is_max_threshold}\n"\
      f"precision_is_max:{precision_is_max}\nprecision_is_max_threshold:{precision_is_max_threshold}"\
      f"recall_no_max:{recall_no_max}\nrecall_no_max_threshold:{recall_no_max_threshold}\n"\
      f"precision_no_max:{precision_no_max}\nprecision_no_max_threshold:{precision_no_max_threshold}"  )
    
"""