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
    file = './data/example_predict.tsv'   # Dataset for predict.
    golden_label_file = './data/example_predict.tsv'  # Golden labels of predict dataset.
    image_is_no_score_file = './data/image_is_no_score.txt' 
    image_is_no_score_end_file = './data/image_is_no_score_end_r_vl_evalLoss-temp.txt' 
    img_key = 'filepath'
    caption_key = 'title'
    max_text_len = 100
    text_ptm = "./models/chinese-roberta-wwm-ext/" # chinese-roberta-wwm-ext chinese-roberta-wwm-ext-large
    img_ptm = "./models/vit-large-patch16-224/" #  vit-base-patch16-224 vit-large-patch16-224
    save_path = "./checkpoints/pretrain-roberta_cn-vit-large-saved-title/best_checkpoint_evalLoss.pt"  # The path of saved model. We got the model based on the eval_loss.
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

predict_image_file_text = {}
with open(FTIRM.file, 'r', encoding='utf-8') as f:
    for row in f.readlines()[1:]:
        row = row.strip()
        image_file, text = row.split('\t')
        predict_image_file_text[image_file] = text

text_labels_cn = ["与洪灾相关", "与洪灾不相关"] # Possible Chinese categories of text

# Stores the probability values ​​of image categories predicted using FICM
image_is_no_score = {}
with open(FTIRM.image_is_no_score_file, 'r', encoding='utf-8') as f:
    for row in f.readlines():
        row = row.strip()
        image_name, is_score, no_score = row.split('\t')
        image_is_no_score[image_name] = [float(is_score), float(no_score)]
image_pre_score_data = np.loadtxt(FTIRM.image_is_no_score_file, dtype=str)
image_is_scores = image_pre_score_data[:,1].astype(float)
image_no_scores = image_pre_score_data[:,2].astype(float)
image_diff_scores = np.abs(image_is_scores - image_no_scores)
image_diff_scores_mean = np.mean(image_diff_scores) 

count = 0
start = time.time()
for image, text in predict_image_file_text.items():
    image_name = image.split('/')[-1]
    text_encode = engine.encode_text(text) # Embedding of text. shape: torch.Size([1, 2048])
    text_labels = ['flood_is', 'flood_no']

    image_files = [image]
    image_encode = engine.encode_img(image) # Embedding of image.  shape: torch.Size([1, 2048])

    image_text_suitability = torch.sigmoid(image_encode @ text_encode.t()) # Calculate the Match Degree bttween image with text.


    text_labels_encode = engine.encode_text(text_labels_cn) # Embeddings of Chinese text labels. shape: torch.Size([2, 2048])
    text_label_p_bert = torch.sigmoid(text_encode @ text_labels_encode.t())  # The probability that a text belongs to each category. shape：torch.Size([1, 2] )
    # Calculate the probability of the image belonging to each category based on the probability 
    # value of the category to which the text belongs and the image-text matching degree。
    image_label_p_text = torch.sigmoid(
                        image_text_suitability.reshape((len(image_files),-1)) @ text_label_p_bert.reshape(1,-1))
    image_label_p_SwinT = torch.tensor([image_is_no_score[image_name]] )  # The probability that the image belongs to each category based on FICM
    
    # Use LDWS to calculate the final image category probability.
    diff_p_SwinT = torch.abs(image_label_p_SwinT[:,0] - image_label_p_SwinT[:,1])
    thresholds = np.arange(0,1.001,0.001)
    text_label_ps = []
    for threshold in thresholds:
        if diff_p_SwinT >= threshold:
            a = 0
        else:
            a = 1-diff_p_SwinT
        b = 1-a
        text_label_p = a * image_label_p_text + b * image_label_p_SwinT
        text_label_ps.append(str(text_label_p.tolist()[0]))
    # Save the final image category probability based on different threshold.
    with open(FTIRM.image_is_no_score_end_file, 'a', encoding='utf-8') as f:
        f.write(image_name + '\t' + '\t'.join(text_label_ps) + '\n')
    count += 1
    end = time.time()
    print(f"{count}/{len(predict_image_file_text)}\tfps:{1/((end-start)/count)}")
