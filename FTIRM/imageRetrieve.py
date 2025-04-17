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
    # image_is_no_score_file = './data/image_is_no_score.txt' 
    # image_is_no_score_end_file = './data/image_is_no_score_end_rl_vl_evalLoss.txt' 
    img_key = 'filepath'
    caption_key = 'title'
    max_text_len = 100
    text_ptm = "./models/chinese-roberta-wwm-ext/" # chinese-roberta-wwm-ext chinese-roberta-wwm-ext-large
    img_ptm = "./models/vit-large-patch16-224/" #  vit-base-patch16-224 vit-large-patch16-224
    save_path = "./checkpoints/pretrain-roberta_cn-vit-large-saved-label/best_checkpoint_evalLoss.pt" # The path of saved model.
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

# Non-batch processing
text_feats = {}
img_feats = {}
for text, img_path in tqdm(zip(texts, img_paths), total=len(texts), desc="Batch encode"):
    text_feat = engine.encode_text(text)
    img_feat = engine.encode_img(img_path)

    text_feats[text] = text_feat
    img_feats[img_path] = img_feat

torch.cuda.empty_cache()
print(len(text_feats), len(img_feats))
query = "与洪灾不相关" # Text for query
image_label = {
    "与洪灾相关":"flood_is",
    "与洪灾不相关":"flood_no",
}
image_count = len(img_feats) # Count of images
# top 5000
topk = 5000 

img_feats_embedding = torch.stack(list(img_feats.values()))
normalized_data = torch.sigmoid(img_feats_embedding @ engine.encode_text(query).t()) 
probabilities = torch.softmax(normalized_data, dim=0)
values, indices = torch.topk(probabilities, topk)

image_pre_info = {}
pre_loss = 0.0
image_golden_label = {}

is2is_count = 0 # Image of flood_is predicted as flood_is.
no2is_count = 0 # Image of flood_no predicted as flood_is.
golden_is_count = 0
golden_no_count = 0
error_score = []

with open(FTIRM.golden_label_file, 'r', encoding='utf-8') as f:
    for row in f.readlines():
        row = row.strip()
        image_name, golden_label = row.split('\t')
        image_golden_label[image_name] = golden_label
        if golden_label == 'flood_is':
            golden_is_count += 1
        if golden_label == 'flood_no':
            golden_no_count += 1

scores = []

for i in range(len(values)):
    image_name = img_paths[indices[i]]
    text_info = texts[indices[i]]
    score = values[i]
    pre_label = image_label[query]
    try:
        golden_label = image_golden_label[image_name]
    except:
        golden_label = 'flood_no'

    if golden_label != pre_label:
        pre_loss += score.item()
        no2is_count += 1
        error_score.append(score.item())
    else:
        is2is_count += 1
    image_pre_info[image_name] = {"label": pre_label, 
                                  "text_info": text_info,
                                  "score": score.item()}
    scores.append(score.item())

pre_loss = round(pre_loss / image_count, 6)
recall = round(is2is_count/golden_is_count, 4)
print(f"Recall of {image_label[query]}：{recall}")