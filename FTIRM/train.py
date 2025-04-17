"""
Thanks to project https://github.com/ZYiJie/Simple-CLIP.
"""
import os
import pandas as pd
import numpy as np
import random
import sys
import json
import math
from pprint import pprint

from tqdm import tqdm
import torch
from model import SLIP_Flood
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import transformers
from transformers import AutoFeatureExtractor, AutoTokenizer
print(f"transformers.__version__: {transformers.__version__}")
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
transformers.logging.set_verbosity_error()


# ### 参数

class FTIRM:
    train_file = './data/example_train.tsv'    
    valid_file = './data/example_test.tsv'   
    proportion = 0.68                     # Proportion of training, validation data, not enough CPU if all used  roberta+vit/vit-large:<=0.7 
    img_key = 'filepath'
    caption_key = 'title'
    max_text_len = 34
    stop_step = 15                       # Early stop indicator
    
    text_ptm = 'roberta_cn'                 # Chinese text encoder
    # text_ptm = 'roberta-large_cn'           # Chinese text encoder

    img_ptm = 'vit-large'                # Image encoder


    output_dir = f'checkpoints/pretrain-{text_ptm}-{img_ptm}-saved'
    pretrained = True                    # Whether to load pre-trained model weights, False means only load model structure random initialization weights.
    freeze = False                       # Whether to freeze text encoder
    load_model = None                    # Whether to load model paths

    # Training parameter
    dim = 2048
    device = 'cuda:0'
    epochs = 300
    learning_rate = 1e-7                 # learning rate

    # batch_size = 4                     # batch size    Set 4 for roberta-large_cn + vit-large, GPU usage 11.8GB
    batch_size = 15                      # batch size    Set 15 for roberta_cn + vit-large, GPU usage 11.7GB
    accumulation_steps = 8               # gradient accumulation
    eval_epoch = 1                       # Evaluate the model every few epochs
    apex = True                          # Whether to use mixed precision acceleration
    
    seed = 115 
    # Scheduler parameters
    scheduler = 'cosine'                 # ['linear', 'cosine'] # lr scheduler type
    last_epoch = -1                      # Start training from last_epoch + 1 epoch
    batch_scheduler = True               # Whether to update lr scheduler after each "step"
    weight_decay = 0.01
    num_warmup_steps = 0
    num_cycles = 0.5                     # If scheduler = 'cosine', this parameter determines the shape of the learning rate curve, 0.5 represents half of the cosine curve
    

    key_metrics_i2t = 'image_to_text_R@10'
    key_metrics_t2i = 'text_to_image_R@10'

ptm = {
    "roberta_cn": './models/chinese-roberta-wwm-ext/',
    "roberta-large_cn": './models/chinese-roberta-wwm-ext-large/',

    "vit-large": './models/vit-large-patch16-224/',
}


#-----Set the Global Random Seed-----#
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Modify the learning rate
def cosine_annealing_learning_rate(current_step, total_steps, initial_lr, eta_min):
    cos = math.cos(math.pi * current_step / total_steps)
    lr = eta_min + (initial_lr - eta_min) * (1 + cos) / 2
    return lr

class TrainDataset(Dataset):
    def __init__(self, input_file):
        self.tokenizer = AutoTokenizer.from_pretrained(ptm[FTIRM.text_ptm])
        data_df = pd.read_csv(input_file, sep='\t')
        data_df = data_df.loc[:data_df.shape[0] * FTIRM.proportion,:]
        self.img_paths = data_df[FTIRM.img_key].values
        self.texts = data_df[FTIRM.caption_key].values

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(ptm[FTIRM.img_ptm])
        
        print(f'load data from {input_file} len={len(self.texts)}')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        img_path = self.img_paths[item]
        text_tensor = self.tokenizer(text, 
                                max_length=FTIRM.max_text_len, 
                                truncation=True, 
                                return_tensors='pt', 
                                padding="max_length",)
        img_tensor = self.feature_extractor(Image.open(img_path).convert("RGB"), return_tensors="pt")
        for k,v in text_tensor.items():
            text_tensor[k] = v.squeeze()
        for k,v in img_tensor.items():
            img_tensor[k] = v.squeeze()

        return {'text':text_tensor, 'img':img_tensor}

def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
    return metrics

def evaluate(model, valid_dataloader, device):
    model.eval()
    all_text_feat = []
    all_img_feat = []
    tk0 = tqdm(enumerate(valid_dataloader),total=len(valid_dataloader), desc="[Dev]")
    total_loss = 0
    for step, batch in tk0:
        for k,v in batch['img'].items():
            batch['img'][k] = v.to(device)
        for k,v in batch['text'].items():
            batch['text'][k] = v.to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=FTIRM.apex):
                loss, text_feat, img_feat, logit_scale = model(batch['text'], 
                                                                batch['img'], 
                                                                outputLoss=True)
        total_loss += loss.item()
        all_text_feat.append(text_feat)
        all_img_feat.append(img_feat)
        
    metrics = get_metrics(image_features=torch.cat(all_img_feat),
                          text_features=torch.cat(all_text_feat),
                          logit_scale=logit_scale)
    metrics['eval_loss'] = total_loss / len(valid_dataloader)
    return metrics

def train_eval(model, train_dataloader, valid_dataloader, save_path):
    assert FTIRM.device.startswith('cuda') or FTIRM.device == 'cpu', ValueError("Invalid device.")
    device = torch.device(FTIRM.device)
    # Indicators for determining the optimal model
    best_score = 0 
    train_loss_base = sys.maxsize
    eval_loss_base = sys.maxsize
    stop_step = FTIRM.stop_step

    total_step = 0
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=FTIRM.apex)
    if not len(train_dataloader):
        raise EOFError("Empty train_dataloader.")

    # Filter out frozen weights
    param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # Set the decay of weights
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": FTIRM.weight_decay},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=FTIRM.learning_rate, weight_decay=FTIRM.weight_decay)
    
    num_train_steps = int(len(train_dataloader) * FTIRM.epochs / FTIRM.accumulation_steps)
    if FTIRM.scheduler=='cosine':
        scheduler = get_cosine_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=FTIRM.num_warmup_steps, 
                    num_training_steps=num_train_steps, 
                    num_cycles=FTIRM.num_cycles, 
                )
    else:
        scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=FTIRM.num_warmup_steps, num_training_steps=num_train_steps
            )
    best_epoch = 0
    initial_lr = FTIRM.learning_rate
    eta_min = 0.00001
    for cur_epc in range(int(FTIRM.epochs)):

        if stop_step <= 0:
            break
        
        training_loss = 0
        train_loss_save = 0
        model.train()
        tk0 = tqdm(enumerate(train_dataloader),total=len(train_dataloader), desc="Epoch: {}".format(cur_epc))
        for step, batch in tk0:
            total_step += 1
            for k,v in batch['img'].items():
                batch['img'][k] = v.to(device)
            for k,v in batch['text'].items():
                batch['text'][k] = v.to(device)
            with torch.cuda.amp.autocast(enabled=FTIRM.apex):
                loss, _, _, _ = model(batch['text'], batch['img'], outputLoss=True)
            scaler.scale(loss).backward()
            if (step+1) % FTIRM.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if FTIRM.batch_scheduler:
                    scheduler.step()
            training_loss += loss.item()
            train_loss_save = training_loss/(step+1)
            tk0.set_postfix(Epoch=cur_epc, Loss=training_loss/(step+1))
      
        if cur_epc % FTIRM.eval_epoch == 0:
            metrics = evaluate(model, valid_dataloader, device)
            print(f"eval metrics = ")
            pprint(metrics)

            # 判断是否早停
            eval_loss = metrics['eval_loss']
            if eval_loss >= eval_loss_base:
                stop_step -= 1

            epoch_score = (metrics[FTIRM.key_metrics_i2t] + metrics[FTIRM.key_metrics_t2i]) / 2
            if epoch_score >= best_score:
                best_score = epoch_score
                # model_save_path = os.path.join(save_path,f'epoch{cur_epc}.pt') # Keep all checkpoints
                model_save_path = os.path.join(save_path,f'best_checkpoint_score.pt') # Keep the best checkpoint based on the epoch_score.
                torch.save(model.state_dict(), model_save_path)
                best_epoch_score = cur_epc
            if train_loss_save < train_loss_base:
                train_loss_base = train_loss_save
                model_save_path = os.path.join(save_path,f'best_checkpoint_trainLoss.pt') # Keep the best checkpoint based on the train_loss.
                torch.save(model.state_dict(), model_save_path)
                best_epoch_trainLoss = cur_epc
            
            if eval_loss < eval_loss_base:
                eval_loss_base = eval_loss
                model_save_path = os.path.join(save_path,f'best_checkpoint_evalLoss.pt') # Keep the best checkpoint based on the eval_loss.
                torch.save(model.state_dict(), model_save_path)
                best_epoch_evalLoss = cur_epc

            # Save log
            with open(f'checkpoints/pretrain-{FTIRM.text_ptm}-{FTIRM.img_ptm}-saved/log.txt', 'a') as f:
                log_data = {'train_loss': train_loss_save, 'lr':optimizer.param_groups[0]["lr"], 'epoch':cur_epc,
                            'key_metrics_i2t': metrics[FTIRM.key_metrics_i2t], 'key_metrics_t2i': metrics[FTIRM.key_metrics_t2i],
                            'eval_loss': metrics['eval_loss'], 
                            'best_epoch_score':best_epoch_score, 'best_epoch_trainLoss':best_epoch_trainLoss, 'best_epoch_evalLoss':best_epoch_evalLoss}
                log_data = json.dumps(log_data)
                f.write(log_data + '\n')
    torch.cuda.empty_cache()          

if __name__ == '__main__':
    seed_everything(FTIRM.seed)
    if not os.path.exists(FTIRM.output_dir):
        os.makedirs(FTIRM.output_dir)
    with open(os.path.join(FTIRM.output_dir, 'config.txt'), 'w') as f:
        for k,v in FTIRM.__dict__.items():
            f.write(f'{k}: {v}\n')

    # Loading data
    train_dataset = TrainDataset(FTIRM.train_file)
    valid_dataset = TrainDataset(FTIRM.valid_file)
    train_dataloader = DataLoader(train_dataset, batch_size=FTIRM.batch_size, num_workers=5)
    valid_dataloader = DataLoader(valid_dataset, batch_size=FTIRM.batch_size, num_workers=5)
    # Loading model
    device = torch.device(FTIRM.device)
    clipModel = SLIP_Flood(FTIRM.dim, ptm[FTIRM.text_ptm], ptm[FTIRM.img_ptm], device, pretrained=FTIRM.pretrained,freeze=FTIRM.freeze)
    if FTIRM.load_model is not None:
        clipModel.load_state_dict(torch.load(FTIRM.load_model))
        print(f"load state from {FTIRM.load_model}")
    
    # train
    train_eval(clipModel, train_dataloader, valid_dataloader, FTIRM.output_dir)


