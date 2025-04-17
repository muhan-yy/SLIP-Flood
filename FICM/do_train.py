import os
import sys
import argparse
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from my_dataset import MyDataSet

#-----Input the Model Structure-----#
from model import swin_tiny_patch4_window7_224 as create_model
# from model import swin_large_patch4_window7_224_in22k as create_model
# from model import swin_base_patch4_window7_224_in22k as create_model
import random
import numpy as np
from utils import read_split_data, train_one_epoch, evaluate
import matplotlib.pyplot as plt
import math

#-----Set the Global Random Seed-----#
def seed_everything(seed=115):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def cosine_annealing_learning_rate(current_step, total_steps, initial_lr, eta_min):
    cos = math.cos(math.pi * current_step / total_steps)
    lr = eta_min + (initial_lr - eta_min) * (1 + cos) / 2
    return lr

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    print("Data Successfully Loaded")
    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # Instanting the training dataset
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # Instanting the test dataset
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate, num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # Deletion of weights related to categorization functions
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # All weights frozen except head
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    
    train_losses = [sys.maxsize]
    val_losses = [sys.maxsize]
    early_stop = 5
    is_stop = 0
    acc_base = 0.0
    train_loss_base = sys.maxsize
    val_loss_base = sys.maxsize
    acc_all = [sys.maxsize]
    train_acc_all = []
    val_acc_all = []
    initial_lr = args.lr
    eta_min = initial_lr * 0.01
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    # 定义余弦退火调度器
    # scheduler = CosineAnnealingLR(optimizer, T_max=1, eta_min=eta_min)

    # 各类模型评测指标
    train_loss_min = sys.maxsize
    train_loss_min_model = "./weights/model-train-loss-min.pth"
    val_loss_min = sys.maxsize
    val_loss_min_model = "./weights/model-val-loss-min.pth"
    aver_loss_min = sys.maxsize
    aver_loss_min_model = "./weights/model-aver-loss-min.pth"
    train_acc_max = 0.0
    train_acc_max_model = "./weights/model-train-acc-max.pth"
    val_acc_max = 0.0
    val_acc_max_model = "./weights/model-val-acc-max.pth"
    aver_acc_max = 0.0
    aver_acc_max_model = "./weights/model-aver-acc-max.pth"
    stop_step = 10

    loss_acc_epoch = {"train_loss_min":0,"val_loss_min":0,"aver_loss_min":0,
                    "train_acc_max":0,"val_acc_max":0,"aver_acc_max":0,}

    for epoch in range(args.epochs):
        # if stop_step <= 0:
        #     break

        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_acc_all.append(train_acc)
        val_acc_all.append(val_acc)

        

        if val_loss >= val_loss_min:
            stop_step -= 1
        if val_loss < val_loss_min:
            stop_step = 5

        # 分别保存对应指标的模型
        if train_loss < train_loss_min:
            # torch.save(model.state_dict(), train_loss_min_model)
            train_loss_min = train_loss
            loss_acc_epoch["train_loss_min"] = epoch
        if val_loss < val_loss_min:
            # torch.save(model.state_dict(), val_loss_min_model)
            val_loss_min = val_loss
            loss_acc_epoch["val_loss_min"] = epoch
        if (train_loss + val_loss)/2.0 < aver_loss_min:
            # torch.save(model.state_dict(), aver_loss_min_model)
            aver_loss_min = (train_loss + val_loss)/2.0
            loss_acc_epoch["aver_loss_min"] = epoch
        if train_acc > train_acc_max:
            # torch.save(model.state_dict(), train_acc_max_model)
            train_acc_max = train_acc
            loss_acc_epoch["train_acc_max"] = epoch
        if val_acc > val_acc_max:
            # torch.save(model.state_dict(), val_acc_max_model)
            val_acc_max = val_acc
            loss_acc_epoch["val_acc_max"] = epoch
        if (train_acc + val_acc)/2.0 > aver_acc_max:
            # torch.save(model.state_dict(), aver_acc_max_model)
            aver_acc_max = (train_acc + val_acc)/2.0
            loss_acc_epoch["aver_acc_max"] = epoch

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        with open('./weights/loss_acc.txt', 'a') as f:
            f.write(str(train_loss) + '\t' + str(train_acc) + '\t' + str(val_loss) + '\t' + str(val_acc) + '\n')
        with open('./weights/lr.txt', 'a') as f:
            f.write(str(optimizer.param_groups[0]["lr"]) + '\n')
        with open('./weights/loss_acc_epoch.txt', 'a') as f:
            loss_acc_epochStr = json.dumps(loss_acc_epoch)
            f.write(loss_acc_epochStr + '\n')
        
        """
        This project retains the model for each epoch and manually stops the training 
        according to the established early stopping strategy.
        """

        """ Optional early stop strategy"""
        # if train_losses[-1] > train_loss and (train_losses[-1] - train_loss) / train_losses[-1] > 0.001:
        #     temp = is_stop - 1
        #     is_stop = max(is_stop, temp)
        # if val_losses[-1] > val_loss and (val_losses[-1] - val_loss) / val_losses[-1] > 0.001:
        #     temp = is_stop - 1
        #     is_stop = max(is_stop, temp)
        
        # if train_loss >= train_losses[-1] or val_loss >= val_losses[-1]:
        #     is_stop += 1

        # if train_loss < train_losses[-1] or val_loss >= val_losses[-1]:
        #     is_stop += 2

        # if (train_losses[-1] > train_loss and (train_losses[-1] - train_loss) / train_losses[-1] > 0.001) and \
        #     (val_losses[-1] > val_loss and (val_losses[-1] - val_loss) / val_losses[-1] > 0.001):
        #     is_stop = 0
        #     train_loss_base = train_loss
        #     val_loss_base = val_loss
        # if train_loss > train_loss_base or val_loss > val_loss_base:
        #     is_stop += 1

        # if is_stop <= early_stop:
        #     train_losses.append(train_loss)
        #     val_losses.append(val_loss)
            
        #     acc_now = (train_acc + val_acc) / 2.0
        #     print(f'{acc_base} - {acc_now}')
        #     # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        #     # acc_all.append(acc_now)
        #     if acc_now >= acc_base:
        #         # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        #         torch.save(model.state_dict(), "./weights/model-best.pth")
        #         with open("./weights/large_model_best_epoch.txt", 'a') as f:
        #             f.write(str(epoch) + '\n')
        #         acc_base = acc_now
        # if is_stop > early_stop:
        #     break

        optimizer.param_groups[0]["lr"] = cosine_annealing_learning_rate(epoch, args.epochs, initial_lr, eta_min)
        print(optimizer.param_groups[0]["lr"])

    print(f"train_losses:\t{train_losses[1:]}\nval_losses:{val_losses[1:]}")
    plt.figure()
    x = list(range(len(train_losses[1:])))
    plt.plot(x, train_losses[1:], label='Train')
    plt.plot(x, val_losses[1:], label='Val')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    seed_everything(seed=115)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=198) 
    parser.add_argument('--lr', type=float, default=7e-6) 
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument('--attn_drop_rate', type=float, default=0.1)

    # The root directory where the dataset is located
    parser.add_argument('--data-path', type=str,
                        default="./data/flood_forTrain") 

    """
    Pre-trained weight paths, set to empty characters if you don't want to load them
    # ./models/swin_tiny_patch4_window7_224_22k.pth
    # ./models/swin_base_patch4_window7_224_22k.pth
    # ./models/swin_large_patch4_window7_224_22k.pth
    """
 
    parser.add_argument('--weights', type=str, default='./models/swin_tiny_patch4_window7_224_22k.pth',
                        help='initial weights path')
    # Whether to freeze weights
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
