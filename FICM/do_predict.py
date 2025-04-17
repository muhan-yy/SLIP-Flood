import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import shutil
from PIL import Image

#-----Input the Model Structure-----#
# from model import swin_tiny_patch4_window7_224 as create_model
from model import swin_base_patch4_window7_224_in22k as create_model
# from model import swin_large_patch4_window7_224_in22k as create_model

images_get_file = 'images_get.txt'
images_get = {}
with open(images_get_file, 'r') as f:
    for row in f.readlines():
        image_name = row.strip()
        images_get[image_name] = None

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    ori_images_file = "./data/predict/images_all/"
    flood_is_file = './data/predict/flood_is/'
    flood_no_file = './data/predict/flood_no/'

    img_paths = [ori_images_file + image_name for image_name in os.listdir(ori_images_file)]
    img_complete = [flood_is_file + image_name for image_name in os.listdir(flood_is_file)] +\
                   [flood_no_file + image_name for image_name in os.listdir(flood_no_file)]
    img_complete = {item:None for item in [ori_images_file + image_name for image_name in os.listdir(flood_is_file)] +\
                                [ori_images_file + image_name for image_name in os.listdir(flood_no_file)]}
    n = 0
    for img_path in img_paths:
        if img_path in img_complete:
            continue
        if img_path in images_get:
            continue
        # Load image
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        try:
            img = Image.open(img_path)
        except:
            continue

        try:
            img = data_transform(img)
        except:
            os.remove(img_path)
            continue

        # Expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # Read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)

        # Create model
        model = create_model(drop_rate=0.2, attn_drop_rate=0.1, num_classes=2).to(device)
        # Load model weights
        model_weight_path = "./weights/train-14/model-9.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # Predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()


        img = Image.open(img_path)
        if img.mode == 'RGB':
            if predict[0].numpy() - predict[1].numpy() >= 0.00: # The values here are thresholds calculated using the Soft Categorization Strategy
                # the image is flooding
                image_predict = flood_is_file + img_path.split('/')[-1]
            else: 
                # the image is not flooding
                image_predict = flood_no_file + img_path.split('/')[-1]
            
            try:
                shutil.copy(img_path, image_predict)
            except Exception as e:
                print(e)
            n += 1
            print(f"{n}/{len(img_paths)}")

            # 保存预测的置信度
            image_name = img_path.split('/')[-1].split('.')[0]
            isflood_score = predict[0].numpy()
            noflood_score = predict[1].numpy()
            with open('./weights/image_score.txt', 'a') as f:
                f.write(image_name + '\t' + str(isflood_score) + '\t' + str(noflood_score) + '\n')


if __name__ == '__main__':
    main()
