import os
import json
import time

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import shutil
from PIL import Image

#-----Input the Model Structure-----#
# from model import swin_tiny_patch4_window7_224 as create_model
# from model import swin_base_patch4_window7_224_in22k as create_model
from model import swin_large_patch4_window7_224_in22k as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    ori_images_file = "./data/flood_forTest/images_all/"
    flood_is_file = './data/flood_forTest/flood_is/'
    flood_no_file = './data/flood_forTest/flood_no/'

    img_paths = [ori_images_file + image_name for image_name in os.listdir(ori_images_file)]
    n = 0
    count = 0
    start = time.time()
    for img_path in img_paths:

        # Load image
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        img = data_transform(img)
        # Expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        # Read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)

        # Create model
        model = create_model(drop_rate=0.2, attn_drop_rate=0.1, num_classes=2).to(device)
        # Load the trained model
        model_weight_path = "./weights/train-0/large/model-0.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # Predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        img = Image.open(img_path)
        if img.mode == 'RGB':
            # Image Category + Copy
            # if predict[0].numpy() - predict[1].numpy() >= 0.35: # the image is flooding
            # if predict[0].numpy() >= 0.75: # the image is flooding
            #     image_predict = flood_is_file + img_path.split('/')[-1]
            # else: # the image is not flooding
            #     image_predict = flood_no_file + img_path.split('/')[-1]
            
            # try:
            #     shutil.copy(img_path, image_predict)
            # except Exception as e:
            #     print(e)
            n += 1
            print(f"{n}/{len(img_paths)}")

            # Save predicted probability values
            image_name = img_path.split('/')[-1]
            isflood_score = predict[0].numpy()
            noflood_score = predict[1].numpy()
            with open('./weights/image_is_no_score.txt', 'a') as f:
                f.write(image_name + '\t' + str(isflood_score) + '\t' + str(noflood_score) + '\n')
        end = time.time()
        count += 1
        time_aver = (end - start) / count
        fps = 1 / time_aver
        print(f"fps:\t{fps}")

if __name__ == '__main__':
    main()
