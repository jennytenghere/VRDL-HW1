# openpyxl
import csv
import cv2 as cv
import gc
import json
# import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
from sklearn.metrics import confusion_matrix
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights
from torchvision.models import ViT_B_16_Weights, VGG16_Weights
import tqdm

# clear cache
torch.cuda.empty_cache()
gc.collect()

# random seed
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set gpu number
torch.cuda.set_device(0)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, val, test):
        super().__init__()
        self.root_dir = root_dir
        self.val = val
        self.test = test
        self.class_names = sorted(os.listdir(root_dir))
        # transform for traning data
        self.train_img_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.05, contrast=0.05, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # transforms for test data and validation data
        self.val_test_img_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.img_paths = []
        self.labels = []

        if not self.test:  # training and validation
            for class_name in self.class_names:
                class_dir = os.path.join(root_dir, class_name)
                # error detection
                if not os.path.exists(class_dir):
                    raise ValueError(f"Path {class_dir} does not exist!")
                for img_file in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_file)
                    # append each image's data path
                    self.img_paths.append(img_path)
                    # append label of each iamge
                    self.labels.append(int(class_name))
        else:  # test
            for img_file in os.listdir(root_dir):
                img_path = os.path.join(root_dir, img_file)
                # append each image's data path
                self.img_paths.append(img_path)
                # append file name of each image
                self.labels.append(img_file[0:-4])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # read image
        image = Image.open(self.img_paths[idx]).convert("RGB")
        # label or file name of the image
        label = self.labels[idx]
        if self.val or self.test:  # validation and traning
            image = self.val_test_img_transform(image)
        else:  # test
            image = self.train_img_transform(image)

        return image, label


def train(model, train_loader, optimizer, criterion, device, epochs, e):
    start_time = time.time()
    for epoch in range(1, epochs + 1):  # epoch start from 1
        model.train()
        # training loss
        train_loss = 0
        # number of correctly predicted images
        train_correct = 0
        # number of images
        train_data_number = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            # move to gpu
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # predict the output
            outputs = model(images)
            # calculate the loss
            ce_loss = criterion(outputs, labels)
            # focal=FocalLoss()
            # focal_loss = focal(outputs, labels)
            # loss = ce_loss+focal_loss
            loss = ce_loss
            # backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # turn logits into labels
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels).item()
            train_data_number += labels.size(0)
        # accuracy of this epoch
        train_acc = train_correct/train_data_number
        # save the results of this round into lists
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"epoch: {e}"
              f"train Loss: {train_loss:.4f} train Acc: {train_acc:.4f}")
        # how much time it cost to finish this epoch
        print(f"Time: {(time.time() - start_time):.2f}")


def val(model, val_loader, criterion, device):
    start_time = time.time()
    model.eval()
    # validation loss
    val_loss = 0
    # number of correctly predicted images
    val_correct = 0
    # number of images
    val_data_number = 0

    with torch.no_grad():  # disable gradient calculation for efficiency
        for images, labels in val_loader:

            images, labels = images.to(device), labels.to(device)
            # Prediction
            outputs = model(images)
            # loss calculation
            ce_loss = criterion(outputs, labels)
            # focal=FocalLoss()
            # focal_loss = focal(outputs, labels)
            # loss = ce_loss+focal_loss
            loss = ce_loss
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data).item()
            val_data_number += labels.size(0)
    # accuracy of the whole dataset
    val_acc = val_correct / val_data_number
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print('Loss: {:.4f}, Accuracy: {:.4f}%'.format(val_loss, val_acc))
    print(f'Time: {(time.time() - start_time):.2f}')
    return val_acc


def test(model, test_loader, criterion, device, output_csv_path):
    model.eval()
    answer = []

    with torch.no_grad():  # disable gradient calculation for efficiency
        for images, image_names in test_loader:
            images = images.to(device)
            # Prediction
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for name, label in zip(image_names, preds.cpu().numpy()):
                answer.append((name, label))
    df = pd.DataFrame(answer, columns=["image_name", "pred_label"])
    df.to_csv(output_csv_path, index=False)

# class FocalLoss(nn.Module):
#     def __init__(self, weight=None,
#                  gamma=2.5, reduction='mean'):
#         nn.Module.__init__(self)
#         self.weight=weight
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, input_tensor, target_tensor):
#         log_prob = F.log_softmax(input_tensor, dim=-1)
#         prob = torch.exp(log_prob)
#         return F.nll_loss(
#             ((1 - prob) ** self.gamma) * log_prob,
#             target_tensor,
#             weight=self.weight,
#             reduction = self.reduction
#         )


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # two branch
        # average
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        # max
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # two branch
        # average
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # max
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # channel->spatial
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# basic block of resnet50
# CBAM between bn3 and concat(relu)
# modified to resnext50
# conv1: 1*1, stride=1
# conv2: 3*3, stride=stride_num, groups=32
# conv3: 1*1, stride=1
class Bottleneck(nn.Module):
    def __init__(self, bottleneck_in_channels, bottleneck_out_channels_quarter,
                 downsample=None, stride_num=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            bottleneck_in_channels, bottleneck_out_channels_quarter*2,
            kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(
            bottleneck_out_channels_quarter*2,
            eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(
            bottleneck_out_channels_quarter*2,
            bottleneck_out_channels_quarter*2,
            kernel_size=(3, 3), stride=(stride_num, stride_num),
            padding=(1, 1), groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(
            bottleneck_out_channels_quarter*2,
            eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(
            bottleneck_out_channels_quarter*2,
            bottleneck_out_channels_quarter*4,
            kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_out_channels_quarter*4,
                                  eps=1e-05, momentum=0.1,
                                  affine=True, track_running_stats=True)
        self.cbam = CBAM(bottleneck_out_channels_quarter*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        original_x = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x = self.cbam(x)

        # first bottleneck block of each layer needs downsample block
        if self.downsample is not None:
            original_x = self.downsample(original_x)
        x += original_x

        x = self.relu(x)

        return x


class ResNeXt50_32by4d_CBAM(nn.Module):
    def __init__(self, num_classes):
        super(ResNeXt50_32by4d_CBAM, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2),
            padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = self.construct_layer(
            layer_in_channels=64, layer_out_channels_quarter=64,
            Bottleneck_num=3, conv2_stride=1)
        self.layer2 = self.construct_layer(
            layer_in_channels=256, layer_out_channels_quarter=128,
            Bottleneck_num=4, conv2_stride=2)
        self.layer3 = self.construct_layer(
            layer_in_channels=512, layer_out_channels_quarter=256,
            Bottleneck_num=6, conv2_stride=2)
        self.layer4 = self.construct_layer(
            layer_in_channels=1024, layer_out_channels_quarter=512,
            Bottleneck_num=3, conv2_stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(
            in_features=2048, out_features=num_classes, bias=True)

    def construct_layer(self, layer_in_channels, layer_out_channels_quarter,
                        Bottleneck_num, conv2_stride):
        # layer1: Bottleneck_num = 3
        # layer2: Bottleneck_num = 4
        # layer3: Bottleneck_num = 6
        # layer4: Bottleneck_num = 3
        layers = []
        downsample_layers = []

        # first bottleneck block has downsample block
        # (identity) (1*1 convolution layer)
        downsample_layers.append(
            nn.Conv2d(
                layer_in_channels, layer_out_channels_quarter*4,
                kernel_size=1, stride=conv2_stride, bias=False))
        downsample_layers.append(
            nn.BatchNorm2d(layer_out_channels_quarter*4,
                           eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True))
        downsample = nn.Sequential(*downsample_layers)
        # append first bottleneck block
        layers.append(
            Bottleneck(
                bottleneck_in_channels=layer_in_channels,
                bottleneck_out_channels_quarter=layer_out_channels_quarter,
                downsample=downsample,
                stride_num=conv2_stride))

        # 2~last bottleneck blocks' input channels number
        bottleneck_in_channels = layer_out_channels_quarter*4

        # 2~last bottleneck blocks
        for i in range(Bottleneck_num - 1):
            layers.append(
                Bottleneck(
                    bottleneck_in_channels=bottleneck_in_channels,
                    bottleneck_out_channels_quarter=layer_out_channels_quarter,
                    downsample=None,
                    stride_num=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)    # in=(   3,224*224), out=(  64,112*112)
        x = self.bn1(x)      # in=(  64,112*112), out=(  64,112*112)
        x = self.relu(x)     # in=(  64,112*112), out=(  64,112*112)
        x = self.maxpool(x)  # in=(  64,112*112), out=(  64, 56*56 )
        x = self.layer1(x)   # in=(  64, 56*56 ), out=( 256, 56*56 )
        x = self.layer2(x)   # in=( 256, 56*56 ), out=( 512, 28*28 )
        x = self.layer3(x)   # in=( 512, 28*28 ), out=(1024, 14*14 )
        x = self.layer4(x)   # in=(1024, 14*14 ), out=(2048,  7*7  )
        x = self.avgpool(x)  # in=(2048,  7*7  ), out=(2048,  1*1  )
        x = x.reshape(x.shape[0], -1)  # in=(2048,1*1), out=2048
        x = self.fc(x)  # in=2048, out=100
        return x


if __name__ == "__main__":
    BATCH_SIZE = 128
    EPOCHS = 100

    save_model_name = "./ResNeXt_50_CBAM/ResNeXt_50_CBAM.pth"
    # last epoch model
    save_model_name_end = "./ResNeXt_50_CBAM/ResNeXt_50_CBAM_end.pth"
    output_csv_path = "./ResNeXt_50_CBAM/ResNeXt_50_CBAM.csv"
    # last epoch csv
    output_csv_path_end = "./ResNeXt_50_CBAM/ResNeXt_50_CBAM_end.csv"
    output_txt_path = "./ResNeXt_50_CBAM/ResNeXt_50_CBAM.txt"

    # image folder path
    train_dir = "./hw1-data/data/train"
    val_dir = "./hw1-data/data/val"
    test_dir = "./hw1-data/data/test"

    # check if gpu os available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("cuda")
    else:
        device = torch.device('cpu')

    # dataset
    train_dataset = CustomImageDataset(
        root_dir=train_dir, val=False, test=False)
    val_dataset = CustomImageDataset(
        root_dir=val_dir, val=True, test=False)
    test_dataset = CustomImageDataset(
        root_dir=test_dir, val=False, test=True)
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # initialize the model
    model = ResNeXt50_32by4d_CBAM(num_classes=1000)
    print(model)
    # load pretrained weight except final layer
    pretrained_model = models.resnext50_32x4d(
        weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    pretrained_dict = pretrained_model.state_dict()
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if not k.startswith("fc.")}
    model.load_state_dict(pretrained_dict, strict=False)
    # change final layer fc
    # output=1000->100
    model.fc = nn.Linear(model.fc.in_features, 100)
    # count the number of parameters
    print("parameters: ", sum(p.numel() for p in model.parameters()))

    # send model to gpu
    model = model.to(device)
    # use only cross entropy loss
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr = 0.001)
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
    # epoch 1~50, lr=0.01
    # epoch 51~100, lr=0.001
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    # log the best accuracy
    # for find the best model (highest score with validation dataset)
    best_acc = 0

    for i in range(EPOCHS):
        print("EPOCH:", i+1)
        # train the model
        train(model, train_loader, optimizer, criterion, device, 1, i+1)
        # scheduler counting
        scheduler.step()
        # validation, return validation accuracy
        val_acc = val(model, val_loader, criterion, device)
        # log losses and accuracy into txt file
        with open(output_txt_path, "a") as f:
            f.write(f"Epoch {i+1} "
                    f"{train_losses[i]:.4f} {train_accuracies[i]:.4f} "
                    f"{val_losses[i]:.4f} {val_accuracies[i]:.4f}\n")
            f.write("-" * 40 + "\n")

        # create a visualization plot every 5 epoch
        if (i+1) % 5 == 0:
            plt.figure(figsize=(12, 6))
            # Loss
            plt.subplot(1, 2, 1)
            plt.plot(
                range(0, i + 1),
                train_losses, label='Train Loss', color='blue')
            plt.plot(
                range(0, i + 1),
                val_losses, label='Validation Loss', color='red')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.title(f'Loss of Epoch 1~{i+1}')

            # Accuracy
            plt.subplot(1, 2, 2)
            plt.plot(
                range(0, i + 1),
                train_accuracies, label='Train Accuracy', color='blue')
            plt.plot(
                range(0, i + 1),
                val_accuracies, label='Validation Accuracy', color='red')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title(f'Accuracy of Epoch 1~{i+1}')

            plt.tight_layout()

            image_path = f"./ResNeXt_50_CBAM/ResNeXt_50_CBAM_epoch_{i+1}.png"
            plt.savefig(image_path)
            plt.close()

        # if get higher score with validation dataset
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"New best model saved with Val Acc: {val_acc:.4f}")
            torch.save(model.state_dict(), save_model_name)
    # save the model of final epoch
    torch.save(model.state_dict(), save_model_name_end)
    # load the best model and do the test
    state_dict = torch.load(save_model_name)
    model.load_state_dict(state_dict)
    test(model, test_loader, criterion, device, output_csv_path)
    # load the final epoch model and do the test
    state_dict = torch.load(save_model_name_end)
    model.load_state_dict(state_dict)
    test(model, test_loader, criterion, device, output_csv_path_end)
