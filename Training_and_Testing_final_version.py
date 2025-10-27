# -*- coding: utf-8 -*-
"""Image Classification - Final Version with EfficientNet-B0 Transfer Learning"""

_exp_name = "homework_1_ZSH_improved_final" # 为最终实验起一个新名字
# 适配Windows操作系统
OS = "Windows"

# Import necessary packages.
import numpy as np
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
# 导入预训练模型库
import torchvision.models as models

# This is for the progress bar.
from tqdm.auto import tqdm
import random

myseed = 2233  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# The FOLDER of the dataset.
_dataset_dir = "./food11"

# 为EfficientNet模型适配的、更丰富的数据处理流程
# EfficientNet-B0 默认输入尺寸是 224x224
image_size = 224
train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    # 这是ImageNet预训练模型通用的标准化参数
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 验证集和测试集的图像变换
test_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FoodDataset(Dataset):
    def __init__(self,path,tfm=test_tfm,files = None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        try:
            filename = os.path.basename(fname)
            label = int(filename.split("_")[0])
        except:
            label = -1
        return im,label

# 使用预训练的 EfficientNet-B0 模型
# 我们不再使用旧的 Classifier 类
class Classifier(nn.Module):
    def __init__(self, num_classes=11):
        super(Classifier, self).__init__()
        # 1. 加载预训练的EfficientNet-B0模型
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # 2. "冻结"所有特征提取层的参数
        for param in self.model.features.parameters():
            param.requires_grad = False
            
        # 3. 替换掉原来的分类器头部
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True), # 加入Dropout防止过拟合
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def Training_Demo():
    """# Training"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128 # 使用更大的batch size
    n_epochs = 40 # 更强大的模型可能需要更多时间收敛
    patience = 10  # 可以给新模型多一点耐心
    
    # 学习率是关键，对于迁移学习，初始学习率可以稍大一些
    learning_rate = 1e-3

    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 使用更先进的学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    train_set = FoodDataset(os.path.join(_dataset_dir,"training"), tfm=train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_set = FoodDataset(os.path.join(_dataset_dir,"validation"), tfm=test_tfm)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    stale = 0
    best_acc = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = []
        train_accs = []
        for batch in tqdm(train_loader):
            imgs, labels = batch
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)
        
        # 在每个epoch结束后，更新学习率
        scheduler.step()
        
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}, lr = {current_lr:.6f}")

        model.eval()
        valid_loss = []
        valid_accs = []
        for batch in tqdm(valid_loader):
            imgs, labels = batch
            with torch.no_grad():
                logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc)
        
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch+1}, saving model with acc {valid_acc:.5f}")
            torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"No improvement for {patience} consecutive epochs, early stopping")
                break

# Testing_Demo 和 Predict_demo 函数保持不变，因为它们会自动加载新的模型架构
def Testing_Demo():
    """# Testing and generate result.txt""" 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    
    test_set = FoodDataset(os.path.join(_dataset_dir,"validation"), tfm=test_tfm)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model_best = Classifier().to(device)
    model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
    model_best.eval()
    
    ids = []
    predictions = []
    
    file_idx = 0
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader):
            logits = model_best(imgs.to(device))
            test_label = np.argmax(logits.cpu().data.numpy(), axis=1)
            for i in range(len(imgs)):
                path = test_loader.dataset.files[file_idx]
                filename = os.path.basename(path)
                ids.append(filename.split('.')[0])
                predictions.append(test_label[i])
                file_idx += 1
    
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            logits = model_best(imgs.to(device))
            correct = (logits.argmax(dim=-1) == labels.to(device)).sum().item()
            total_correct += correct
            total_samples += len(labels)
    
    valid_acc = total_correct / total_samples

    with open("result.txt", "w") as f:
        f.write(f"ID Category Accuracy: {valid_acc}\n")
        for i in range(len(ids)):
            f.write(f"{ids[i]} {predictions[i]}\n")

def Predict_demo():
    """# Predict on test set and generate submission.csv"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    
    import pandas as pd
    test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=test_tfm)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    model_best = Classifier().to(device)
    model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
    model_best.eval()
    
    prediction = []
    with torch.no_grad():
        for data,_ in tqdm(test_loader):
            test_pred = model_best(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            prediction += test_label.squeeze().tolist()

    df = pd.DataFrame()
    test_ids = [os.path.basename(f).split('.')[0] for f in test_set.files]
    df["Id"] = test_ids
    df["Category"] = prediction
    df.sort_values(by="Id", inplace=True)
    df.to_csv("submission.csv",index = False)

if __name__ == '__main__':
    Training_Demo()
    Testing_Demo()
    Predict_demo()