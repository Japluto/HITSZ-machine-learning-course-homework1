# -*- coding: utf-8 -*-
"""Image Classification with Data Augmentation for 128x128 Input"""

_exp_name = "homework_1_ZSH_improved" # 为新实验起一个新名字
# 适配Windows操作系统
OS = "Windows"

# Import necessary packages.
import numpy as np
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset

# This is for the progress bar.
from tqdm.auto import tqdm
import random

myseed = 1234  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# The FOLDER of the dataset.
_dataset_dir = "./food11"

# --- [核心修改处] ---
# 增加了丰富的数据增强，但保持输入尺寸为 128x128
train_tfm = transforms.Compose([
    # 将图片尺寸统一调整为 128x128
    transforms.Resize((128, 128)),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(),
    # 随机在-20到+20度之间旋转
    transforms.RandomRotation(20),
    # 随机改变亮度、对比度和饱和度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # 转换为PyTorch Tensor
    transforms.ToTensor(),
])

# 验证集和测试集不使用数据增强，保持原样
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 修正后的FoodDataset类，确保标签提取正确
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
            label = -1 # test has no label
            
        return im,label

# 原始的Classifier模型，为128x128输入设计
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x

def Training_Demo():
    """# Training"""

    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The number of batch size.
    batch_size = 64

    # The number of training epochs.
    n_epochs = 30

    # If no improvement in 'patience' epochs, early stop.
    patience = 5

    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = nn.CrossEntropyLoss()

    # Initialize a model, and put it on the device specified.
    model = Classifier().to(device)

    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)


    # Construct train and valid datasets.
    # The whole dataset is split into train and valid set.
    train_set = FoodDataset(os.path.join(_dataset_dir,"training"), tfm=train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_set = FoodDataset(os.path.join(_dataset_dir,"validation"), tfm=test_tfm)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # Initialize trackers, these are not parameters and should not be changed
    stale = 0
    best_acc = 0

    for epoch in range(n_epochs):

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()
            #print(imgs.shape,labels.shape)

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
        
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            #break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


        # update logs
        if valid_acc > best_acc:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


        # save models
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch+1}, saving model")
            torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"No improvement for {patience} consecutive epochs, early stopping")
                break


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

    #create test csv
    def pad4(i):
        return "0"*(4-len(str(i)))+str(i)
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