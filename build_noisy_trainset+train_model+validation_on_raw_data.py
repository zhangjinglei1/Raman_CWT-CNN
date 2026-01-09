#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
import random
import numpy as np
import pandas as pd
from typing import List, Tuple
import glob
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


def build_sampled_dataset(
        root_dir: str,
        snr_list: List[int],
        ratios: List[float],
        original_labels: List[int],
        noise_type: str = 'gaussian',
        seed: int = 42
) -> Tuple[List[str], List[int]]:
    """
    从 CWT 图像数据集中按 SNR 分层随机采样，构建训练集路径和标签。

    Parameters:
        root_dir (str): 数据根目录，如 "D:/Desktop/cwt_images_1223train"
        snr_list (List[int]): SNR 列表，如 [0, 5, 10, 15, 20, 25, 30]
        ratios (List[float]): 对应每个 SNR 的采样比例，如 [0.1, 0.2, ..., 1.0]
        original_labels (List[int]): 全局标签列表，长度应 >= 最大样本索引（如 8064）
        noise_type (str): 噪声类型子文件夹名，默认 'gaussian'
        seed (int): 随机种子，保证可复现

    Returns:
        train_paths (List[str]): 采样后的图像路径列表
        train_labels (List[int]): 对应的真实标签列表
    """
    assert len(snr_list) == len(ratios), "SNR 列表与比例列表长度必须一致"

    random.seed(seed)

    train_paths = []
    train_labels = []

    for snr, ratio in zip(snr_list, ratios):
        if ratio <= 0:
            continue

        snr_folder = os.path.join(root_dir, f"snr_{snr}", noise_type)
        if not os.path.exists(snr_folder):
            print(f"⚠️ 跳过不存在的路径: {snr_folder}")
            continue

        # 获取所有 sample_*.jpg 文件
        all_files = [f for f in os.listdir(snr_folder) if f.startswith("sample_") and f.endswith(".jpg")]
        all_files.sort()  # 确保顺序一致：sample_0000.jpg, sample_0001.jpg, ...

        n_total = len(all_files)
        if n_total == 0:
            print(f"⚠️ {snr_folder} 中无图像文件")
            continue

        # 计算采样数量
        n_sample = int(round(ratio * n_total))
        n_sample = min(n_sample, n_total)  # 防止四舍五入超过总数
        if n_sample == 0 and ratio > 0:
            n_sample = 1  # 至少采 1 个

        # 随机采样索引（对应 sample_XXXX 的 XXXX）
        sampled_indices = random.sample(range(n_total), n_sample)

        for idx in sampled_indices:
            filename = all_files[idx]  # e.g., "sample_0123.jpg"
            # 提取编号：0123 → 123
            try:
                sample_id = int(filename.split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                print(f"⚠️ 无法解析文件名: {filename}")
                continue

            # 检查标签是否存在
            if sample_id >= len(original_labels):
                print(f"⚠️ 标签越界: sample_id={sample_id}, max_label_index={len(original_labels) - 1}")
                continue

            file_path = os.path.join(snr_folder, filename)
            train_paths.append(file_path)
            train_labels.append(original_labels[sample_id])

    print(f"✅ 共采样 {len(train_paths)} 张图像，对应 {len(set(train_labels))} 个类别")
    return train_paths, train_labels

original_labels=np.loadtxt(r"train_label1223.txt")
# 设置参数
root = r"cwt_images_1223train"
snrs = [0, 5, 10, 15, 20, 25, 30]
ratios = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]  # 取样比例

# 构建训练集
train_paths, train_labels = build_sampled_dataset(
    root_dir=root,
    snr_list=snrs,
    ratios=ratios,
    original_labels=original_labels,
    noise_type='gaussian',
    seed=123
)

# 可选：打乱整个训练集（虽然分层采样后通常再 shuffle）
combined = list(zip(train_paths, train_labels))
random.shuffle(combined)
train_paths, train_labels = zip(*combined)
train_paths, train_labels = list(train_paths), list(train_labels)


7# 使用glob方法来获取数据图片的所有路径
patient_label=pd.read_csv(r"patient_label202.csv")
train_patient=np.loadtxt('train_patient9-202.txt')
test_patient=np.loadtxt('test_patient9-202.txt')
val_patient=np.loadtxt('val_patient9-202.txt')
import glob
path=r'patient202-cwt'
all_imgs_path=[]
all_imgs_path1=[]
all_imgs_path2=[]
all_labels=[]
all_labels1=[]
all_labels2=[]
for j in range(train_patient.shape[0]):
#     print(path+'\\'+str(int(train_patient[j]))+'\\'+'*.jpg')
    all_imgs = glob.glob(path+'\\'+str(int(train_patient[j]))+'\\*.jpg')
#     print(all_imgs)
    for img0 in all_imgs:
        all_imgs_path.append(img0)
    all_0=patient_label['label#'][patient_label['patient_code']==int(train_patient[j])]
    all_0=all_0.values.tolist()
    for item0 in all_0:
        for n in range(64):
            all_labels.append(item0)
for k in range(test_patient.shape[0]):
    all_imgs1 = glob.glob(path+'\\'+str(int(test_patient[k]))+'\\*.jpg')
    for img1 in all_imgs1:
        all_imgs_path1.append(img1)
    all_1=patient_label['label#'][patient_label['patient_code']==int(test_patient[k])]
    all_1=all_1.values.tolist()
    for item1 in all_1:
        for n in range(64):
            all_labels1.append(item1)
for p in range(val_patient.shape[0]):
    all_imgs2= glob.glob(path+'\\'+str(int(val_patient[p]))+'\\*.jpg')
    for img2 in all_imgs2:
        all_imgs_path2.append(img2)
    all_2=patient_label['label#'][patient_label['patient_code']==int(val_patient[p])]
    all_2=all_2.values.tolist()
    for item2 in all_2:
        for n in range(64):
            all_labels2.append(item2)



#训练集加上无噪声数据
all_imgs_path=all_imgs_path+train_paths
all_labels=all_labels+train_labels


# In[29]:


from torch.utils import data
# 通过创建data.Dataset子类Mydataset来创建输入
class Mydataset(data.Dataset):
    # 类初始化
    def __init__(self, root):
        self.imgs_path = root

    # 进行切片
    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        return img_path

    # 返回长度
    def __len__(self):
        return len(self.imgs_path)


# 对数据进行转换处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((112, 224)),  # 做的第一步转换
    transforms.ToTensor(),  # 第二步转换，作用：第一转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
    transforms.Normalize(std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406])
])


class Mydatasetpro(data.Dataset):
    # 类初始化
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    # 进行切片
    def __getitem__(self, index):  # 根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
        img = self.imgs[index]
        label = self.labels[index]
        pil_img = Image.open(img)  # pip install pillow
        data = self.transforms(pil_img)
        return data, label

    # 返回长度
    def __len__(self):
        return len(self.imgs)


BATCH_SIZE = 256

from torch.utils import data

train_imgs = all_imgs_path
train_labels = all_labels
test_imgs = all_imgs_path1
test_labels = all_labels1

train_ds = Mydatasetpro(train_imgs, train_labels, transform)  # TrainSet TensorData
test_ds = Mydatasetpro(test_imgs, test_labels, transform)  # TestSet TensorData

train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)  # TrainSet Labels
test_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)  # TestSet Labels

val_imgs = all_imgs_path2
val_labels = all_labels2
val_ds = Mydatasetpro(val_imgs, val_labels, transform)  # TestSet TensorData
val_dl = data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)  # TestSet Labels




import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        '''
        alpha: list or tensor of shape [2]，分别是类别0和类别1的权重
        gamma: 聚焦参数
        '''
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.tensor([0.5, 0.5])
        else:
            self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [batch, 2], targets: [batch]
        probs = F.softmax(logits, dim=1)              # [batch, 2]
        targets = targets.long()
        
        # 选取每个样本对应的类别概率
        pt = probs[range(len(targets)), targets]       # [batch]
        # 选取每个样本的alpha权重
        at = self.alpha[targets]                      # [batch]
        
        # focal loss公式
        loss = -at * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)   # [batch]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
#GoogleNet训练函数
def fit(epoch, model, trainloader, testloader):
    lr=[]
    corret = 0
    total = 0
    running_loss = 0

    model.train()
    for x, y in trainloader:
#         print(x,y)
        y = torch.tensor(y, dtype=torch.long)
#         if torch.cuda.is_available():
#             x, y = x.to('cuda'), y.to('cuda')
        
        y_pred,aux2,aux1 = model(x)
        loss = loss_fn(y_pred,y)
        optim.zero_grad()
        loss.backward()

        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            corret = corret +(y_pred == y).sum().item()
            total = total + y.size(0)
            running_loss = running_loss + loss.item()

    epoch_loss = running_loss/len(trainloader.dataset)      #!!!!小心变量名错误
    epoch_acc = corret/total
    optim.step()
    scheduler.step()
    print("第%d个epoch的学习率：%f" % (epoch, optim.param_groups[0]['lr']))
    lr.append(scheduler.get_lr()[0])
        
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in testloader:
#             print(y)
#             print(y.type)
            y = torch.tensor(np.array(y), dtype=torch.long)
#             if torch.cuda.is_available():
#                 x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred =torch.argmax(y_pred, dim=1)
            test_correct = test_correct + (y_pred == y).sum().item()
            test_total = test_total + y.size(0)
            test_running_loss = test_running_loss + loss.item()
        epoch_test_loss = test_running_loss / len(testloader.dataset)  # !!!!小心变量名错误
        epoch_test_acc = test_correct / test_total
    
    print('epoch:', epoch,
          'loss:', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss', round(epoch_test_loss, 3),
          'test_accuracy', round(epoch_test_acc, 3)
        )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

model=googlenet
# model = DCNN()
# model=torch.load("0825jg_gln_max.pt",map_location=torch.device('cpu'))
# model=torch.load("20250829_GLN_spect_p_f_202_fl_1.pt")

#Focal loss
alpha_cancer = 0.394
alpha_non_cancer = 0.606
alpha = [alpha_cancer, alpha_non_cancer]  
gamma = 3.0

criterion = FocalLoss(alpha=alpha, gamma=gamma)
loss_fn=criterion


#定义优化函数和损失函数    
lr=0.0001###0.00001
# loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.99))
# optim = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optim,T_max = 200 )#T_max ，这个参数指的是cosine函数经过多少次更新完成四分之一个周期

epochs = 200
train_loss = []
train_acc = []
test_loss = []
test_acc = []
lr = []
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch, model, train_dl, test_dl)
#     epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = dcnn_fit(epoch, model, train_dl, test_dl)

    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

    torch.save(model, '20251223_GLN_spect_p_f_202_fl_'+str(epoch)+'.pt')
#     torch.save(model.state_dict(),'20250814_DCNN_152_'+str(epoch)+ '-model_weights.pt')


# In[37]:


#validation_on_raw_data
y_pred=[]
y_prob=[]
y_test=[]
patient_acc=[]
path=r"D:\Desktop\raw-data-df\patient202-cwt"
all_imgs_path2=[]                                             
all_labels2=[]
for p in val_patient:
    y_pred0=[]
    y_test0=[]
    y_prob0=[]
    all_imgs_path2= glob.glob(path+'\\'+str(int(p))+'\\*.jpg')
    all_2=patient_label['label#'].iloc[int(p)]
    all_labels2=[]
    for n in range(64):
            all_labels2.append(all_2)
    val_ds=Mydatasetpro(all_imgs_path2, all_labels2, transform)
    val_dl = data.DataLoader(val_ds, batch_size=32, shuffle=True,drop_last=True)

    correct = 0
    total = 0
    for data0 in val_dl:

        images, labels = data0     
        y_test0.append(labels.numpy())
        labels = labels.long()
    #     images =  images.to(device)
    #     labels = labels.to(device)
        outputs = model(images)
        y_prob0.append(outputs.detach().numpy())
        outputs = torch.argmax(outputs,1)
        y_pred0.append(outputs.detach().numpy())

        total += labels.size(0)
        correct += (outputs == labels).sum().item()

    y_prob0=np.array(y_prob0)
    y_pred0=np.array(y_pred0)
    y_test0=np.array(y_test0)

    patient_acc.append(100*correct/total)
    print(patient_label['patient#'].iloc[int(p)],'accuracy on val set: %d %% ' % (100*correct/total))
    y_test.append(y_test0)
    y_pred.append(y_pred0)
    y_prob.append(y_prob0)

y_pred=np.array(y_pred).reshape(1,-1)[0]
#     print(y_pred0)
y_test=np.array(y_test).reshape(1,-1)[0]
#     print(y_test0)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
print(np.mean(patient_acc))


# In[ ]:




