#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import glob
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# 假设你已有 Mydatasetpro 类（接受图像路径列表和标签列表）
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

# ========================
# 1. 加载模型和原始标签
# ========================
model = torch.load(r"20251223_GLN_spect_p_f_202_fl_22.pt", map_location='cpu')
model.eval()  # 切换到评估模式

# 2. 定义图像变换（需与训练时一致！）
# 对数据进行转换处理
transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((112,224)), #做的第一步转换
                transforms.ToTensor(), #第二步转换，作用：第一转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
                #transforms.Normalize(std=(0.5,0.5,0.5),mean=(0.5,0.5,0.5))
])

# ========================
# 3. 遍历所有噪声子文件夹并评估
# ========================
base_dir =r'cwt_images_1218val'
results_summary = {}  # 存储每组的准确率
original_labels=np.loadtxt("train_label1223.txt")
# 自动获取所有 snr_XX 文件夹
snr_folders = [d for d in os.listdir(base_dir) if d.startswith('snr_') and os.path.isdir(os.path.join(base_dir, d))]
print(snr_folders)
for snr_folder in sorted(snr_folders):
    snr_path = os.path.join(base_dir, snr_folder)
    noise_types = [d for d in os.listdir(snr_path) if os.path.isdir(os.path.join(snr_path, d))]
    
    for noise_type in sorted(noise_types):
        print(f"\n Evaluating → {snr_folder} / {noise_type}")
        
        img_dir = os.path.join(snr_path, noise_type)
        # 获取所有图像路径，并按文件名排序以保证顺序一致
        img_paths = sorted(glob.glob(os.path.join(img_dir, "sample_*.jpg")))

#         if len(img_paths) != 2560:
#             print(f"⚠️ 警告: {snr_folder}/{noise_type} 中图像数量={len(img_paths)} ≠ 2560，跳过")
#             continue
        
        # 提取索引（从 sample_0000.png → 0）
        indices = []
        for path in img_paths:
            basename = os.path.basename(path)
            idx = int(basename.split('_')[1].split('.')[0])  # 'sample_0000.png' → 0
            indices.append(idx)
        indices = np.array(indices)
        
#         # 检查是否为 0~1279 的排列
#         if not (np.sort(indices) == np.arange(2560)).all():
#             print(f"⚠️ 警告: 索引不完整或重复，跳过 {snr_folder}/{noise_type}")
#             continue
        
        # 按索引对齐标签
        aligned_labels = original_labels[indices]  # shape: (2560,)
        
        # 创建 dataset 和 dataloader
        try:
            val_ds = Mydatasetpro(img_paths, aligned_labels.tolist(), transform)
            val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=False)
        except Exception as e:
            print(f"❌ Dataset 创建失败: {e}")
            continue
        
        # 推理
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, labels in val_dl:
                # images: (B, C, H, W), labels: (B,)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
        
        # 合并结果
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        
        # 计算准确率
        acc = accuracy_score(y_true, y_pred)
        results_summary[(snr_folder, noise_type)] = acc
        
        print(f"  → Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print("  → Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

# ========================
# 4. 打印汇总结果
# ========================
print("\n" + "="*60)
print("📊 最终准确率汇总:")
print("="*60)
for (snr, noise), acc in sorted(results_summary.items()):
    print(f"{snr:>8} | {noise:<12} | Accuracy: {acc*100:6.2f}%")

# 可选：保存为 CSV
import pandas as pd
summary_df = pd.DataFrame([
    {'SNR': snr, 'NoiseType': noise, 'Accuracy': acc}
    for (snr, noise), acc in results_summary.items()
])


# In[11]:


summary_df.to_csv("cwt_evaluation_summary-20251223_GLN_spect_p_f_202_fl_22.csv")
print(f"\n✅ 汇总结果已保存)





