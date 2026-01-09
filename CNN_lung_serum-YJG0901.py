# %%
import glob
import torch
from torch.utils import data
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt

import sklearn


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



patient_label=pd.read_csv(r"/mnt/petrelfs/yangjiange/lele/patient_label202.csv")


train_patient=np.loadtxt('/mnt/petrelfs/yangjiange/lele/train_patient9-202.txt')
test_patient=np.loadtxt('/mnt/petrelfs/yangjiange/lele/test_patient9-202.txt')
val_patient=np.loadtxt('/mnt/petrelfs/yangjiange/lele/val_patient9-202.txt')


# 使用glob方法来获取数据图片的所有路径
import glob
path=r'/mnt/petrelfs/yangjiange/all_data/yangjiange/patient202new/patient202-cwt'
# path='D:\Desktop\raw-data-df\processed-scalograms-gaus2-21'
all_imgs_path=[]
all_imgs_path1=[]
all_imgs_path2=[]                                             
all_labels=[]
all_labels1=[]
all_labels2=[]
for j in range(train_patient.shape[0]):
#     print(path+'/'+str(int(train_patient[j]))+'/'+'*.jpg')
    all_imgs = glob.glob(path+'/'+str(int(train_patient[j]))+'/*.jpg')
#     print(all_imgs)
    for img0 in all_imgs:
        all_imgs_path.append(img0)
    all_0=patient_label['label#'][patient_label['patient_code']==int(train_patient[j])]
    all_0=all_0.values.tolist()
    for item0 in all_0:
        for n in range(64):
            all_labels.append(item0)
for k in range(test_patient.shape[0]):
    all_imgs1 = glob.glob(path+'/'+str(int(test_patient[k]))+'/*.jpg')
    for img1 in all_imgs1:
        all_imgs_path1.append(img1)
    all_1=patient_label['label#'][patient_label['patient_code']==int(test_patient[k])]
    all_1=all_1.values.tolist()
    for item1 in all_1:
        for n in range(64):
            all_labels1.append(item1)
for p in range(val_patient.shape[0]):    
    all_imgs2= glob.glob(path+'/'+str(int(val_patient[p]))+'/*.jpg')
    for img2 in all_imgs2:
        all_imgs_path2.append(img2)
    all_2=patient_label['label#'][patient_label['patient_code']==int(val_patient[p])]
    all_2=all_2.values.tolist()
    for item2 in all_2:
        for n in range(64):
            all_labels2.append(item2)

# # 去除list外面的中括号
# all_imgs_path=all_imgs_path[0]
# all_imgs_path1=all_imgs_path1[0]
# all_imgs_path2=all_imgs_path2[0]
# all_labels=all_labels[0]
# all_labels1=all_labels1[0]
# all_labels2=all_labels2[0]

import numpy as np  
all_labels=np.array(all_labels)
all_labels1=np.array(all_labels1)
all_labels2=np.array(all_labels2)


species = ['lc','hc']
species_to_id = dict((c, i) for i, c in enumerate(species))
print(species_to_id)
id_to_species = dict((v, k) for k, v in species_to_id.items())
print(id_to_species)


from torch.utils import data
#通过创建data.Dataset子类Mydataset来创建输入
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




#利用自定义类Mydataset创建对象serum_dataset
serum_dataset = Mydataset(all_imgs_path)
print(len(serum_dataset)) #返回文件夹中图片总个数
print(serum_dataset[12:15])#切片，显示第12至第十五张图片的路径
serum_datalodaer = torch.utils.data.DataLoader(serum_dataset, batch_size=5) #每次迭代时返回五个数据
print(next(iter(serum_datalodaer)))



# 对数据进行转换处理
transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((112,224)), #做的第一步转换
                transforms.ToTensor(), #第二步转换，作用：第一转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
                transforms.Normalize(std=[0.229, 0.224, 0.225],mean = [0.485, 0.456, 0.406])
])



class Mydatasetpro(data.Dataset):
# 类初始化
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform
# 进行切片
    def __getitem__(self, index):                #根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
        img = self.imgs[index]
        label = self.labels[index]
        pil_img = Image.open(img)                 #pip install pillow
        data = self.transforms(pil_img)
        return data, label
# 返回长度
    def __len__(self):
        return len(self.imgs)


BATCH_SIZE = 256
serum_dataset = Mydatasetpro(all_imgs_path, all_labels, transform)
serum_datalodaer = data.DataLoader(
                            serum_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            drop_last=True
)

imgs_batch, labels_batch = next(iter(serum_datalodaer))
print(imgs_batch.shape)


plt.figure(figsize=(12, 8))
for i, (img, label) in enumerate(zip(imgs_batch[:6], labels_batch[:6])):
    img = img.permute(1, 2, 0).numpy()
    plt.subplot(2, 3, i+1)
    plt.title(id_to_species.get(label.item()))
    plt.imshow(img)
plt.show()#展示图片



from torch.utils import data
train_imgs = all_imgs_path
train_labels = all_labels
test_imgs = all_imgs_path1
test_labels = all_labels1

train_ds = Mydatasetpro(train_imgs, train_labels, transform) #TrainSet TensorData
test_ds = Mydatasetpro(test_imgs, test_labels, transform) #TestSet TensorData

train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)#TrainSet Labels
test_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)#TestSet Labels




val_imgs = all_imgs_path2
val_labels = all_labels2
val_ds = Mydatasetpro(val_imgs, val_labels, transform) #TestSet TensorData
val_dl = data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)#TestSet Labels





import torch
import torchvision
googlenet= net = torchvision.models.googlenet(pretrained = True, aux_logits=True)  # 使用在ImageNet上训练好的参数
# print(net)
## 重新定义Fully Connected layer
import torch.nn as nn
net.fc = nn.Linear(in_features=1024, out_features=2, bias=True)####




# # 加载预训练的 ResNet-18
# resnet18 = torchvision.models.resnet18(pretrained=True)  # 加载 ImageNet 预训练参数
# resnet18.fc = nn.Linear(in_features=512, out_features=2, bias=True)
# net = resnet18




import torch
import torch.nn as nn
import torch.nn.functional as F

class DCNN(nn.Module):
    def __init__(self):
        super(DCNN, self).__init__()
        
        # Input: 3 channels (RGB), 224x224
        # First convolution layer with 7x7 kernel
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Second convolution layer with 3x3 kernel
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Additional convolution layers
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Calculate the correct input size for the first fully connected layer
        # After two pooling layers: 224 -> 112 -> 56
        self.fc_input_size = 802816
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)  # Binary classification
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Input shape: (batch_size, 3, 224, 224)
        
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))  # (batch_size, 64, 224, 224)
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))  # (batch_size, 128, 224, 224)
        
        # First pooling
        x = self.pool(x)  # (batch_size, 128, 112, 112)
        
        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))  # (batch_size, 256, 112, 112)
        
        # Fourth conv block
        x = F.relu(self.bn4(self.conv4(x)))  # (batch_size, 512, 112, 112)
        
        # Second pooling
        x = self.pool(x)  # (batch_size, 512, 56, 56)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # (batch_size, 512*56*56)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x




import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=3.0, reduction='mean'):
        '''
        alpha: list or tensor of shape [2]，分别是类别0和类别1的权重
        gamma: 聚焦参数
        '''
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.tensor([0.5, 0.5]).to(device)
        else:
            self.alpha = torch.tensor(alpha).to(device)
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


# %%
#GoogleNet训练函数
def fit(epoch, model, trainloader, testloader):
    lr=[]
    corret = 0
    total = 0
    running_loss = 0

    model.train()

    model = model.to(device)

    for x, y in trainloader:
#         print(x,y)
        y = torch.tensor(y, dtype=torch.long)
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        
        
        y_pred,aux2,aux1 = model(x)

        #y_pred = model(x)

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
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
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


# %%
#DCNN训练函数
def dcnn_fit(epoch, model, trainloader, testloader):
    lr=[]
    corret = 0
    total = 0
    running_loss = 0

    model.train()
    for x, y in trainloader:
#         print(x,y)
        # x =  x.to(device)
        # y =  y.to(device)


        y = torch.tensor(y, dtype=torch.long)
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        
        
        y_pred = model(x)
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


# %%
model=googlenet
#@model = resnet18
# model = DCNN()

model = model.to(device)

#Focal loss
alpha_cancer = 0.394
alpha_non_cancer = 0.606
alpha = [alpha_cancer, alpha_non_cancer]  
gamma = 3.0

criterion = FocalLoss(alpha=alpha, gamma=gamma)
loss_fn=criterion

#0.0001 3.0 200 这个效果比较好了
#0.00005 3.0 200 试一试这个 

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

    #torch.save(model.state_dict(),'20250814_DCNN_152_'+str(epoch)+ '-model_weights.pt')

    y_pred=[]
    y_prob=[]
    y_test=[]
    patient_acc=[]

    path=r"/mnt/petrelfs/yangjiange/all_data/yangjiange/patient202new/patient202-cwt"
    all_imgs_path2=[]                                             
    all_labels2=[]
    for p in val_patient:
        y_pred0=[]
        y_test0=[]
        y_prob0=[]
        all_imgs_path2= glob.glob(path+'/'+str(int(p))+'/*.jpg')
    #     print(all_imgs_path2)
        all_2=patient_label['label#'].iloc[int(p)]
        all_labels2=[]
        for n in range(64):
                all_labels2.append(all_2)
    #     print(len(all_imgs_path2))
        val_ds=Mydatasetpro(all_imgs_path2, all_labels2, transform)
        val_dl = data.DataLoader(val_ds, batch_size=32, shuffle=True,drop_last=True)
    #     print(val_dl.__len__())

        correct = 0
        total = 0
        model.eval()
        for data0 in val_dl:

            images, labels = data0     
            y_test0.append(labels.numpy())

            labels = labels.long()
  
            images =  images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            images =  images.to('cpu')
            labels = labels.to('cpu')
            outputs = outputs.to('cpu')
            
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
    print(y_pred0)
    y_test=np.array(y_test).reshape(1,-1)[0]
    print(y_test0)
    from sklearn.metrics import classification_report

    aaa = classification_report(y_test,y_pred)
    print(aaa)
    bbb = classification_report(y_test,y_pred,output_dict=True)

    weighted_avg = bbb['weighted avg']['precision']
    xxx = bbb['0']['precision']
    yyy = bbb['1']['precision']



    print(np.mean(patient_acc))


    #torch.save(model, '/mnt/petrelfs/yangjiange/lele/results/20250824_GLN_spect_p_f_202_fl_'+str(epoch)+''+ "" +"---".pt')
    torch.save(model, '/mnt/petrelfs/yangjiange/all_data/yangjiange/lele/3.0norm/20250824_GLN_spect_p_f_202_fl_'+str(epoch)+'_'+str(weighted_avg)+'*'+str(xxx)+'*'+str(yyy)+'.pt')
