import torch
import torchvision
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
from torchsummary import summary
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

train_dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
train_datasetloader=torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=0)
test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())
test_datasetloader=torch.utils.data.DataLoader(test_dataset,batch_size=16,shuffle=True,num_workers=0)
classes = train_dataset.classes

# hyperparameters
model = models.vgg16()
batch_size = 256
learning_rate = 0.001
num_epoches = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate)




def hw1_5_1(dataset):
    rand = random.randint(1,len(dataset))
    x=0
    for i in range(9):
        rand = random.randint(1,len(dataset))
        images, labels = dataset[rand]
        plt.subplot(3,3,x+1)
        plt.tight_layout()
        images = images.numpy().transpose(1, 2, 0)  # 把channel那一维放到最后
        plt.title(str(classes[labels]))
        plt.imshow(images)
        plt.xticks([])
        plt.yticks([])
        x+=1
        if x==9:
            break
    plt.show()


def hw1_5_2(batch_size, learning_rate, optimizer):
    print("hyperparameters : ")
    print("batch size : " + f"{batch_size}")
    print("learning rate : " + f"{learning_rate}")
    print("optimizer : " + f"{optimizer}")

# model = models.vgg16()
# summary(model, (3, 32, 32))

def hw1_5_4(model):
    # check if GPU is available
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    for epoch in range(num_epoches):
        print('*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)      # .format为输出格式，formet括号里的即为左边花括号的输出
        running_loss = 0.0
        running_acc = 0.0
        for i, data in tqdm(enumerate(test_datasetloader, 1)):
            
            img, label = data
            # cuda
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            img = Variable(img)
            label = Variable(label)
            # 向前传播
            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)     # 预测最大值所在的位置标签
            num_correct = (pred == label).sum()
            accuracy = (pred == label).float().mean()
            running_acc += num_correct.item()
            # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))))
        
        model.eval()    # 模型评估
        eval_loss = 0
        eval_acc = 0
        for data in test_datasetloader:      # 测试模型
            img, label = data
            if use_gpu:
                img = Variable(img, volatile=True).cuda()
                label = Variable(label, volatile=True).cuda()
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)
            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            test_dataset)), eval_acc / (len(test_dataset))))
        print()

    # 保存模型
    torch.save(model.state_dict(), './cnn.pth')