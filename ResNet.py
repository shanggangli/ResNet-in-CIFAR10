import torch
from torchvision import transforms, datasets
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


class Res_block(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(Res_block, self).__init__()
        self.conv=nn.Sequential(       #Black architecture
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut=nn.Sequential()  # shortcut
        if in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels))


    def forward(self,x):
        out=self.conv(x)
        residual=self.shortcut(x)
        out+=residual
        return out

class ResNet(nn.Module):
    def __init__(self,num_classes=10):
        super(ResNet, self).__init__()
        self.pre=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        #(b,64,32,32)->(b,512,1,1)
        self.layer1=self.make_layer(in_channels=64,out_channels=64,block_num=3)
        self.layer2 = self.make_layer(64, 128, 4, stride=2)
        self.layer3 = self.make_layer(128, 256, 6, stride=2)
        self.layer4 = self.make_layer(256,512,3, stride=2)

        self.fc=nn.Linear(512,num_classes)                  #full conntion

    def make_layer(self,in_channels,out_channels,block_num,stride=1):   

        layers=[]
        layers.append(Res_block(in_channels,out_channels,stride))

        for i in range(1,block_num):
            layers.append(Res_block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.pre(x)     

        x=self.layer1(x)
        x=self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x,1) ####
        x = x.view(x.size(0), -1)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #gpu
resnet=ResNet().to(device)

# test model
# print(resnet)
# input= torch.autograd.Variable(torch.randn(64,3,32,32))
# o=resnet(input)
# print(o.size())


# load data
train_data=datasets.CIFAR10(root='./cifar10',train=True,
                            transform=transforms.Compose([transforms.Resize(32,32),transforms.ToTensor(),
                                                          transforms.Normalize(mean=[0.485,0.456,0.406],        #Normalize data easy to train
                                                                               std=[0.229,0.224,0.225])]),
                            download=False)     # if you haven't data need to download=True

test_data=datasets.CIFAR10(root='./cifar10',train=False,
                            transform=transforms.Compose([transforms.Resize(32,32),transforms.ToTensor(),
                                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                               std=[0.229, 0.224, 0.225])
                                                          ]),
                            download=False)
train_loader=DataLoader(dataset=train_data,batch_size=64,shuffle=True)          #make batch
test_loader=DataLoader(dataset=test_data,batch_size=10000,shuffle=True)            #make batch


Lr=0.01     #learning rate
Epoch=10     
optimtier=torch.optim.Adam(resnet.parameters(),lr=Lr)
Loss_func=nn.CrossEntropyLoss()

for epoch in range(Epoch):
    for step,(input,label) in enumerate(train_loader):
        input=input.to(device,dtype=torch.float)    # gpu
        label=label.to(device,dtype=torch.long)     # gpu
        #print(input.shape)
        output=resnet(input)
        #print(output.shape)
        #print(label.shape)
        loss=Loss_func(output,label)
        optimtier.zero_grad()
        loss.backward()
        optimtier.step()

        #test
        if step%100==0:
          total_ccorect=0
          total_num=0
          for i,(input_test,label_test) in enumerate(test_loader):
              input_test=input_test.to(device,dtype=torch.float)
              label_test=label_test.to(device,dtype=torch.long)
              #print(input_test.shape)
              output_test=resnet(input_test)
              #print(label_test.shape)
              pred=output_test.argmax(dim=1)
              #print(pred.shape)
              total_ccorect+=torch.eq(pred,label_test).float().sum().item()
              total_num+=label_test.size(0)
          acc=total_ccorect/total_num
          print('epoch:',epoch,'step:',step,'| test accuracy: %.2f' % acc)

