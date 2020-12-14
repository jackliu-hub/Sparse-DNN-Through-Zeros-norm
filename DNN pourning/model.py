import torch.nn as nn
import torch.nn.functional as F




class L_softmax(nn.Module):
    def __init__(self):
        super(L_softmax, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.linear1(out)
        return F.log_softmax(out, dim=1)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 200)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(200, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.relu1(self.linear1(out))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)
        return F.log_softmax(out, dim=1)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

class VGG(nn.Module):
    # 定义模型
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # 第一层
            # 取卷积核为3*3，补边为1，输入通道为3，输出通道为64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            # 批标准化
            # nn.BatchNorm2d(64, affine=True),
            # 激活函数
            nn.ReLU(inplace=True),
            # 池化，核为2*2，步长为2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二层
            # 取卷积核为3*3，补边为1，输入通道为64，输出通道为128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三层
            # 取卷积核为3*3，补边为1，输入通道为128，输出通道为256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            # 取卷积核为3*3，补边为1，输入通道为256，输出通道为256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第四层
            # 取卷积核为3*3，补边为1，输入通道为256，输出通道为512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            # 取卷积核为3*3，补边为1，输入通道为512，输出通道为512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第五层
            # 取卷积核为3*3，补边为1，输入通道为512，输出通道为512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            # 取卷积核为3*3，补边为1，输入通道为512，输出通道为512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            # 一层全连接层，输入512层，输出10（10类）
            nn.Linear(512, num_classes),
        )
    #前向计算
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


