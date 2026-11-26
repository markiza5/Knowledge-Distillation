import torch    
import torch.nn as nn

class SmallCNN(nn.Module):

    def __init__(self,num_classes=8):
        super().__init__()

        self.conv1= nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,padding=1, stride=1)
        self.bn1= nn.BatchNorm2d(num_features=16)
        self.relu1= nn.ReLU()
        self.Pool1=nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2= nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1, stride=1)
        self.bn2= nn.BatchNorm2d(num_features=32)
        self.relu2= nn.ReLU()
        self.Pool2=nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3= nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1, stride=1)
        self.bn3= nn.BatchNorm2d(num_features=64)
        self.relu3= nn.ReLU()
        self.Pool3=nn.MaxPool2d(kernel_size=2,stride=2)

        
        self.conv4= nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1, stride=1)
        self.bn4= nn.BatchNorm2d(num_features=128)
        self.relu4= nn.ReLU()
        self.Pool4=nn.MaxPool2d(kernel_size=2,stride=2)

        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten= nn.Flatten(start_dim=1)
        self.fc= nn.Linear(in_features=128, out_features=num_classes)

        

    def forward(self, x):
        x = self.conv1(x)
        x= self.bn1(x)
        x= self.relu1(x)
        x= self.Pool1(x)

        x = self.conv2(x)
        x= self.bn2(x)
        x= self.relu2(x)
        x= self.Pool2(x)

        x = self.conv3(x)
        x= self.bn3(x)
        x= self.relu3(x)
        x= self.Pool3(x)
        
        x = self.conv4(x)
        x= self.bn4(x)
        x= self.relu4(x)
        x= self.Pool4(x)

        x=self.avgpool(x)
        x= self.flatten(x)
        x= self.fc(x)

        return x

model= SmallCNN()
model.eval()
x= torch.randn(2, 3, 224, 224)
output = model(x)
print(output)



