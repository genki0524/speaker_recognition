import torch.nn as nn

class Net1D(nn.Module):
    def __init__(self):
        super(Net1D,self).__init__()
         
        self.conv1 = nn.Sequential(nn.Conv1d(1, 8, kernel_size=3, stride=1),
                                   nn.BatchNorm1d(8),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool1d(kernel_size=3, stride=2),
                                  )
         
        self.conv2 = nn.Sequential(nn.Conv1d(8, 32, kernel_size=5, stride=1),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool1d(kernel_size=5, stride=2),
                                  )
         
        self.conv3 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=7, stride=1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool1d(kernel_size=7, stride=2),
                                  )
        
        self.conv4 = nn.Sequential(nn.Conv1d(64,128,kernel_size=9,stride=1),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool1d(kernel_size=9,stride=2),)
                                   
 
        self.dense = nn.Sequential(nn.Linear(128*2988, 512),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.5),
                                   nn.Linear(512,128),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.5),
                                   nn.Linear(128, 3),
                                  )
 
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0),-1)
        x = self.dense(x)
 
        return x
     
    def check_size(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0),-1)
 
        return x


