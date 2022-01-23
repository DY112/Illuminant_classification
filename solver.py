import torch.nn as nn
from torchvision.models import resnet18

class Solver(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config
        
        # initialize network and modify FC layer
        self.net = resnet18(pretrained=True)
        self.net.fc = nn.Linear(512,config.num_classes)
        

    def train(self):
        pass

    def test(self):
        pass