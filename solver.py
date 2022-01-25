import torch
import torch.nn as nn
from torch import optim
from torchvision.models import resnet18

def lr_decay(epoch):
    if epoch < 500:
        return 1
    elif epoch < 800:
        return 0.5
    elif epoch < 1200:
        return 0.1
    else:
        return 0.05

class Solver(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config
        self.build_model(config)

    def build_model(self,config):
        # Build network
        self.net = resnet18(pretrained=True)
        self.net.fc = nn.Linear(512,config.num_classes)
        
        # Device setting
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
        self.net.to(self.device)

        # Configure optimizer,scheduler
        if self.config.checkpoint == None:  # start from scratch
            self.optimizer = optim.Adam(params=self.net.parameters(),lr=config.lr, betas=[config.beta1,config.beta2])
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                         lr_lambda=lr_decay,
                                                         last_epoch=-1,
                                                         verbose=False)
        else:   # load from checkpoint
            # TODO
            raise NotImplementedError()
            self.net.load_state_dict()
            self.optimizer = None
            self.scheduler = None

        self.criterion = nn.CrossEntropyLoss()

        print('[Model]\tBuild complete.')

    def train(self,loader,config):
        print("[Train]\tStart training process.")

        for epoch in range(config.num_epochs):
            for phase in ['train','valid']:
                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()
                running_loss = 0.0
                running_corrects = 0

                for inputs in loader[phase]:
                    if config.input_type == 'uvl':
                        imgs = inputs['input_uvl'].to(self.device)
                    else:
                        imgs = inputs['input_rgb'].to(self.device)
                    valid_idx = torch.tensor([int(i)-1 for i in inputs['illum_count']]).unsqueeze(-1)
                    gt_classes = torch.gather(inputs['illum_class'],1,valid_idx).squeeze().to(self.device)
                    
                    outputs = self.net(imgs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs,gt_classes)

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                    
                    running_loss += loss.item() * imgs.size(0)
                    running_corrects += torch.sum(preds == gt_classes)
                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / len(loader[phase])
                epoch_acc = running_corrects.double() / len(loader[phase])
                print(f'[{phase}] [{epoch+1}/{config.num_epochs}] | Loss : {epoch_loss:.4f} | Acc : {epoch_acc:.4f}')
            

    def test(self,test_loader,config):
        pass