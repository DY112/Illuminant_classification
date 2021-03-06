import torch,time,os,json
import torch.nn as nn
from torch import optim
from torchvision.models import resnet34
from torch.utils.tensorboard import SummaryWriter

def lr_decay(epoch):
    if epoch < 700:
        return 1
    elif epoch < 1400:
        return 0.5
    elif epoch < 2100:
        return 0.25
    else:
        return 0.1

class Solver(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config

        # initialize path
        if config.checkpoint == None:
            self.train_date = time.strftime("%y%m%d_%H%M", time.localtime(time.time()))
        else:
            self.train_date = config.checkpoint.split('/')[0]
        self.result_path = os.path.join('results',self.train_date+'_'+config.mode)
        self.model_path = os.path.join('models',self.train_date)
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.model_path,exist_ok=True)
        if config.mode == 'train':
            self.writer = SummaryWriter(os.path.join('logs',self.train_date))
            with open(os.path.join(self.model_path,'args.txt'), 'w') as f:
                json.dump(config.__dict__, f, indent=2)
            f.close()

        self.build_model(config)

    def build_model(self,config):
        # Build network
        self.net = resnet34(pretrained=config.pretrained=='yes')
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
            ckpt = os.path.join(self.model_path,'best.pt')
            self.net.load_state_dict(torch.load(ckpt))
            # TODO : train continue from checkpoint
            self.optimizer = None
            self.scheduler = None

        # set loss criterion
        self.criterion = nn.CrossEntropyLoss()

        print('[Model]\tBuild complete.')

    def train(self,loader,config):
        print("[Train]\tStart training process.")
        best_loss = 987654321.

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

                # calculate epoch loss, acc
                epoch_loss = running_loss / len(loader[phase])
                epoch_acc = running_corrects.double() / len(loader[phase])
                print(f'[{phase}] [{epoch+1}/{config.num_epochs}] | Loss : {epoch_loss:.4f} | Acc : {epoch_acc:.4f}')
                self.writer.add_scalar(f'{phase}/Loss',epoch_loss,epoch)
                self.writer.add_scalar(f'{phase}/Acc',epoch_acc,epoch)

                # save best model
                if phase == 'valid' and best_loss > epoch_loss:
                    best_net = self.net.module.state_dict()
                    torch.save(best_net,os.path.join(self.model_path,'best.pt'))
            

    def test(self,loader,config):
        print('[Test] \tStart testing process.')

        running_corrects = 0
        for inputs in loader['test']:
            if config.input_type == 'uvl':
                imgs = inputs['input_uvl'].to(self.device)
            else:
                imgs = inputs['input_rgb'].to(self.device)
            valid_idx = torch.tensor([int(i)-1 for i in inputs['illum_count']]).unsqueeze(-1)
            gt_classes = torch.gather(inputs['illum_class'],1,valid_idx).squeeze().to(self.device)
            
            outputs = self.net(imgs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == gt_classes)

            print(f'GT : {gt_classes}, Pred : {preds.item()}')
        acc = running_corrects.double() / len(loader['test'])
        print(f'ACC : {acc}')