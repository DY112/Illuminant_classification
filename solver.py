import torch
import torch.nn as nn
from torch import optim
from torchvision.models import resnet18

class Solver(nn.Module):
    def __init__(self,config,train_loader,valid_loader,test_loader) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        

        self.build_model(config)

    def build_model(self,config):
        # Build network
        self.net = resnet18(pretrained=True)
        self.net.fc = nn.Linear(512,config.num_classes)
        
        # Multi-GPU setting
        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
        self.net.to(self.device)

        # Configure optimizer,scheduler
        # without checkpoint
        if self.config.checkpoint == None:
            self.optimizer = optim.Adam(params=self.net.parameters(),lr=config.lr)
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                         lr_lambda=lambda epoch:config.decay_lambda**epoch,
                                                         last_epoch=-1,
                                                         verbose=False)
        else:   # load from checkpoint
            # TODO
            raise NotImplementedError()
            self.net.load_state_dict()
            self.optimizer = None
            self.scheduler = None

        print('[Model]\tBuild complete.')

    def train(self):
        print("[Train]\tStart training process.")
        best_loss = 9876543210.
        best_mae_illum = 9876543210.
        best_psnr = 0.
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch+1
            self.net.train(True)
            minibatch_count = len(self.train_loader)

            # Train
            for i, batch in enumerate(self.train_loader):
                ret_dict = self.inference(batch,'train')
                self.logger(mode='train',
                            ret_dict=ret_dict,
                            writer_itr=epoch*minibatch_count+i,
                            epoch=epoch+1,
                            num_epochs=self.num_epochs,
                            batch=i+1,
                            num_batchs=minibatch_count)

            # Validation
            if self.epoch % self.val_step == 0:
                self.net.eval()
                accumulate_dict = {}
                valid_data_count = 0

                for i, batch in enumerate(self.valid_loader):
                    ret_dict = self.inference(batch,'valid')
                    
                    minibatch_len = len(batch['input_rgb'])
                    if not accumulate_dict:    # initial empty dict
                        for key,value in ret_dict.items():
                            accumulate_dict[key] = value * minibatch_len
                    else:                           # accumulate valid metric
                        for key,value in ret_dict.items():
                            accumulate_dict[key] = accumulate_dict[key] + value * minibatch_len
                    valid_data_count += minibatch_len

                for key in accumulate_dict.keys():
                    accumulate_dict[key] /= valid_data_count
                self.logger(mode='valid',
                            ret_dict=accumulate_dict,
                            writer_itr=epoch+1,
                            epoch=epoch+1,
                            num_epochs=self.num_epochs)

                # Save best model
                if accumulate_dict['total_loss'] < best_loss:
                    best_loss = accumulate_dict['total_loss']
                    best_net = self.net.module.state_dict()
                    print(f'Best net Score : {best_loss:.4f}')
                    torch.save(best_net, os.path.join(self.model_path, 'best_loss.pt'))
                if accumulate_dict['MAE_illum'] < best_mae_illum:
                    best_mae_illum = accumulate_dict['MAE_illum']
                    best_net = self.net.module.state_dict()
                    print(f'Best MAE_illum : {best_mae_illum:.4f}')
                    torch.save(best_net, os.path.join(self.model_path, 'best_mae_illum.pt'))
                if accumulate_dict['PSNR'] > best_psnr:
                    best_psnr = accumulate_dict['PSNR']
                    best_net = self.net.module.state_dict()
                    print(f'Best PSNR : {best_psnr:.4f}')
                    torch.save(best_net, os.path.join(self.model_path, 'best_psnr.pt'))
            
            # Save every N epoch
            if self.save_epoch > 0 and epoch % self.save_epoch == self.save_epoch-1:
                state_dict = self.net.module.state_dict()
                torch.save(state_dict, os.path.join(self.model_path, str(epoch)+'.pt'))

            # lr decay
            if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                self.lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                print(f'Decay lr to {self.lr}')

    def test(self):
        pass