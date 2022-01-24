from argparse import ArgumentParser
from solver import Solver
from dataloader_v4 import get_loader

def main(config):

    train_loader = None
    valid_loader = None
    test_loader = None

    if config.mode == 'train':	
        train_loader = get_loader(config,'train')
        valid_loader = get_loader(config,'val')
    elif config.mode == 'test':
        if config.checkpoint == None:
            print("[ERROR]\tCheckpoint required in test mode!")
            exit()
        test_loader = get_loader(config,'test')

    solver = Solver(config, train_loader, valid_loader, test_loader)
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

if __name__ == '__main__':
    parser = ArgumentParser()
    
    # Model
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=20)

    # Training
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--uv_coeff', type=float, default=1)

    # Optimizer & Scheduler
    parser.add_argument('--decay_lambda', type=int, default=0.95)

    # Data & Dataloader
    parser.add_argument('--data_root', type=str, default='../data/galaxy_512')
    parser.add_argument('--camera', type=str, default='galaxy')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--image_pool', type=int, nargs='+', default=[1])
    parser.add_argument('--input_type', type=str, default='uvl', choices=['rgb','uvl'])
    parser.add_argument('--uncalculable', type=int, default=-1)
    parser.add_argument('--mask_uncalculable', type=int, default=None)
    parser.add_argument('--mask_highlight', type=int, default=None)
    parser.add_argument('--mask_black', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=10)

    # Data augmentation config
    parser.add_argument('--random_crop', type=str, default='no', choices=['yes','no'])
    parser.add_argument('--illum_augmentation', type=str, default='no', choices=['yes','no'])
    parser.add_argument('--sat_min', type=float, default=0.2)
    parser.add_argument('--sat_max', type=float, default=0.8)
    parser.add_argument('--val_min', type=float, default=1.0)
    parser.add_argument('--val_max', type=float, default=1.0)
    parser.add_argument('--hue_threshold', type=float, default=0.2)

    config = parser.parse_args()
    main(config)