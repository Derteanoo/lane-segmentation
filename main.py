import os
import argparse
from solver import Solver
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    if config.mode == 'train':
        from data_loader import get_loader
        # Data loader
        data_loader = get_loader(config.train_image_src, config.train_image_tgt,\
                                 config.batch_size, config.mode, config.mask_w, config.mask_h)
        data_loader2 = get_loader(config.val_image_src, config.val_image_tgt,\
                                 config.batch_size, config.mode, config.mask_w, config.mask_h)
        # Solver
        solver = Solver(data_loader, data_loader2, config)
        solver.train()
    elif config.mode == 'test':
        from data_loader3 import get_loader
        # Data loader
        data_loader = get_loader(image_src=config.test_image_src, mode='test', width=config.mask_w, height=config.mask_h)
        # Solver
        solver = Solver(data_loader, data_loader, config)
        solver.test()
    elif config.mode == 'evaluate':
        from  data_loader4 import get_loader
        # Data loader
        data_loader = get_loader(config.eval_image_src, config.eval_image_tgt,\
                                 mode='evaluate', width=config.mask_w, height=config.mask_h)
        # Solver
        solver = Solver(data_loader, data_loader, config)
        solver.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--g_lr', type=float, default=0.001)
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--d_train_repeat', type=int, default=5)

    # Training settings
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--num_epochs_decay', type=int, default=100)
    parser.add_argument('--num_iters', type=int, default=200)
    parser.add_argument('--num_iters_decay', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--beta1', type=float, default=0.9)#0.5
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--pretrained_model', type=str, default='')#166_400 200_800
    parser.add_argument('--loss', type=str, default='nll', choices=['nll', 'focal','dice'])

    # Test settings
    parser.add_argument('--test_model', type=str, default='')#185_500

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'evaluate'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--print_network', type=str2bool, default=False)

    # Path
    parser.add_argument('--train_image_src', type=str, default='/media/data1/zdw/t3/train_image')
    parser.add_argument('--train_image_tgt', type=str, default='/media/data1/zdw/t3/train_local_label')
    parser.add_argument('--val_image_src', type=str, default='/media/data1/zdw/t3/val_image')
    parser.add_argument('--val_image_tgt', type=str, default='/media/data1/zdw/t3/val_local_label')
    parser.add_argument('--test_image_src', type=str, default='/media/data1/zdw/t3/val_image')
    parser.add_argument('--eval_image_src', type=str, default='/media/data1/zdw/t3/val_image')
    parser.add_argument('--eval_image_tgt', type=str, default='/media/data1/zdw/t3/val_local_label')

    parser.add_argument('--log_path', type=str, default='./stargan/logs')
    parser.add_argument('--model_save_path', type=str, default='./stargan/models')
    parser.add_argument('--sample_path', type=str, default='./stargan/samples')
    parser.add_argument('--result_path', type=str, default='./stargan/results')

    # Step size
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--val_log_step', type=int, default=200)    
    parser.add_argument('--sample_step', type=int, default=200)#400
    parser.add_argument('--model_save_step', type=int, default=200)

    #mask size
    parser.add_argument('--mask_w', type=int, default=128)
    parser.add_argument('--mask_h', type=int, default=128)
    parser.add_argument('--class_num', type=int, default=5)

    #network structure
    parser.add_argument('--network', type=str, default='hrnet', 
                       choices=['unet', 'deeplabv3', 'espnetv2', 'enet', 'erfnet', 'hrnet', 'scnn', 
                       'hourglass', 'deeplabv3_plus', 'fast_scnn', 'mobilenetv2', 'mnasnet',
                       'dfanet','lednet'])

    config = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6"
    print(config)
    main(config)
