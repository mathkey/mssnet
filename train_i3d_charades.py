import argparse
import os
import engine
from torch import nn
import torch
from dataset import TSNDataSet
import datasets_video
import torchvision
from transforms import *
import model_zoo.gcn_i3d as modelfile
import time


parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save_model_path', default='checkpoint/coco/', type=str,
                    help='the path of saved models')
parser.add_argument('--log_path', default='./log/', type=str,
                    help='the path of log files')
parser.add_argument('--data_length', default=1, type=int, help='the length of a segment')
parser.add_argument('--num_segments', default=3, type=int, help='the number of segments')


def get_config_optim(model, lr, weight_decay):

    params_dict = dict(model.named_parameters())
    params = []

    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key or 'adj' == key else 1.0

        if key.startswith('conv1') or key.startswith('bn1'):
            lr_mult = 0.1
        elif 'fc' in key:
            lr_mult = 1.0
        elif 'adj' == key:
            lr_mult = 0.0
        elif 'gc' in key:
            lr_mult = 1.0
        else:
            lr_mult = 0.1

        params.append({'params': value,
                    'lr': lr,
                    'lr_mult': lr_mult,
                    'weight_decay': weight_decay,
                    'decay_mult': decay_mult})

    return params

def get_optim_fix_conv(model, lr, weight_decay):

    params_dict = dict(model.named_parameters())
    params = []

    for key, value in params_dict.items():
        if 'gc' in key:
            decay_mult = 0.0 if 'bias' in key or 'adj' == key else 1.0

            if key.startswith('conv1') or key.startswith('bn1'):
                lr_mult = 0.1
            elif 'fc' in key:
                lr_mult = 1.0
            elif 'A' == key:
                lr_mult = 0.0
            elif 'gc' in key:
                lr_mult = 1.0
            else:
                lr_mult = 0.1
        else:
            decay_mult = 0.0
            lr_mult = 0.0

        params.append({'params': value,
                    'lr': lr,
                    'lr_mult': lr_mult,
                    'weight_decay': weight_decay,
                    'decay_mult': decay_mult})

    return params

def load_pretrained(model, pretrained_path):
    pretrained = torch.load(pretrained_path)
    total_num = len(pretrained)
    model_state = model.state_dict()
    load_parameters = {key: value for key, value in pretrained.items()
        if key in model_state and value.shape == model_state[key].shape}
    load_num = len(load_parameters)
    model_state.update(load_parameters)
    model.load_state_dict(model_state)
    print('pretrained model:{}'.format(pretrained_path))
    print('loaded:{}({})\ttotal:{}'.format(load_num, float(load_num) / total_num, total_num))

def main_charades():
    global args, best_prec1, use_gpu
    
    use_gpu = torch.cuda.is_available()

    categories, args.train_list, args.val_list, args.train_num_list, args.val_num_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.modality, args.root_path)
    num_class = len(categories)
    crop_size = args.crop_size
    scale_size = args.scale_size
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]

    train_dataset = TSNDataSet(args.root_path, args.train_list, args.train_num_list,
                            num_class=num_class,
                            num_segments=args.num_segments,
                            new_length=args.data_length,
                            modality=args.modality,
                            image_tmpl=prefix,
                            transform=torchvision.transforms.Compose([
                                GroupMultiScaleCrop(crop_size, [1.0, 0.875, 0.75, 0.66, 0.5], max_distort=2),
                                GroupRandomHorizontalFlip(is_flow=False),
                                Stack(roll=False),
                                ToTorchFormatTensor(div=True),
                                GroupNormalize(input_mean, input_std),
                                ChangeToCTHW(modality=args.modality)
                            ]))

    val_dataset = TSNDataSet(args.root_path, args.val_list, args.val_num_list,
                            num_class=num_class,
                            num_segments=args.num_segments,
                            new_length=args.data_length,
                            modality=args.modality,
                            image_tmpl=prefix,
                            random_shift=False,
                            transform=torchvision.transforms.Compose([
                                GroupScale(int(scale_size)),
                                GroupCenterCrop(crop_size),
                                Stack(roll=False),
                                ToTorchFormatTensor(div=True),
                                GroupNormalize(input_mean, input_std),
                                ChangeToCTHW(modality=args.modality)
                            ]))

    # model = modelfile.InceptionI3d(num_class, in_channels=3)
    model = modelfile.gcn_i3d(num_class=num_class, t=0.4, adj_file='./data/Charades_v1/gcn_info/class_graph_conceptnet_context_0.8.pkl', word_file='./data/Charades_v1/gcn_info/class_word.pkl')
    
    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()

    # define optimizer
    params = get_config_optim(model, lr=args.lr, weight_decay=args.weight_decay)
    # params = get_optim_fix_conv(model, lr=args.lr, weight_decay=args.weight_decay)

    # optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer = torch.optim.Adam(params, eps=1e-8)

    state = {'batch_size': args.batch_size,
            'val_batch_size': args.val_batch_size,
            'image_size': args.image_size,
            'max_epochs': args.epochs,
             'evaluate': args.evaluate,
             'resume': args.resume,
             'num_classes':num_class}
    state['difficult_examples'] = False
    state['print_freq'] = args.print_freq
    state['save_model_path'] = args.save_model_path
    state['log_path'] = args.log_path
    state['logname'] = args.logname
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    state['device_ids'] = list(range(torch.cuda.device_count()))
    if args.evaluate:
        state['evaluate'] = True
    mapengine = engine.GCNMultiLabelMAPEngine(state, inp_file='./data/Charades_v1/gcn_info/class_word.pkl')
    mapengine.learning(model, criterion, train_dataset, val_dataset, optimizer)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    gpu_num = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    args = parser.parse_args()
    args.lr = 0.001
    args.weight_decay = 1e-4
    args.momentum = 0.9
    args.workers = 2 * gpu_num
    args.epoch_step = [400, 500, 1000, 2000]
    args.epochs = 10000
    args.print_freq = 5
    args.batch_size = 16
    args.val_batch_size = 8
    args.crop_size = 224
    args.scale_size = 256
    args.start_epoch = 0
    args.num_segments = 64
    args.data_length = 1
    args.evaluate = False   # Change to True if you want to evaluate a checkpoint.
    args.dataset = 'charades'
    args.modality = 'RGB'
    args.root_path = './data/Charades_v1/Charades_v1_rgb'
    args.resume = './checkpoint/charades/model_best.pth.tar'
    args.save_model_path = './checkpoint/charades/'
    args.log_path = './log/'
    args.logname = 'train_charades_gcn_i3d.log'
    
    main_charades()
