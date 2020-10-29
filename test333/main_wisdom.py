'''Train  with PyTorch.'''
from __future__ import print_function

import torch.nn as nn

from models import *
import numpy as np
import torch
from net_util import *
import torch.utils.data as Data
import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms

from arg_parser import *
from Adam import *
cudnn.benchmark = True
import os
from functools import partial
os.environ['CUDA_VISIBLE_DEVICES']='0'


if __name__ == '__main__':

    args = parse_args()

    from util import save_path_formatter

    log_dir = save_path_formatter(args)
    args.checkpoint_path = log_dir
    args.result_path = log_dir
    args.log_path = log_dir

    if args.save_plot:
        import matplotlib

        matplotlib.use('agg')
        import matplotlib.pyplot as plt

    if args.deconv:
        args.deconv = partial(FastDeconv, bias=args.bias, eps=args.eps, n_iter=args.deconv_iter, block=args.block,
                              sampling_stride=args.stride)
    else:
        args.deconv = None

    if args.delinear:
        args.channel_deconv = None
        if args.block_fc > 0:
            args.delinear = partial(Delinear, block=args.block_fc, eps=args.eps, n_iter=args.deconv_iter)
        else:
            args.delinear = None
    else:
        args.delinear = None
        if args.block_fc > 0:
            args.channel_deconv = partial(ChannelDeconv, block=args.block_fc, eps=args.eps, n_iter=args.deconv_iter,
                                          sampling_stride=args.stride)
        else:
            args.channel_deconv = None

    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    args.start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
#########################3#################
    if (args.dataset == 'wisdm'):
        args.in_planes = 1
        print("| Preparing wisdm dataset...")
        train_x = np.load('/mnt/ba3b04da-ce1b-4c21-ad1b-3aff7d337cdf/tangyin/project/LegoNet-master/data_uci/wisdm/x_train.npy')
        shape = train_x.shape
        train_x = torch.from_numpy(np.reshape(train_x.astype(np.float), [shape[0], 1, shape[1], shape[2]]))
        train_x = train_x.type(torch.FloatTensor).cuda()
        print(train_x.shape)
        print("-" * 100)

        train_y = (np.load('/mnt/ba3b04da-ce1b-4c21-ad1b-3aff7d337cdf/tangyin/project/LegoNet-master/data_uci/wisdm/y_train.npy'))
        # train_y = np.asarray(pd.get_dummies(train_y))
        train_y = torch.from_numpy(train_y)
        train_y = train_y.type(torch.FloatTensor).cuda()
        print(train_y.shape)
        print("-" * 100)

        # test cifar10
        test_x = np.load('/mnt/ba3b04da-ce1b-4c21-ad1b-3aff7d337cdf/tangyin/project/LegoNet-master/data_uci/wisdm/x_test.npy')
        # print(test_x.shape,'test_x.shapetest_x.shapetest_x.shape')
        test_x = torch.from_numpy(
            np.reshape(test_x.astype(np.float), [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]))
        test_x = test_x.type(torch.FloatTensor).cuda()

        test_y = np.load('/mnt/ba3b04da-ce1b-4c21-ad1b-3aff7d337cdf/tangyin/project/LegoNet-master/data_uci/wisdm/y_test.npy')
        test_y = torch.from_numpy(test_y.astype(np.float32))
        test_y = test_y.type(torch.FloatTensor).cuda()
        # print(train_x.shape, train_y.shape)

        torch_dataset = Data.TensorDataset(train_x, train_y)
        train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=60, shuffle=True, num_workers=0)

        torch_dataset = Data.TensorDataset(test_x, test_y)
        test_loader = Data.DataLoader(dataset=torch_dataset, batch_size=60, shuffle=True, num_workers=0)
        args.num_outputs = 6
############################################

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    args.train_epoch_logger = Logger(os.path.join(args.result_path, 'train.log'),
                                     ['epoch', 'loss', 'top1', 'top5', 'time'])
    args.train_batch_logger = Logger(os.path.join(args.result_path, 'train_batch.log'),
                                     ['epoch', 'batch', 'loss', 'top1', 'top5', 'time'])
    args.test_epoch_logger = Logger(os.path.join(args.result_path, 'test.log'),
                                    ['epoch', 'loss', 'top1', 'top5', 'time'])

    # Model

    print('==> Building model..')

    if args.deconv:
        args.batchnorm = False
        print('************ Batch norm disabled when deconv is used. ************')

    if (not args.deconv) and args.channel_deconv:
        print(
            '************ Channel Deconv is used on the original model, this accelrates the training. If you want to turn it off set --num-groups-final 0 ************')


    if args.arch == 'vgg11':
        net = VGG('VGG11', num_classes=args.num_outputs, deconv=args.deconv, delinear=args.delinear,
                  channel_deconv=args.channel_deconv)


    if args.loss == 'CE':
        args.criterion = nn.CrossEntropyLoss()
        if args.use_gpu:
            args.criterion = nn.CrossEntropyLoss().cuda()

    elif args.loss == 'L2':
        args.criterion = nn.MSELoss()
        if args.use_gpu:
            args.criterion = nn.MSELoss().cuda()


    args.logger_n_iter = 0

    # Training
    print(net)
    print(args)

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in parameters])
    print(params, 'trainable parameters in the network.')

    set_parameters(args)

    lr = args.lr

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    if args.optimizer == 'SGD':
        args.current_optimizer = optim.SGD(parameters, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        # args.current_optimizer = optim.Adam(parameters, lr=lr, weight_decay=args.weight_decay)
        args.current_optimizer = Adam_GC(net.parameters(), lr=0.0001)

    if args.lr_scheduler == 'multistep':

        milestones = [int(args.milestone * args.epochs)]
        while milestones[-1] + milestones[0] < args.epochs:
            milestones.append(milestones[-1] + milestones[0])

        args.current_scheduler = optim.lr_scheduler.MultiStepLR(args.current_optimizer, milestones=milestones,
                                                                gamma=args.multistep_gamma)

    if args.lr_scheduler == 'cosine':
        total_steps = math.ceil(len(train_loader) / args.batch_size) * args.epochs
        args.current_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(args.current_optimizer, total_steps,
                                                                            eta_min=0, last_epoch=-1)

    args.total_steps = math.ceil(len(train_loader) / args.batch_size) * args.epochs
    args.cur_steps = 0

    if args.use_gpu:
        net = torch.nn.DataParallel(net).cuda()
    device = torch.device("cuda")

    plotting_accuracies = []


    if args.resume:
        lr = args.lr
        for param_group in args.current_optimizer.param_groups:
            param_group['lr'] = lr
        if args.lr_scheduler == 'multistep':
            for i in range(args.start_epoch):
                args.current_scheduler.step()
        if args.lr_scheduler == 'cosine':
            total_steps = math.ceil(len(trainset) / args.batch_size) * args.start_epoch
            for i in range(total_steps):
                args.current_scheduler.step()

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        args.epoch = epoch
        if args.lr_scheduler == 'multistep':
            args.current_scheduler.step()
        if args.lr_scheduler == 'multistep' or args.lr_scheduler == 'cosine':
            print('Current learning rate:', args.current_scheduler.get_lr()[0])

        args.data_loader = train_loader
        train_net(net, args)
        args.data_loader = test_loader

        args.validating = False
        args.testing = True

        eval_net(net, args)





    print('Training finished successfully. Model size: ', params, )
    if args.best_acc > 0:
        print('Best acc: ', args.best_acc)


