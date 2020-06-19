import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import functional as F

import joint_transforms
from config import msra10k_path, video_train_path, datasets_root, video_seq_gt_path, video_seq_path
from datasets import ImageFolder, VideoImageFolder, VideoSequenceFolder, VideoImage2Folder
from misc import AvgMeter, check_mkdir, CriterionKL3, CriterionKL, CriterionPairWise
from model_distill import Distill
from torch.backends import cudnn
import time
from utils_mine import load_part_of_model
import random

cudnn.benchmark = True
device_id = 3
torch.manual_seed(2019)
torch.cuda.set_device(device_id)


time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
ckpt_path = './ckpt'
exp_name = 'VideoSaliency' + '_' + time_str

args = {
    'basic_model': 'resnet50',
    'motion': '',
    'se_layer': False,
    'dilation': False,
    'distillation': True,
    'L2': False,
    'KL': False,
    'iter_num': 80000,
    'iter_save': 10000,
    'iter_start_seq': 0,
    'train_batch_size': 8,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.95,
    'snapshot': '',
    'pretrain': os.path.join(ckpt_path, 'VideoSaliency_2019-12-24 22:05:11', '50000.pth'),
    # 'pretrain': '',
    'imgs_file': 'Pre-train/pretrain_all_seq_DUT_TR_DAFB2_DAVSOD2.txt',
    # 'imgs_file': 'video_saliency/train_all_DAFB2_DAVSOD_5f.txt',
    'train_loader': 'both'
    # 'train_loader': 'video_sequence'
}

imgs_file = os.path.join(datasets_root, args['imgs_file'])
# imgs_file = os.path.join(datasets_root, 'video_saliency/train_all_DAFB3_seq_5f.txt')

joint_transform = joint_transforms.Compose([
    joint_transforms.ImageResize(520),
    joint_transforms.RandomCrop(473),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])

# joint_seq_transform = joint_transforms.Compose([
#     joint_transforms.ImageResize(520),
#     joint_transforms.RandomCrop(473)
# ])

input_size = (473, 473)

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

# train_set = ImageFolder(msra10k_path, joint_transform, img_transform, target_transform)
if args['train_loader'] == 'video_sequence':
    train_set = VideoSequenceFolder(video_seq_path, video_seq_gt_path, imgs_file, joint_transform, img_transform, target_transform)
elif args['train_loader'] == 'video_image':
    train_set = VideoImageFolder(video_train_path, imgs_file, joint_transform, img_transform, target_transform)
else:
    train_set = VideoImage2Folder(video_train_path, imgs_file, video_seq_path + '/DAFB2', video_seq_gt_path + '/DAFB2',
                                  joint_transform, None, input_size, img_transform, target_transform)

train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)

criterion = nn.BCEWithLogitsLoss().cuda()
if args['L2']:
    criterion_l2 = nn.MSELoss().cuda()
    # criterion_pair = CriterionPairWise(scale=0.5).cuda()
if args['KL']:
    criterion_kl = CriterionKL3().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')

total_loss_record, loss0_record, loss1_record, loss2_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

def fix_parameters(parameters):
    for name, parameter in parameters:
        if name.find('motion') >= 0 \
                or name.find('GRU') >= 0 or name.find('predict') >= 0:
            print(name, 'is not fixed')

        else:
            print(name, 'is fixed')
            parameter.requires_grad = False


def main():
    net = Distill(basic_model=args['basic_model'], seq=True).cuda().train()

    # fix_parameters(net.named_parameters())
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    if len(args['pretrain']) > 0:
        print('pretrain model from ' + args['pretrain'])
        net = load_part_of_model(net, args['pretrain'], device_id=device_id)

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:

        # loss3_record = AvgMeter()

        for i, data in enumerate(train_loader):

            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, labels, previous_frame, previous_gt, current_frame, current_gt, next_frame, next_gt = data


            if curr_iter < args['iter_start_seq']:
                train_single(net, inputs, labels, criterion, optimizer, curr_iter)

            else:

                if curr_iter % 3 == 0:
                    previous_frame1, previous_frame2 = torch.chunk(previous_frame, 2, 0)
                    current_frame1, current_frame2 = torch.chunk(current_frame, 2, 0)

                    previous_gt1, previous_gt2 = torch.chunk(previous_gt, 2, 0)
                    current_gt1, current_gt2 = torch.chunk(current_gt, 2, 0)
                    if random.uniform(0, 1) > 0.5:
                        train_seq(net, previous_frame1, previous_gt1, current_frame1, current_gt1, optimizer, criterion, curr_iter)
                    else:
                        train_seq(net, previous_frame2, previous_gt2, current_frame2, current_gt2, optimizer, criterion, curr_iter)
                else:
                    # first, second = torch.chunk(inputs, 2, 0)
                    # first_gt, second_gt = torch.chunk(labels, 2, 0)
                    #
                    # train_seq(net, first, first_gt, first, first_gt, optimizer, criterion,
                    #           curr_iter)
                    # train_seq(net, second, second_gt, second, second_gt, optimizer, criterion,
                    #           curr_iter)
                    train_single(net, inputs, labels, criterion, optimizer, curr_iter)

            curr_iter += 1

            if curr_iter % args['iter_save'] == 0:
                print('taking snapshot ...')
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))

            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
                return

def train_single(net, inputs, labels, criterion, optimizer, curr_iter):
    inputs = Variable(inputs).cuda()
    labels = Variable(labels).cuda()

    optimizer.zero_grad()
    outputs0, outputs1, outputs2, feat_high, _ = net(inputs, inputs, flag='single')
    loss0 = criterion(outputs0, labels)
    loss1 = criterion(outputs1, labels)
    loss2 = criterion(outputs2, labels)
    if args['distillation']:
        loss02 = criterion(outputs0, F.sigmoid(outputs2))
        loss12 = criterion(outputs1, F.sigmoid(outputs2))

        total_loss = loss0 + loss1 + loss2 + 0.5 * loss02 + 0.5 * loss12
    else:
        total_loss = loss0 + loss1 + loss2

    total_loss.backward()
    optimizer.step()

    print_log(total_loss, loss0, loss1, loss2, args['train_batch_size'], curr_iter, optimizer)

    return

def train_seq(net, previous_frame, previous_gt, current_frame, current_gt, optimizer, criterion, curr_iter):
    previous_gt = Variable(previous_gt).cuda()
    current_gt = Variable(current_gt).cuda()

    previous_frame = Variable(previous_frame).cuda()
    current_frame = Variable(current_frame).cuda()

    optimizer.zero_grad()

    predict0_pre, predict0_cur, predict1_pre, predict1_cur, predict2_pre, \
    predict2_cur, predict3_pre, predict3_cur = net(previous_frame, current_frame, 'seq')
    loss0_pre = criterion(predict0_pre, previous_gt)
    loss1_pre = criterion(predict1_pre, previous_gt)
    loss2_pre = criterion(predict2_pre, previous_gt)
    loss3_pre = criterion(predict3_pre, previous_gt)

    loss0_pre_cur = criterion(predict0_pre, F.sigmoid(predict2_cur))
    loss1_pre_cur = criterion(predict1_pre, F.sigmoid(predict2_cur))
    loss2_pre_cur = criterion(predict2_pre, F.sigmoid(predict2_cur))
    # loss3_pre_cur = criterion(predict3_pre, F.sigmoid(predict2_cur))

    loss0_cur = criterion(predict0_cur, current_gt)
    loss1_cur = criterion(predict1_cur, current_gt)
    loss2_cur = criterion(predict2_cur, current_gt)
    loss3_cur = criterion(predict2_cur, current_gt)

    loss0_cur_pre = criterion(predict0_cur, F.sigmoid(predict2_pre))
    loss1_cur_pre = criterion(predict1_cur, F.sigmoid(predict2_pre))


    loss0_pre = loss0_pre + 0.3 * loss0_pre_cur
    loss1_pre = loss1_pre + 0.3 * loss1_pre_cur
    loss2_pre = loss2_pre + 0.3 * loss2_pre_cur
    loss3_pre = loss3_pre

    total_loss_pre = loss0_pre + loss1_pre + loss2_pre + loss3_pre

    loss0_cur = loss0_cur + 0.3 * loss0_cur_pre
    loss1_cur = loss1_cur + 0.3 * loss1_cur_pre
    loss2_cur = loss2_cur
    loss3_cur = loss3_cur

    total_loss_cur = loss0_cur + loss1_cur + loss2_cur + loss3_cur

    total_loss = total_loss_pre + total_loss_cur
    total_loss.backward()
    optimizer.step()

    print_log(total_loss, loss1_pre, loss2_pre, loss3_pre, args['train_batch_size'], curr_iter,
              optimizer, 'previous')

    print_log(total_loss, loss1_cur, loss2_cur, loss3_cur, args['train_batch_size'], curr_iter,
              optimizer, 'current')

    return

def print_log(total_loss, loss0, loss1, loss2, batch_size, curr_iter, optimizer, type='normal'):
    total_loss_record.update(total_loss.data, batch_size)
    loss0_record.update(loss0.data, batch_size)
    loss1_record.update(loss1.data, batch_size)
    loss2_record.update(loss2.data, batch_size)
    # loss3_record.update(loss3.data, batch_size)
    # loss4_record.update(loss4.data, batch_size)
    log = '[iter %d][%s], [total loss %.5f], [loss0 %.5f], [loss1 %.5f], [loss2 %.5f] ' \
          '[lr %.13f]' % \
          (curr_iter, type, total_loss_record.avg, loss0_record.avg, loss1_record.avg, loss2_record.avg,
           optimizer.param_groups[1]['lr'])
    print(log)
    open(log_path, 'a').write(log + '\n')


if __name__ == '__main__':
    main()
