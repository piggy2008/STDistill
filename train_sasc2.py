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
from model import R3Net
from torch.backends import cudnn
import time
from utils_mine import load_part_of_model

cudnn.benchmark = True
device_id = 2
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
    'iter_num': 40000,
    'iter_save': 10000,
    'iter_start_seq': 0,
    'train_batch_size': 6,
    'last_iter': 0,
    'lr': 1e-4,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'pretrain': os.path.join(ckpt_path, 'VideoSaliency_2019-12-23 22:26:30', '60000.pth'),
    # 'pretrain': '',
    'imgs_file': 'Pre-train/pretrain_all_seq_DUT_TR_DAFB2_DAVSOD.txt',
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
    train_set = VideoImage2Folder(video_train_path, imgs_file, video_seq_path + '/DAFB2_DAVSOD', video_seq_gt_path + '/DAFB2_DAVSOD',
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
    net = R3Net(motion=args['motion'],
                se_layer=args['se_layer'],
                dilation=args['dilation'], basic_model=args['basic_model']).cuda().train()

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
                    train_seq(net, previous_frame, previous_gt, current_frame, current_gt, next_frame, next_gt,
                               optimizer, criterion, curr_iter)
                else:
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

def train_seq(net, previous_frame, previous_gt, current_frame, current_gt, next_frame, next_gt, optimizer, criterion, curr_iter):
    previous_gt = Variable(previous_gt).cuda()
    current_gt = Variable(current_gt).cuda()
    next_gt = Variable(next_gt).cuda()

    # outputs0_pre_tmp = torch.zeros(previous_gt.size())
    # outputs1_pre_tmp = torch.zeros(previous_gt.size())
    outputs2_pre_tmp = torch.zeros(previous_gt.size())


    optimizer.zero_grad()

    previous_frame = Variable(previous_frame).cuda()
    outputs0_pre, outputs1_pre, outputs2_pre, feat_high_pre, feat_low_pre = net(previous_frame)

    feat_high_pre_tmp = torch.zeros(feat_high_pre.size())
    # feat_low_pre_tmp = torch.zeros(feat_low_pre.size())

    # outputs0_pre_tmp.copy_(outputs0_pre)
    # outputs1_pre_tmp.copy_(outputs1_pre)
    outputs2_pre_tmp.copy_(outputs0_pre)
    feat_high_pre_tmp.copy_(feat_high_pre)
    # feat_low_pre_tmp.copy_(feat_low_pre)

    # outputs0_pre_tmp = Variable(outputs0_pre_tmp).cuda()
    # outputs1_pre_tmp = Variable(outputs1_pre_tmp).cuda()
    outputs2_pre_tmp = Variable(outputs2_pre_tmp).cuda()
    feat_high_pre_tmp = Variable(feat_high_pre_tmp).cuda()
    # feat_low_pre_tmp = Variable(feat_low_pre_tmp).cuda()

    # loss0_pre = criterion(outputs0_pre, previous_gt)
    # loss1_pre = criterion(outputs1_pre, previous_gt)
    # loss2_pre = criterion(outputs2_pre, previous_gt)
    # total_loss_pre = loss0_pre + loss1_pre + loss2_pre
    # total_loss_pre.backward()
    # optimizer.step()

    # print_log(total_loss_pre, loss0_pre, loss1_pre, loss2_pre, args['train_batch_size'], curr_iter, optimizer,
    #           'previous')

    # outputs0_next_tmp = torch.zeros(next_gt.size())
    # outputs1_next_tmp = torch.zeros(next_gt.size())
    # outputs2_next_tmp = torch.zeros(next_gt.size())

    # optimizer.zero_grad()
    # next_frame = Variable(next_frame).cuda()
    # outputs0_next, outputs1_next, outputs2_next = net(next_frame)
    #
    # outputs0_next_tmp.copy_(outputs0_next)
    # outputs1_next_tmp.copy_(outputs1_next)
    # outputs2_next_tmp.copy_(outputs0_next)
    #
    # outputs0_next_tmp = Variable(outputs0_next_tmp).cuda()
    # outputs1_next_tmp = Variable(outputs1_next_tmp).cuda()
    # outputs2_next_tmp = Variable(outputs2_next_tmp).cuda()
    #
    # loss0_next = criterion(outputs0_next, next_gt)
    # loss1_next = criterion(outputs1_next, next_gt)
    # loss2_next = criterion(outputs2_next, next_gt)
    # total_loss_next = loss0_next + loss1_next + loss2_next
    # total_loss_next.backward()
    # optimizer.step()
    #
    # print_log(total_loss_next, loss0_next, loss1_next, loss2_next, args['train_batch_size'], curr_iter,
    #           optimizer, 'next')

    optimizer.zero_grad()
    current_frame = Variable(current_frame).cuda()
    outputs0_cur, outputs1_cur, outputs2_cur, feat_high_cur, feat_low_cur = net(current_frame)
    loss0_cur = criterion(outputs0_cur, current_gt)
    loss1_cur = criterion(outputs1_cur, current_gt)
    loss2_cur = criterion(outputs2_cur, current_gt)


    loss0_cur_loc = criterion(outputs0_cur, F.sigmoid(outputs2_cur))
    loss1_cur_loc = criterion(outputs1_cur, F.sigmoid(outputs2_cur))

    loss0_cur_pre = criterion(outputs0_cur, F.sigmoid(outputs2_pre_tmp + outputs2_cur))
    loss1_cur_pre = criterion(outputs1_cur, F.sigmoid(outputs2_pre_tmp + outputs2_cur))
    loss2_cur_pre = criterion(outputs2_cur, F.sigmoid(outputs2_pre_tmp + outputs2_cur))

    if args['KL']:
        loss0_cur_pre_kl = criterion_kl(F.sigmoid(outputs0_cur), F.sigmoid(outputs2_pre_tmp))
        loss1_cur_pre_kl = criterion_kl(F.sigmoid(outputs1_cur), F.sigmoid(outputs2_pre_tmp))
        loss2_cur_pre_kl = criterion_kl(F.sigmoid(outputs2_cur), F.sigmoid(outputs2_pre_tmp))

    if args['L2']:
        loss_feat_high = criterion_l2(feat_high_cur, feat_high_pre_tmp)
        # loss_feat_high2 = criterion_kl(feat_high_cur, feat_high_pre_tmp)
        # loss_pair = criterion_pair(feat_high_pre_tmp, feat_high_cur)

        # loss_feat_low = criterion_l2(feat_low_cur, feat_high_pre_tmp)
        # loss_pair_low = criterion_pair(feat_low_pre_tmp, feat_low_cur)

        # loss_feat = loss_feat_high + loss_feat_low
        # print('[iter %d], [loss_feat %.5f]' % (curr_iter, loss_feat))

    # loss0_cur_next = criterion(outputs0_cur, F.sigmoid(outputs0_next_tmp))
    # loss1_cur_next = criterion(outputs1_cur, F.sigmoid(outputs1_next_tmp))
    # loss2_cur_next = criterion(outputs2_cur, F.sigmoid(outputs2_next_tmp))
    if args['KL']:
        loss0_cur = loss0_cur + 0.3 * loss0_cur_pre + 0.5 * loss0_cur_loc + 0.1 * loss0_cur_pre_kl
        loss1_cur = loss1_cur + 0.3 * loss1_cur_pre + 0.5 * loss1_cur_loc + 0.1 * loss1_cur_pre_kl
        loss2_cur = loss2_cur + 0.3 * loss2_cur_pre + 0.1 * loss2_cur_pre_kl
    else:
        loss0_cur = loss0_cur + 0.5 * loss0_cur_pre
        loss1_cur = loss1_cur + 0.5 * loss1_cur_pre
        loss2_cur = loss2_cur + 0.5 * loss2_cur_pre
    if args['L2']:
        total_loss_cur = loss0_cur + loss1_cur + loss2_cur + 10 * loss_feat_high
    else:
        total_loss_cur = loss0_cur + loss1_cur + loss2_cur
    total_loss_cur.backward()
    optimizer.step()

    print_log(total_loss_cur, loss0_cur, loss1_cur, loss2_cur, args['train_batch_size'], curr_iter,
              optimizer, 'current')

    return

def train_seq2(net, previous_frame, previous_gt, current_frame, current_gt, next_frame, next_gt, optimizer, criterion, curr_iter):
    previous_frame_list = previous_frame.split(2)
    previous_gt_list = previous_gt.split(2)
    current_frame_list = current_frame.split(2)
    current_gt_list = current_gt.split(2)
    next_frame_list = next_frame.split(2)
    next_gt_list = next_gt.split(2)

    for index in range(len(previous_frame_list)):
        batch_frame = torch.cat([previous_frame_list[index], current_frame_list[index], next_frame_list[index]], dim=0)
        batch_gt = torch.cat([previous_gt_list[index], current_gt_list[index], next_gt_list[index]], dim=0)

        batch_frame = Variable(batch_frame).cuda()
        batch_gt = Variable(batch_gt).cuda()
        outputs0, outputs1, outputs2 = net(batch_frame)

        # outputs2_tmp = F.sigmoid(outputs2).data.cpu().numpy()
        # from matplotlib import pyplot as plt
        #
        # plt.subplot(2, 3, 1)
        # plt.imshow(outputs2_tmp[0, 0, :, :])
        # plt.subplot(2, 3, 2)
        # plt.imshow(outputs2_tmp[1, 0, :, :])
        # plt.subplot(2, 3, 3)
        # plt.imshow(outputs2_tmp[2, 0, :, :])
        # plt.subplot(2, 3, 4)
        # plt.imshow(outputs2_tmp[3, 0, :, :])
        # plt.subplot(2, 3, 5)
        # plt.imshow(outputs2_tmp[4, 0, :, :])
        # plt.subplot(2, 3, 6)
        # plt.imshow(outputs2_tmp[5, 0, :, :])
        # plt.show()

        loss0 = criterion(outputs0.narrow(0, 0, 2), batch_gt.narrow(0, 0, 2))
        loss1 = criterion(outputs1.narrow(0, 0, 2), batch_gt.narrow(0, 0, 2))
        loss2 = criterion(outputs2.narrow(0, 0, 2), batch_gt.narrow(0, 0, 2))

        # total_loss = loss0 + loss1 + loss2
        # total_loss.backward(retain_graph=True)

        loss0_pre = criterion(outputs0.narrow(0, 0, 2), F.sigmoid(outputs0.narrow(0, 2, 2)))
        loss1_pre = criterion(outputs1.narrow(0, 0, 2), F.sigmoid(outputs1.narrow(0, 2, 2)))
        loss2_pre = criterion(outputs2.narrow(0, 0, 2), F.sigmoid(outputs2.narrow(0, 2, 2)))

        total_loss_pre = loss0 + 0.5 * loss0_pre + loss1 + 0.5 * loss1_pre + loss2 + 0.5 * loss2_pre
        total_loss_pre.backward(retain_graph=True)
        print_log(total_loss_pre, loss0_pre, loss1_pre, loss2_pre, args['train_batch_size'], curr_iter, optimizer,
              'previous')

        loss0 = criterion(outputs0.narrow(0, 4, 2), batch_gt.narrow(0, 4, 2))
        loss1 = criterion(outputs1.narrow(0, 4, 2), batch_gt.narrow(0, 4, 2))
        loss2 = criterion(outputs2.narrow(0, 4, 2), batch_gt.narrow(0, 4, 2))

        loss0_next = criterion(outputs0.narrow(0, 4, 2), F.sigmoid(outputs0.narrow(0, 2, 2)))
        loss1_next = criterion(outputs1.narrow(0, 4, 2), F.sigmoid(outputs1.narrow(0, 2, 2)))
        loss2_next = criterion(outputs2.narrow(0, 4, 2), F.sigmoid(outputs2.narrow(0, 2, 2)))

        total_loss_next = loss0 + 0.5 * loss0_next + loss1 + 0.5 * loss1_next + loss2 + 0.5 * loss2_next
        total_loss_next.backward(retain_graph=True)
        print_log(total_loss_next, loss0_next, loss1_next, loss2_next, args['train_batch_size'], curr_iter, optimizer,
                  'next')

        loss0 = criterion(outputs0.narrow(0, 2, 2), batch_gt.narrow(0, 2, 2))
        loss1 = criterion(outputs1.narrow(0, 2, 2), batch_gt.narrow(0, 2, 2))
        loss2 = criterion(outputs2.narrow(0, 2, 2), batch_gt.narrow(0, 2, 2))

        loss0_cur1 = criterion(outputs0.narrow(0, 2, 2), F.sigmoid(outputs0.narrow(0, 0, 2)))
        loss1_cur1 = criterion(outputs1.narrow(0, 2, 2), F.sigmoid(outputs1.narrow(0, 0, 2)))
        loss2_cur1 = criterion(outputs2.narrow(0, 2, 2), F.sigmoid(outputs2.narrow(0, 0, 2)))

        loss0_cur2 = criterion(outputs0.narrow(0, 2, 2), F.sigmoid(outputs0.narrow(0, 4, 2)))
        loss1_cur2 = criterion(outputs1.narrow(0, 2, 2), F.sigmoid(outputs1.narrow(0, 4, 2)))
        loss2_cur2 = criterion(outputs2.narrow(0, 2, 2), F.sigmoid(outputs2.narrow(0, 4, 2)))

        total_loss_cur = loss0 + 0.5 * (loss0_cur1 + loss0_cur2) + loss1 + 0.5 * (loss1_cur1 + loss1_cur2) + \
                         loss2 + 0.5 * (loss2_cur1 + loss2_cur2)
        total_loss_cur.backward()

        print_log(total_loss_cur, loss0_cur1 + loss0_cur2, loss1_cur1 + loss1_cur2, loss2_cur1 + loss2_cur2,
                  args['train_batch_size'], curr_iter, optimizer, 'current')

        optimizer.step()

def train_seq3(net, previous_frame, previous_gt, current_frame, current_gt, next_frame, next_gt, optimizer, criterion, curr_iter):
    # previous_gt = Variable(previous_gt).cuda()
    current_gt = Variable(current_gt).cuda()
    # next_gt = Variable(next_gt).cuda()

    # optimizer.zero_grad()
    previous_frame = Variable(previous_frame).cuda()
    net.eval()
    outputs0_pre, outputs1_pre, outputs2_pre, feat_pre = net(previous_frame)

    net.train()
    optimizer.zero_grad()
    current_frame = Variable(current_frame).cuda()
    outputs0_cur, outputs1_cur, outputs2_cur, feat_cur = net(current_frame)
    loss0_cur = criterion(outputs0_cur, current_gt)
    loss1_cur = criterion(outputs1_cur, current_gt)
    loss2_cur = criterion(outputs2_cur, current_gt)

    loss0_cur_pre = criterion(outputs0_cur, outputs0_pre)
    loss1_cur_pre = criterion(outputs1_cur, outputs1_pre)
    loss2_cur_pre = criterion(outputs2_cur, outputs2_pre)

    if args['L2']:
        loss_feat = criterion_l2(feat_pre, feat_cur)
        print('[iter %d], [loss_feat %.5f]' % curr_iter, loss_feat)

    loss0_cur = loss0_cur + 0.5 * loss0_cur_pre
    loss1_cur = loss1_cur + 0.5 * loss1_cur_pre
    loss2_cur = loss2_cur + 0.5 * loss2_cur_pre
    if args['L2']:
        total_loss_cur = loss0_cur + loss1_cur + loss2_cur + 5 * loss_feat
    else:
        total_loss_cur = loss0_cur + loss1_cur + loss2_cur
    total_loss_cur.backward()
    optimizer.step()

    print_log(total_loss_cur, loss0_cur, loss1_cur, loss2_cur, args['train_batch_size'], curr_iter,
              optimizer, 'current')

    return


def train_single(net, inputs, labels, criterion, optimizer, curr_iter):
    inputs = Variable(inputs).cuda()
    labels = Variable(labels).cuda()

    optimizer.zero_grad()
    outputs0, outputs1, outputs2, feat_high, feat_low = net(inputs)
    loss0 = criterion(outputs0, labels)
    loss1 = criterion(outputs1, labels)
    loss2 = criterion(outputs2, labels)
    if args['distillation']:
        loss02 = criterion(outputs0, F.sigmoid(outputs2))
        loss12 = criterion(outputs1, F.sigmoid(outputs2))
        if args['L2']:
            loss_feat = criterion_l2(feat_low, feat_high)
            total_loss = loss0 + loss1 + loss2 + 0.5 * loss02 + 0.5 * loss12 + 10 * loss_feat
        else:
            total_loss = loss0 + loss1 + loss2 + 0.5 * loss02 + 0.5 * loss12
    else:
        total_loss = loss0 + loss1 + loss2

    total_loss.backward()
    optimizer.step()

    print_log(total_loss, loss0, loss1, loss2, args['train_batch_size'], curr_iter, optimizer)

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
