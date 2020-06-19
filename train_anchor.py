import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import joint_transforms
from config import msra10k_path, video_train_path, datasets_root, video_seq_gt_path, video_seq_path
from datasets import ImageFolder, VideoImageFolder, VideoSequenceFolder
from misc import AvgMeter, check_mkdir
from model_sasc import R3Net

from torch.backends import cudnn
import time
from utils_mine import load_part_of_model
from utils.parallel import DataParallelModel, DataParallelCriterion
from dataloader.davis_2016 import DAVIS2016
from dataloader.davis_single import DAVIS_Single
import dataloader.custom_transforms as tr
from networks.deeplabv3 import ResNetDeepLabv3, ConcatNet
from misc import CriterionDSN
from torch.nn import functional as F


cudnn.benchmark = True
device_id = 1
torch.manual_seed(2019)
torch.cuda.set_device(device_id)

time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
ckpt_path = './ckpt'
exp_name = 'VideoSaliency' + '_' + time_str

args = {
    'basic_model': 'resnet50',
    'motion': '',
    'se_layer': False,
    'attention': True,
    'dilation': True,
    'iter_num': 80000,
    'iter_save': 20000,
    'train_batch_size': 6,
    'last_iter': 0,
    'lr': 5 * 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    # 'pretrain': os.path.join(ckpt_path, 'VideoSaliency_2019-11-27 22:38:04', '10000.pth'),
    'pretrain': '',
    'imgs_file': 'Pre-train/pretrain_all_seq_DUT_TR_DAFB2_DAVSOD.txt',
    # 'imgs_file': 'video_saliency/train_all_DAFB2_DAVSOD_5f.txt',
    'train_loader': 'video_image'
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
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

target_transform = transforms.ToTensor()

davis_transforms = transforms.Compose([
    tr.RandomHorizontalFlip(),
    tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
    tr.ToTensor()]
)



# train_set = ImageFolder(msra10k_path, joint_transform, img_transform, target_transform)
if args['train_loader'] == 'video_sequence':
    train_set = VideoSequenceFolder(video_seq_path, video_seq_gt_path, imgs_file, joint_transform, img_transform, target_transform)
else:
    # train_set = DAVIS2016(db_root_dir='/home/ty/data/davis', train=True, transform=None)
    train_set = DAVIS_Single(db_image_dir='/home/ty/data/Pre-train', train=True, transform=None)
    # train_set = VideoImageFolder(video_train_path, imgs_file, joint_transform, img_transform, target_transform)

train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)

criterion = nn.BCEWithLogitsLoss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')

def fix_parameters(parameters):
    for name, parameter in parameters:
        if name.find('motion') >= 0 \
                or name.find('GRU') >= 0 or name.find('predict') >= 0:
            print(name, 'is not fixed')

        else:
            print(name, 'is fixed')
            parameter.requires_grad = False


def main():

    net = ConcatNet(backbone=args['basic_model'], embedding=128, batch_mode='old')
    net.train()
    net.float()
    net.cuda()
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
        net = load_part_of_model(net, args['pretrain'], device_id=0)

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, loss0_record, loss1_record, loss2_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        # loss3_record = AvgMeter()

        for i, data in enumerate(train_loader):

            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            # inputs = data['image']
            # labels = data['gt']
            # inputs, labels = data['video'], data['video_gt']
            # search, search_labels = data['search'], data['search_gt']
            image, image_labels = data['img'], data['img_gt']
            # print(labels.size())

            batch_size = image.size(0)
            # inputs = Variable(inputs).cuda()
            # labels = Variable(labels.float().unsqueeze(1)).cuda()
            #
            # search = Variable(search).cuda()
            # search_labels = Variable(search_labels.float().unsqueeze(1)).cuda()

            image = Variable(image).cuda()
            image_labels = Variable(image_labels.float().unsqueeze(1)).cuda()

            optimizer.zero_grad()
            # outputs0, outputs1 = net(inputs)

            preds = net(image, image, flag='image')
            preds = F.upsample(preds, size=image.size()[2:], mode='bilinear', align_corners=True)
            loss0 = criterion(preds, image_labels)

            # if curr_iter % 3 == 0:
            #     preds = net(search, inputs, flag='video')
            #     preds = F.upsample(preds, size=inputs.size()[2:], mode='bilinear', align_corners=True)
            #     loss0 = criterion(preds, labels)
            # else:
            #     preds = net(image, image, flag='image')
            #     preds = F.upsample(preds, size=image.size()[2:], mode='bilinear', align_corners=True)
            #     loss0 = criterion(preds, image_labels)
            # loss2 = criterion(outputs2, labels)
            # loss3 = criterion(outputs3, labels)
            # loss4 = criterion(outputs4, labels)

            total_loss = loss0

            total_loss.backward()
            optimizer.step()

            total_loss_record.update(total_loss.data, batch_size)
            loss0_record.update(loss0.data, batch_size)
            loss1_record.update(loss0.data, batch_size)
            # loss2_record.update(loss2.data, batch_size)
            # loss3_record.update(loss3.data, batch_size)
            # loss4_record.update(loss4.data, batch_size)



            curr_iter += 1

            log = '[iter %d], [total loss %.5f], [loss0 %.5f], [loss1 %.5f] ' \
                  '[lr %.13f]' % \
                  (curr_iter, total_loss_record.avg, loss0_record.avg, loss1_record.avg,
                   optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

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


if __name__ == '__main__':
    main()
