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
from model_step import R3Net
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
    'motion': '',
    'iter_num': 30000,
    'iter_save': 10000,
    'train_batch_size': 5,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    # 'pretrain': os.path.join(ckpt_path, 'VideoSaliency_2019-04-26 05:36:42', '30000.pth'),
    'pretrain': '',
    'imgs_file': 'Pre-train/pretrain_all_seq2.txt',
    # 'imgs_file': 'video_saliency/train_all_DAFB2_seq_5f.txt',
    'train_loader': 'video_image'
    # 'train_loader': 'video_sequence'
}

imgs_file = os.path.join(datasets_root, args['imgs_file'])
# imgs_file = os.path.join(datasets_root, 'video_saliency/train_all_DAFB3_seq_5f.txt')

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(473),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

# train_set = ImageFolder(msra10k_path, joint_transform, img_transform, target_transform)
if args['train_loader'] == 'video_sequence':
    train_set = VideoSequenceFolder(video_seq_path, video_seq_gt_path, imgs_file, joint_transform, img_transform, target_transform)
else:
    train_set = VideoImageFolder(video_train_path, imgs_file, joint_transform, img_transform, target_transform)

train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True)

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
    net = R3Net(motion=args['motion']).cuda().train()

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
        total_loss_record, loss0_record, loss1_record = AvgMeter(), AvgMeter(), AvgMeter()
        loss2_record, loss3_record, loss4_record, loss5_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        # loss2_record, loss3_record, loss4_record = AvgMeter(), AvgMeter(), AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, labels = data
            if args['train_loader'] == 'video_sequence':
                inputs = inputs.squeeze(0)
                labels = labels.squeeze(0)
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()
            outputs0, outputs1, outputs2, outputs3, outputs4, outputs5 = net(inputs)
            loss0 = criterion(outputs0, labels)
            loss1 = criterion(outputs1, labels)
            loss2 = criterion(outputs2, labels)
            loss3 = criterion(outputs3, labels)
            loss4 = criterion(outputs4, labels)
            loss5 = criterion(outputs5, labels)

            total_loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
            total_loss.backward()
            optimizer.step()

            total_loss_record.update(total_loss.data, batch_size)
            loss0_record.update(loss0.data, batch_size)
            loss1_record.update(loss1.data, batch_size)
            loss2_record.update(loss2.data, batch_size)
            loss3_record.update(loss3.data, batch_size)
            loss4_record.update(loss4.data, batch_size)
            loss5_record.update(loss5.data, batch_size)

            curr_iter += 1

            log = '[iter %d], [total loss %.5f], [loss0 %.5f], [loss1 %.5f], [loss2 %.5f], [loss3 %.5f], [loss4 %.5f], [loss5 %.5f], [lr %.13f]' % \
                  (curr_iter, total_loss_record.avg, loss0_record.avg, loss1_record.avg,
                   loss2_record.avg, loss3_record.avg, loss4_record.avg, loss5_record.avg, optimizer.param_groups[1]['lr'])
            print (log)
            open(log_path, 'a').write(log + '\n')

            if curr_iter % args['iter_save'] == 0:
                print('taking snapshot ...', ckpt_path)
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))

            if curr_iter == args['iter_num']:
                print('taking snapshot ...', ckpt_path)
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
                return


if __name__ == '__main__':
    main()
