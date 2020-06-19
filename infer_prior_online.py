import numpy as np
import os
import torch.nn as nn
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

from config import ecssd_path, hkuis_path, pascals_path, sod_path, dutomron_path, davis_path, fbms_path
from misc import check_mkdir, crf_refine, AvgMeter, cal_precision_recall_mae, cal_fmeasure
from model_prior import R3Net_prior
from torch import optim

import joint_transforms
from config import msra10k_path, video_train_path, datasets_root, video_seq_gt_path, video_seq_path
from datasets import ImageFolder, VideoImageFolder, VideoFirstImageFolder
from matplotlib import pyplot as plt

torch.manual_seed(2018)

# set which gpu to use
torch.cuda.set_device(0)

# the following two args specify the location of the file of trained model (pth extension)
# you should have the pth file in the folder './$ckpt_path$/$exp_name$'
ckpt_path = './ckpt'
exp_name = 'VideoSaliency_2019-05-13 18:57:27'

args = {
    'snapshot': '30000',  # your snapshot filename (exclude extension name)
    'crf_refine': False,  # whether to use crf to refine results
    'save_results': True,  # whether to save the resulting masks
    'input_size': (473, 473),
    'batch_size': 5
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
to_pil = transforms.ToPILImage()

to_test = {'davis': os.path.join(davis_path, 'davis_test2')}
gt_root = os.path.join(davis_path, 'GT')
imgs_path = os.path.join(davis_path, 'davis_test2_5f.txt')

# to_test = {'FBMS': os.path.join(fbms_path, 'FBMS_Testset')}
# gt_root = os.path.join(fbms_path, 'GT')
# imgs_path = os.path.join(fbms_path, 'FBMS_seq_file_5f.txt')

def fix_parameters(parameters):
    for name, parameter in parameters:
        if name.find('motion') >= 0 or name.find('attention') >= 0\
                or name.find('GRU') >= 0:
            print(name, 'is not fixed')

        else:
            print(name, 'is fixed')
            parameter.requires_grad = False

def train_online(net, seq_name='breakdance'):
    online_args = {
        'iter_num': 100,
        'train_batch_size': 5,
        'lr': 1e-8,
        'lr_decay': 0.95,
        'weight_decay': 5e-4,
        'momentum': 0.95,
    }

    joint_transform = joint_transforms.Compose([
        joint_transforms.ImageResize(473),
        # joint_transforms.RandomCrop(473),
        # joint_transforms.RandomHorizontallyFlip(),
        # joint_transforms.RandomRotate(10)
    ])
    target_transform = transforms.ToTensor()
    train_set = VideoFirstImageFolder(to_test['davis'], gt_root, seq_name, online_args['train_batch_size'], joint_transform, img_transform, target_transform)
    online_train_loader = DataLoader(train_set, batch_size=online_args['train_batch_size'], num_workers=1, shuffle=False)

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * online_args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': online_args['lr'], 'weight_decay': online_args['weight_decay']}
    ], momentum=online_args['momentum'])

    criterion = nn.BCEWithLogitsLoss().cuda()
    net.train().cuda()
    fix_parameters(net.named_parameters())
    for curr_iter in range(0, online_args['iter_num']):
        total_loss_record, loss0_record, loss1_record = AvgMeter(), AvgMeter(), AvgMeter()
        loss2_record, loss3_record, loss4_record = AvgMeter(), AvgMeter(), AvgMeter()

        for i, data in enumerate(online_train_loader):
            optimizer.param_groups[0]['lr'] = 2 * online_args['lr'] * (1 - float(curr_iter) / online_args['iter_num']
                                                                ) ** online_args['lr_decay']
            optimizer.param_groups[1]['lr'] = online_args['lr'] * (1 - float(curr_iter) / online_args['iter_num']
                                                            ) ** online_args['lr_decay']
            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()
            outputs0, outputs1, outputs2, outputs3, outputs4 = net(inputs)
            loss0 = criterion(outputs0, labels)
            loss1 = criterion(outputs1, labels.narrow(0, 1, 4))
            loss2 = criterion(outputs2, labels.narrow(0, 2, 3))
            loss3 = criterion(outputs3, labels.narrow(0, 3, 2))
            loss4 = criterion(outputs4, labels.narrow(0, 4, 1))

            total_loss = loss0 + loss1 + loss2 + loss3 + loss4
            total_loss.backward()
            optimizer.step()

            total_loss_record.update(total_loss.data, batch_size)
            loss0_record.update(loss0.data, batch_size)
            loss1_record.update(loss1.data, batch_size)
            loss2_record.update(loss2.data, batch_size)
            loss3_record.update(loss3.data, batch_size)
            loss4_record.update(loss4.data, batch_size)

            log = '[iter %d], [total loss %.5f], [loss0 %.5f], [loss1 %.5f], [loss2 %.5f], [loss3 %.5f], ' \
                  '[loss4 %.5f], [lr %.13f]' % \
                  (curr_iter, total_loss_record.avg, loss0_record.avg, loss1_record.avg, loss2_record.avg,
                   loss3_record.avg, loss4_record.avg, optimizer.param_groups[1]['lr'])
            print(log)

    return net


def main():
    net = R3Net_prior(motion='GRU', se_layer=False, st_fuse=False)

    print ('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'), map_location='cuda:0'))
    # net = train_online(net)
    results = {}



    for name, root in to_test.items():

        precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
        mae_record = AvgMeter()

        if args['save_results']:
            check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

        folders = os.listdir(root)
        folders.sort()
        for folder in folders:
            net = train_online(net, seq_name=folder)
            with torch.no_grad():

                net.eval()
                net.cuda()
                imgs = os.listdir(os.path.join(root, folder))
                imgs.sort()
                for i in range(1, len(imgs) - args['batch_size'] + 1):
                    print(imgs[i])
                    img_var = []
                    img_names = []
                    for j in range(0, args['batch_size']):
                        img = Image.open(os.path.join(root, folder, imgs[i + j])).convert('RGB')
                        img_names.append(imgs[i + j])
                        shape = img.size
                        img = img.resize(args['input_size'])
                        img_var.append(Variable(img_transform(img).unsqueeze(0), volatile=True).cuda())

                    img_var = torch.cat(img_var, dim=0)
                    prediction = net(img_var)
                    precision = to_pil(prediction.data.squeeze(0).cpu())
                    precision = precision.resize(shape)
                    prediction = np.array(precision)

                    if args['crf_refine']:
                        prediction = crf_refine(np.array(img), prediction)
                    gt = np.array(Image.open(os.path.join(gt_root, folder, img_names[-1][:-4] + '.png')).convert('L'))
                    precision, recall, mae = cal_precision_recall_mae(prediction, gt)
                    for pidx, pdata in enumerate(zip(precision, recall)):
                        p, r = pdata
                        precision_record[pidx].update(p)
                        recall_record[pidx].update(r)
                    mae_record.update(mae)

                    if args['save_results']:
                        # folder, sub_name = os.path.split(img_names[-1])
                        save_path = os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot']),
                                                 folder)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        Image.fromarray(prediction).save(os.path.join(save_path, img_names[-1][:-4] + '.png'))

        fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                [rrecord.avg for rrecord in recall_record])

        results[name] = {'fmeasure': fmeasure, 'mae': mae_record.avg}

    print ('test results:')
    print (results)


if __name__ == '__main__':
    main()
