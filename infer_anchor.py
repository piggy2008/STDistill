import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from misc import AvgMeter, check_mkdir
from PIL import Image
from matplotlib import pyplot as plt
import time

from networks.deeplabv3 import ResNetDeepLabv3, ConcatNet
from misc import CriterionDSN
from torch.nn import functional as F
from dataloader.davis_test import PairwiseImg
from misc import check_mkdir, crf_refine, AvgMeter, cal_precision_recall_mae, cal_fmeasure
import cv2
import numpy as np
from utils_mine import MaxMinNormalization

torch.manual_seed(2018)

# set which gpu to use
torch.cuda.set_device(0)

# the following two args specify the location of the file of trained model (pth extension)
# you should have the pth file in the folder './$ckpt_path$/$exp_name$'
ckpt_path = './ckpt'
exp_name = 'VideoSaliency_2019-12-04 22:25:56'

args = {
    'snapshot': '80000',  # your snapshot filename (exclude extension name)
    'crf_refine': False,  # whether to use crf to refine results
    'save_results': True,  # whether to save the resulting masks
    'input_size': (473, 473),
    'sample_range': 1,
    'data_dir': '/home/ty/data/davis',
    'gt_dir': '/home/ty/data/davis/GT'
}
db_test = PairwiseImg(train=False, inputRes=(473,473), db_root_dir=args['data_dir'],  transform=None, seq_name = None, sample_range = args['sample_range']) #db_root_dir() --> '/path/to/DAVIS-2016' train path
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

# to_test = {'ecssd': ecssd_path, 'hkuis': hkuis_path, 'pascal': pascals_path, 'sod': sod_path, 'dutomron': dutomron_path}
# to_test = {'ecssd': ecssd_path}

# to_test = {'davis': os.path.join(davis_path, 'davis_test2')}
# gt_root = os.path.join(davis_path, 'GT')
# imgs_path = os.path.join(davis_path, 'davis_test2_single.txt')

# to_test = {'FBMS': os.path.join(fbms_path, 'FBMS_Testset')}
# gt_root = os.path.join(fbms_path, 'GT')
# imgs_path = os.path.join(fbms_path, 'FBMS_test_single.txt')

# to_test = {'SegTrackV2': os.path.join(segtrack_path, 'SegTrackV2_test')}
# gt_root = os.path.join(segtrack_path, 'GT')
# imgs_path = os.path.join(segtrack_path, 'SegTrackV2_test_single2.txt')

# to_test = {'ViSal': os.path.join(visal_path, 'ViSal_test')}
# gt_root = os.path.join(visal_path, 'GT')
# imgs_path = os.path.join(visal_path, 'ViSal_test_single.txt')

# to_test = {'VOS': os.path.join(vos_path, 'VOS_test')}
# gt_root = os.path.join(vos_path, 'GT')
# imgs_path = os.path.join(vos_path, 'VOS_test_single.txt')

# to_test = {'MCL': os.path.join(mcl_path, 'MCL_test')}
# gt_root = os.path.join(mcl_path, 'GT')
# imgs_path = os.path.join(mcl_path, 'MCL_test_single.txt')

def main():
    net = ConcatNet(backbone='resnet50', embedding=128, batch_mode='old')

    print ('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'), map_location='cuda:0'))
    net.eval()
    net.cuda()
    results = {}

    with torch.no_grad():

        precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
        mae_record = AvgMeter()

        if args['save_results']:
            check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, 'DAVIS', args['snapshot'])))

        index = 0
        old_seq = ''
        # img_list = [os.path.splitext(f)[0] for f in os.listdir(root) if f.endswith('.jpg')]
        for idx, batch in enumerate(testloader):

            print('%d processd ' %(idx))
            target = batch['target']
            seq_name = batch['seq_name']
            args['seq_name'] = seq_name[0]
            print('sequence name:', seq_name[0])
            if old_seq == args['seq_name']:
                index = index + 1
            else:
                index = 0
            output_sum = 0
            for i in range(0, args['sample_range']):
                search = batch['search_' + str(i)]
                output = net(Variable(target, volatile=True).cuda(), Variable(search, volatile=True).cuda())
                output = F.upsample(output, size=target.size()[2:], mode='bilinear', align_corners=True)
                output = F.sigmoid(output)
                output_sum = output_sum + output[0].data.cpu().numpy()

            output_final = output_sum / args['sample_range']
            output_final = cv2.resize(output_final[0], (854, 480))
            output_final = (output_final * 255).astype(np.uint8)

            gt = np.array(Image.open(os.path.join(args['gt_dir'], args['seq_name'], str(index).zfill(5) + '.png')).convert('L'))
            precision, recall, mae = cal_precision_recall_mae(output_final, gt)
            for pidx, pdata in enumerate(zip(precision, recall)):
                p, r = pdata
                precision_record[pidx].update(p)
                recall_record[pidx].update(r)
            mae_record.update(mae)

            if args['save_results']:
                old_seq = args['seq_name']
                save_path = os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, 'DAVIS', args['snapshot']), args['seq_name'])
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                Image.fromarray(output_final).save(os.path.join(save_path, str(index).zfill(5) + '.png'))

        fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                [rrecord.avg for rrecord in recall_record])

        results['DAVIS'] = {'fmeasure': fmeasure, 'mae': mae_record.avg}

    print ('test results:')
    print (results)


if __name__ == '__main__':
    main()