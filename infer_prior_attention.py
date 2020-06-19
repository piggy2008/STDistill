import numpy as np
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import ecssd_path, hkuis_path, pascals_path, sod_path, dutomron_path, \
    davis_path, fbms_path, mcl_path, uvsd_path, visal_path, vos_path, segtrack_path, davsod_path
from misc import check_mkdir, crf_refine, AvgMeter, cal_precision_recall_mae, cal_fmeasure
from model_prior_attention import R3Net_prior
from utils_mine import MaxMinNormalization
import time
torch.manual_seed(2018)

# set which gpu to use
torch.cuda.set_device(3)

# the following two args specify the location of the file of trained model (pth extension)
# you should have the pth file in the folder './$ckpt_path$/$exp_name$'
ckpt_path = './ckpt'
exp_name = 'VideoSaliency_2020-02-16 22:31:50'

# VideoSaliency_2019-08-15 05:22:35
args = {
    'snapshot': '20000',  # your snapshot filename (exclude extension name)
    'crf_refine': False,  # whether to use crf to refine results
    'save_results': True,  # whether to save the resulting masks
    'input_size': (473, 473),
    'scale_ratio': [1]
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

to_test = {'davis': os.path.join(davis_path, 'davis_test2')}
gt_root = os.path.join(davis_path, 'GT')
imgs_path = os.path.join(davis_path, 'davis_test2_6f.txt')
#
# to_test = {'FBMS': os.path.join(fbms_path, 'FBMS_Testset')}
# gt_root = os.path.join(fbms_path, 'GT')
# imgs_path = os.path.join(fbms_path, 'FBMS_seq_file_5f.txt')

# to_test = {'MCL': os.path.join(mcl_path, 'MCL_test')}
# gt_root = os.path.join(mcl_path, 'GT')
# imgs_path = os.path.join(mcl_path, 'MCL_test_5f.txt')

# to_test = {'UVSD': os.path.join(uvsd_path, 'UVSD_test')}
# gt_root = os.path.join(uvsd_path, 'GT')
# imgs_path = os.path.join(uvsd_path, 'UVSD_test_5f.txt')

# to_test = {'ViSal': os.path.join(visal_path, 'ViSal_test')}
# gt_root = os.path.join(visal_path, 'GT')
# imgs_path = os.path.join(visal_path, 'ViSal_test_5f.txt')

# to_test = {'VOS': os.path.join(vos_path, 'VOS_test')}
# gt_root = os.path.join(vos_path, 'GT')
# imgs_path = os.path.join(vos_path, 'VOS_test_6f.txt')


# to_test = {'SegTrackV2': os.path.join(segtrack_path, 'SegTrackV2_rain_test')}
# gt_root = os.path.join(segtrack_path, 'GT')
# imgs_path = os.path.join(segtrack_path, 'SegTrackV2_test_5f.txt')

# to_test = {'DAVSOD': os.path.join(davsod_path, 'DAVSOD_test')}
# gt_root = os.path.join(davsod_path, 'GT')
# imgs_path = os.path.join(davsod_path, 'DAVSOD_test_6f.txt')

def main():
    net = R3Net_prior(motion='GRU', se_layer=False, attention=False, pre_attention=True, basic_model='resnet50', sta=False, naive_fuse=False)

    print ('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'), map_location='cuda:1'))
    net.eval()
    net.cuda()
    results = {}

    with torch.no_grad():

        for name, root in to_test.items():

            precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
            mae_record = AvgMeter()

            if args['save_results']:
                check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))
            img_list = [i_id.strip() for i_id in open(imgs_path)]
            # img_list = [os.path.splitext(f)[0] for f in os.listdir(root) if f.endswith('.jpg')]
            for idx, img_names in enumerate(img_list):
                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                img_seq = img_names.split(',')

                for ratio in args['scale_ratio']:
                    prediction_scale = []
                    img_var = []
                    for img_name in img_seq:
                        if name == 'VOS' or name == 'DAVSOD':
                            img = Image.open(os.path.join(root, img_name + '.png')).convert('RGB')
                        else:
                            img = Image.open(os.path.join(root, img_name + '.jpg')).convert('RGB')
                        shape = img.size

                        new_dims = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                        img = img.resize(new_dims, Image.BILINEAR)
                        img = img.resize(args['input_size'])
                        img_var.append(Variable(img_transform(img).unsqueeze(0), volatile=True).cuda())

                    img_var = torch.cat(img_var, dim=0)
                    start = time.time()
                    prediction = net(img_var)
                    end = time.time()
                    print('running time:', (end - start))
                    prediction_scale.append(prediction)
                prediction = torch.mean(torch.stack(prediction_scale, dim=0), 0)
                prediction = to_pil(prediction.data.squeeze(0).cpu())
                prediction = prediction.resize(shape)
                prediction = np.array(prediction)
                prediction = prediction.astype('float')
                prediction = MaxMinNormalization(prediction, prediction.max(), prediction.min()) * 255.0
                prediction = prediction.astype('uint8')

                if args['crf_refine']:
                    prediction = crf_refine(np.array(img), prediction)

                gt = np.array(Image.open(os.path.join(gt_root, img_seq[-1] + '.png')).convert('L'))
                precision, recall, mae = cal_precision_recall_mae(prediction, gt)
                for pidx, pdata in enumerate(zip(precision, recall)):
                    p, r = pdata
                    precision_record[pidx].update(p)
                    recall_record[pidx].update(r)
                mae_record.update(mae)

                if args['save_results']:
                    folder, sub_name = os.path.split(img_name)
                    save_path = os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot']), folder)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    Image.fromarray(prediction).save(os.path.join(save_path, sub_name + '.png'))

            fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                    [rrecord.avg for rrecord in recall_record])

            results[name] = {'fmeasure': fmeasure, 'mae': mae_record.avg}

    print ('test results:')
    print (results)


if __name__ == '__main__':
    main()
