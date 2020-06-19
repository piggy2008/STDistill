import numpy as np
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import ecssd_path, hkuis_path, pascals_path, sod_path, dutomron_path, davis_path
from misc import check_mkdir, crf_refine, AvgMeter, cal_precision_recall_mae, cal_fmeasure
from model_metric import R3Net_prior

torch.manual_seed(2018)

# set which gpu to use
torch.cuda.set_device(0)

# the following two args specify the location of the file of trained model (pth extension)
# you should have the pth file in the folder './$ckpt_path$/$exp_name$'
ckpt_path = './ckpt'
exp_name = 'VideoSaliency_2019-05-03 23:59:44'

args = {
    'snapshot': '30000',  # your snapshot filename (exclude extension name)
    'crf_refine': False,  # whether to use crf to refine results
    'save_results': True,  # whether to save the resulting masks
    'input_size': (473, 473)
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
to_pil = transforms.ToPILImage()

# to_test = {'ecssd': ecssd_path, 'hkuis': hkuis_path, 'pascal': pascals_path, 'sod': sod_path, 'dutomron': dutomron_path}
# to_test = {'ecssd': ecssd_path}
to_test = {'davis': os.path.join(davis_path, 'davis_test2')}
gt_root = os.path.join(davis_path, 'GT')
imgs_path = os.path.join(davis_path, 'davis_test2_5f.txt')
def main():
    net = R3Net_prior(motion='GRU')

    print ('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'), map_location='cuda:0'))
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
                img_var = []
                for img_name in img_seq:
                    img = Image.open(os.path.join(root, img_name + '.jpg')).convert('RGB')
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
