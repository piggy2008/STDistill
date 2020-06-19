import os
import numpy as np
from PIL import Image
from misc import check_mkdir, crf_refine, AvgMeter, cal_precision_recall_mae, cal_fmeasure

ckpt_path = './ckpt'
exp_name = 'VideoSaliency_2019-08-15 05:22:35'
name = 'davis'
root = '/home/qub/data/saliency/davis/davis_test2'
gt_root = '/home/qub/data/saliency/davis/GT'
# gt_root = '/home/qub/data/saliency/VOS/GT'
args = {
    'snapshot': '20000',  # your snapshot filename (exclude extension name)
    'crf_refine': True,  # whether to use crf to refine results
    'save_results': True  # whether to save the resulting masks
}

precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
mae_record = AvgMeter()
results = {}

save_path = os.path.join(ckpt_path, exp_name, '(%s) %s_%s_eval' % (exp_name, name, args['snapshot']))
crf_save_path = os.path.join(ckpt_path, exp_name, '(%s) %s_%s_crf_eval' % (exp_name, name, args['snapshot']))
folders = os.listdir(save_path)
folders.sort()
for folder in folders:
    imgs = os.listdir(os.path.join(save_path, folder, '1'))
    imgs.sort()

    for img in imgs:
        print(os.path.join(folder, '1', img))
        if name == 'VOS':
            image = Image.open(os.path.join(root, folder, img[:-4] + '.png')).convert('RGB')
        else:
            image = Image.open(os.path.join(root, folder, img[:-4] + '.jpg')).convert('RGB')
        gt = np.array(Image.open(os.path.join(gt_root, folder, img)).convert('L'))
        pred = np.array(Image.open(os.path.join(save_path, folder, '1', img)).convert('L'))
        if args['crf_refine']:
            pred = crf_refine(np.array(image), pred)
        precision, recall, mae = cal_precision_recall_mae(pred, gt)

        for pidx, pdata in enumerate(zip(precision, recall)):
            p, r = pdata
            precision_record[pidx].update(p)
            recall_record[pidx].update(r)
        mae_record.update(mae)

        if args['save_results']:
            if not os.path.exists(os.path.join(crf_save_path, folder, '1')):
                os.makedirs(os.path.join(crf_save_path, folder, '1'))
            Image.fromarray(pred).save(os.path.join(crf_save_path, folder, '1', img))

fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                    [rrecord.avg for rrecord in recall_record])

results[name] = {'fmeasure': fmeasure, 'mae': mae_record.avg}

print ('test results:')
print (results)

# THUR15K + DAVIS snap:10000 {'davis': {'mae': 0.03617724595417807, 'fmeasure': 0.8150494537915058}}
# THUR15K + DAVIS(input no mea & std) snap:30000 {'davis': {'mae': 0.03403602471853535}, 'fmeasure': 0.8208723312824877}
# THUR15K + DAVIS snap:30000 {'davis': {'mae': 0.02795341027164935}, 'fmeasure': 0.846696146351338}
# THUR15K + DAVIS resize:473*473 snap:30000 {'davis': 'mae': 0.02464488739008121, ''fmeasure': 0.8753527027151914}
# THUR15K + DAVIS resize:473*473 model:R1 high and low, snap:30000 {'davis': {'fmeasure': 0.8657611483587979, 'mae': 0.028688147260396805}}
# THUR15K + DAVIS resize:473*473 model: model prior recurrent snap:30000 {'davis': {'mae': 0.02533309706615563, 'fmeasure': 0.8745875295714605}}
# THUR15K + DAVIS resize:473*473 model: model prior recurrent + feature maps plus
# snap:30000 {'davis': {'fmeasure': 0.8751256401745396, 'mae': 0.025352599605078505}}

# VideoSaliency_2019-05-03 00:54:21 is better, using model_prior, R3Net base and add previous frame supervision and recurrent GRU motion extraction
# training details, first, directly train R3Net using DAFB2 and THUR15K, second, finetune the model by add recurrent module and GRU, then finetune twice
# using DAFB2 and THUR15K but dataloader shuffle=false in order to have consecutive frames. The specific super parameter is in VideoSaliency_2019-05-03 00:54:21
# VideoSaliency_2019-05-01 23:29:39 and VideoSaliency_2019-04-20 23:11:17/30000.pth

# VideoSaliency_2019-05-03 23:59:44: finetune model prior from 05-01 model, fix other layers excepet motion module
# {'davis': {'mae': 0.031455319655690664, 'fmeasure': 0.8687384596915435}}

# VideoSaliency_2019-05-14 17:13:16: no finetune
# using dataset:DUT-OMRON + DAVIS R3Net pre-train
# {'davis': {'fmeasure': 0.8760938218680382, 'mae': 0.03375186721061853}}

# VideoSaliency_2019-05-15 03:06:29: finetune model prior from VideoSaliency_2019-05-14 17:13:16 model, train entire network with lr:1e-6
# using dataset:DUT-OMRON + DAVIS R3Net pre-train
# {'davis': {'fmeasure': 0.8770158996877871, 'mae': 0.03235241246303723}}

# VideoSaliency_2019-05-15 03:06:29: finetune model prior from VideoSaliency_2019-05-14 17:13:16 model, train entire network with lr:1e-5
# using dataset:DUT-OMRON + DAVIS R3Net pre-train
# {'davis': {'mae': 0.02977316776424702, 'fmeasure': 0.8773961688318479}}
# {'FBMS': {'fmeasure': 0.8462238927200698, 'mae': 0.05929029351096353}}

# VideoSaliency_2019-05-17 03:27:37: finetune model prior from VideoSaliency_2019-05-14 17:13:16 model, train entire network with lr:1e-5
# using dataset:DUT-OMRON + DAVIS R3Net pre-train
# model: self-attention + motion enhancement + prior attention weight learning
# {'FBMS': {'fmeasure': 0.8431560452294077, 'mae': 0.0572594186609631}}
# {'FBMS': {'mae': 0.05151967407911611, 'fmeasure': 0.8512965990283861}} with crf
# {'VOS': {'fmeasure': 0.7693856907104227, 'mae': 0.07323270547216723}}
# {'VOS': {'mae': 0.061354405913717075, 'fmeasure': 0.76979294074132}} with crf
# {'SegTrackV2': {'fmeasure': 0.8900102827035228, 'mae': 0.02371825726384187}}
# {'SegTrackV2': {'mae': 0.01414643253248216, 'fmeasure': 0.8974274867145704}} with CRF
# {'MCL': {'fmeasure': 0.7941665988086701, 'mae': 0.03365593652205517}}
# {'MCL': {'fmeasure': 0.8033409666446579, 'mae': 0.030916401685247424}} with crf
# {'ViSal': {'mae': 0.01547489956096272, 'fmeasure': 0.9517413442552852}}
# {'ViSal': {'fmeasure': 0.9541724935997185, 'mae': 0.009944043273381801}} with crf
# {'davis': {'fmeasure': 0.877271448077333, 'mae': 0.028900763530552247}}
# {'davis': {'fmeasure': 0.8877485369547635, 'mae': 0.017803576387589698}} with crf