import os
import numpy as np
from PIL import Image
from misc import check_mkdir, crf_refine, AvgMeter, cal_precision_recall_mae, cal_fmeasure

ckpt_path = './ckpt'
exp_name = 'VideoSaliency_2019-12-19 10:43:54'
name = 'davis'
# root = '/home/qub/data/saliency/FBMS/FBMS_Testset2'
root = '/home/ty/data/davis/davis_test2'
# root = '/home/ty/data/VOS/VOS_test'
# root = '/home/ty/data/SegTrack-V2/SegTrackV2_test'
# root = '/home/ty/data/ViSal/ViSal_test'
# root = '/home/ty/data/MCL/MCL_test'
# root = '/home/ty/data/DAVSOD/DAVSOD_test'

gt_root = '/home/ty/data/davis/GT'
# gt_root = '/home/ty/data/VOS/GT'
# gt_root = '/home/qub/data/saliency/FBMS/GT'
# gt_root = '/home/ty/data/MCL/GT'
# gt_root = '/home/ty/data/ViSal/GT'
# gt_root = '/home/ty/data/DAVSOD/GT'
# gt_root = '/home/ty/data/SegTrack-V2/GT'
args = {
    'snapshot': '80000',  # your snapshot filename (exclude extension name)
    'crf_refine': False,  # whether to use crf to refine results
    'save_results': True  # whether to save the resulting masks
}

precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
mae_record = AvgMeter()
results = {}

save_path = os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot']))
folders = os.listdir(save_path)
folders.sort()
for folder in folders:
    imgs = os.listdir(os.path.join(save_path, folder))
    imgs.sort()

    for img in imgs:
        print(os.path.join(folder, img))
        if name == 'VOS' or name == 'DAVSOD':
            image = Image.open(os.path.join(root, folder, img[:-4] + '.png')).convert('RGB')
        else:
            image = Image.open(os.path.join(root, folder, img[:-4] + '.jpg')).convert('RGB')
        gt = np.array(Image.open(os.path.join(gt_root, folder, img)).convert('L'))
        pred = np.array(Image.open(os.path.join(save_path, folder, img)).convert('L'))
        if args['crf_refine']:
            pred = crf_refine(np.array(image), pred)
        precision, recall, mae = cal_precision_recall_mae(pred, gt)

        for pidx, pdata in enumerate(zip(precision, recall)):
            p, r = pdata
            precision_record[pidx].update(p)
            recall_record[pidx].update(r)
        mae_record.update(mae)

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

# VideoSaliency_2019-06-25 00:58:16 traning from original resnext50 model of torch parameter, using dataset:DUT-TR + DAVIS model: raw R3Net lr:0.001
# {'davis': {'fmeasure': 0.8697159794201897, 'mae': 0.035606949365716525}}

# VideoSaliency_2019-06-25 17:44:55 traning from original resnext50 model of torch parameter, using dataset:DUT-TR + DAVIS
# finetune VideoSaliency_2019-06-25 00:58:16 model:resnext50 + R3Net + GRU + motion enhancement + saliency guide block
# {'davis': {'mae': 0.034840332588290626, 'fmeasure': 0.8566140865851933}}

# VideoSaliency_2019-06-25 00:42:59 traning from original resnext50 model of torch parameter, using dataset:DUT-OMRON + DAVIS model: raw R3Net lr:0.001
# {'davis': {'mae': 0.026136525479663834, 'fmeasure': 0.8583683681098009}}

# VideoSaliency_2019-06-25 18:35:28 traning from original resnext50 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-25 00:42:59 model:resnext50 + R3Net + GRU + motion enhancement + saliency guide block
# {'davis': {'mae': 0.027190812067633865, 'fmeasure': 0.8564467742823199}}

# VideoSaliency_2019-06-25 18:46:13 traning from original resnext50 model of torch parameter, using dataset:DUT-OMRON + DAVIS model: raw R3Net lr:0.001
# no self_attention
# {'davis': {'mae': 0.028268766387925158, 'fmeasure': 0.8641904514712092}}

# VideoSaliency_2019-06-26 00:07:16 traning from original resnext101 model of torch parameter, using dataset:DUT-TR + DAVIS model: raw R3Net lr:0.001
# no self_attention
# {'davis': {'fmeasure': 0.8744918145377412, 'mae': 0.028497783782586317}}

# VideoSaliency_2019-06-26 00:49:01 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS model: raw R3Net lr:0.001
# no self_attention
# {'davis': {'mae': 0.02956942893893325, 'fmeasure': 0.8636986541096229}}

# finetune VideoSaliency_2019-06-26 00:07:16 traning from original resnext101 model of torch parameter, using dataset:DUT-TR + DAVIS model: raw R3Net lr:0.001
# self_attention
# {'davis': {'mae': 0.0284977837825863, 'fmeasure': 0.874491814537741}}

# VideoSaliency_2019-06-26 18:08:11(20000)traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-26 00:07:16 model:resnext101 + R3Net + GRU + motion enhancement + saliency guide block
# no self_attention
# {'davis': {'mae': 0.027365099548091857, 'fmeasure': 0.8843037688863674}} 20000
# {'davis': {'fmeasure': 0.882500630734551, 'mae': 0.028656513632573044}} 30000
# {'MCL': {'fmeasure': 0.7797077599010649, 'mae': 0.03490434030025704}}
# {'ViSal': {'mae': 0.015330959018609812, 'fmeasure': 0.949898057517949}}
# {'FBMS': {'mae': 0.06180595295896911, 'fmeasure': 0.8339872466494525}}
# {'VOS': {'mae': 0.07404737148285567, 'fmeasure': 0.759238636002531}}

# VideoSaliency_2019-06-26 18:42:54 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-26 00:49:01 model:resnext101 + R3Net + GRU + motion enhancement + saliency guide block
# no self_attention
# {'davis': {'fmeasure': 0.8632752266559531, 'mae': 0.029003498754963063}} 20000
# {'davis': {'fmeasure': 0.864925, 'mae': 0.0289121}} 30000


# VideoSaliency_2019-07-11 00:09:51 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-26 00:07:16 model:resnext101 + R3Net + no motion block + motion enhancement + saliency guide block
# no self_attention
# {'davis': {'fmeasure': 0.88136, 'mae': 0.028495}} 30000
# {'davis': {'fmeasure': 0.884773, 'mae': 0.0271906}} 20000
# {'davis': {'fmeasure': 0.885063, 'mae': 0.0285210}} 10000
# {'FBMS': {'mae': 0.05814119064871477, 'fmeasure': 0.840094490963166}}
# {'MCL': {'mae': 0.0345719343627967, 'fmeasure': 0.7831879684122651}}

# VideoSaliency_2019-07-11 00:09:51 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-26 00:07:16 model:resnext101 + R3Net + motion block + motion enhancement + saliency guide block
# no self_attention motion block channel is 256 to 64
# {'davis': {'mae': 0.027200999676079675, 'fmeasure': 0.8819144797935291}}
# {'FBMS': {'mae': 0.058138874315261733, 'fmeasure': 0.8362034796659599}}
# {'ViSal': {'mae': 0.01453721749511041, 'fmeasure': 0.9498225723351809}}
# {'MCL': {'mae': 0.033866961735088005, 'fmeasure': 0.7857094621253102}}
# {'SegTrackV2': {'mae': 0.023640045394179604, 'fmeasure': 0.8774014546489907}}
# {'VOS': {'mae': 0.07308082925115635, 'fmeasure': 0.761926607316529}}
# {'UVSD': {'mae': 0.03607475861206765, 'fmeasure': 0.7011512131755825}}
# {'DAVSOD': {'fmeasure': 0.5845565863898012, 'mae': 0.09428676452230018}}

# VideoSaliency_2019-06-27 00:56:18 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-26 00:07:16 model:resnext101 + R3Net + no motion block + motion enhancement
# no self_attention
# {'davis':  {'mae': 0.027359359945784614, 'fmeasure': 0.8840887753328227}} 30000
# {'davis': {'mae': 0.017455241696254332, 'fmeasure': 0.8905493155021433}} with crf


# VideoSaliency_2019-06-28 22:24:22 traning from original resnet50 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-28 18:33:02 model:resnet50 + R3Net + GRU + motion enhancement + saliency guide block
# no self_attention
# {'MCL': {'fmeasure': 0.7759971653989058, 'mae': 0.033103970530375657}}
# {'davis': {'mae': 0.02952947523541565, 'fmeasure': 0.8601778871578505}}

# VideoSaliency_2019-06-27 19:00:52 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-05-14 17:13:16 model:resnext101 + R3Net + GRU + motion enhancement + no saliency guide block
# have self_attention
# {'davis': {'fmeasure': 0.8751047167822349, 'mae': 0.03059594151109138}} 20000
# {'FBMS': {'mae': 0.05827975689206127, 'fmeasure': 0.8395263230105977}} 20000
# {'MCL': {'fmeasure': 0.7897838103776252, 'mae': 0.0341193568105776}} 20000

# VideoSaliency_2019-07-04 00:05:22 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-05-14 17:13:16 model:resnext101 + R3Net + no motion block + motion enhancement + no saliency guide block
# have self_attention
# {'davis': {'fmeasure': 0.874, 'mae': 0.0306}} 20000
# {'FBMS': {'mae': 0.841, 'fmeasure': 0.0573}} 20000
# {'MCL': {'fmeasure': 0.7978824584120152, 'mae': 0.03270654914596075}} 20000

# {'FBMS': {'mae': 0.05898730165783062, 'fmeasure': 0.8394230953578746}} 10000

# VideoSaliency_2019-07-01 22:14:16 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-05-14 17:13:16 model:resnext101 + R3Net + no motion block + motion enhancement
# have self_attention
# {'davis': {'mae': 0.030584487861882274, 'fmeasure': 0.8761062884262997}}
# {'FBMS': {'mae': 0.05905227985899478, 'fmeasure': 0.8397254002559016}}
# {'MCL': {'fmeasure': 0.7899358517201228, 'mae': 0.033703267332709085}}

# VideoSaliency_2019-07-01 17:32:33 traning from original resnet101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-28 22:46:18 model:resnet101 + R3Net + GRU + motion enhancement + saliency guide block
# no self_attention
# {'davis': {'fmeasure': 0.8502764930743375, 'mae': 0.026479752498128697}}

# VideoSaliency_2019-07-02 04:18:40 traning from original resnet101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-07-01 23:50:14 model:resnet101 + R3Net + GRU + motion enhancement + saliency guide block
# have self_attention
# {'davis': {'mae': 0.02984727148613444, 'fmeasure': 0.8756064480635604}}
# {'MCL': {'fmeasure': 0.7657679645629271, 'mae': 0.033812008388706564}}
# {'FBMS': {'mae': 0.05899833031167441, 'fmeasure': 0.829591156478791}}

# VideoSaliency_2019-07-02 04:10:31 traning from original resnet50 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-07-01 23:51:39 model:resnet50 + R3Net + GRU + motion enhancement + saliency guide block
# have self_attention
# {'davis': {'fmeasure': 0.8572470140650937, 'mae': 0.03127575596579786}}
# {'FBMS': {'mae': 0.065907850117285, 'fmeasure': 0.8169151015700682}}
# {'MCL': {'fmeasure': 0.7650784843213123, 'mae': 0.03801076001268533}}


# VideoSaliency_2019-07-02 17:47:38 traning from original resnext50 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-07-02 04:21:40 model:resnext50 + R3Net + GRU + motion enhancement + saliency guide block
# have self_attention
# {'davis': {'fmeasure': 0.867985179773024, 'mae': 0.031125308718449575}}
# {'FBMS': {'mae': 0.05804739949561263, 'fmeasure': 0.8342736409382326}}
# {'MCL': {'fmeasure': 0.7773786755646357, 'mae': 0.034323540436367886}}

# VideoSaliency_2019-07-02 17:43:09: no finetune, only resnext101
# using dataset:DUT-TR + DAVIS R3Net pre-train
# {'davis': {'fmeasure': 0.8653577036343185, 'mae': 0.04190723255244401}} 20000
# {'davis': {'mae': 0.04510454890929087, 'fmeasure': 0.8482913046852011}} 15000
# {'FBMS': {'mae': 0.060115473676380815, 'fmeasure': 0.8370974160080146}} 15000
# {'MCL': {'fmeasure': 0.7719638752799074, 'mae': 0.03398585222840852}} 20000
# {'MCL': {'fmeasure': 0.7643802009424475, 'mae': 0.037740759818777427}} 15000
# {'SegTrackV2': {'fmeasure': 0.8833902740630123, 'mae': 0.019613184126835628}} 20000
# {'SegTrackV2': {'fmeasure': 0.8661164437880786, 'mae': 0.022897514345448376}} 15000
# {'VOS': {'fmeasure': 0.7689440285547244, 'mae': 0.06967526071368603}} 20000

# VideoSaliency_2019-07-02 23:41:10: no finetune, only resnext101 no self_attention
# using dataset:DUT-TR + DAVIS R3Net pre-train
# {'FBMS': {'fmeasure': 0.8408206546850922, 'mae': 0.05546505643440736}} 30000
# {'MCL': {'mae': 0.03155394830915426, 'fmeasure': 0.7538068242918796}} 30000

# VideoSaliency_2019-07-02 23:42:38, only resnext101 no self_attention
# using dataset:DUT-OMRON + DAVIS R3Net pre-train
# {'davis': {'fmeasure': 0.8566587089872602, 'mae': 0.027955859338707836}} 30000
# {'davis': {'mae': 0.03151518507370385, 'fmeasure': 0.8296323204360312}} 20000
# {'FBMS': {'fmeasure': 0.8275514791821478, 'mae': 0.05934688602250718}} 30000
# {'FBMS': {'fmeasure': 0.8191922301416684, 'mae': 0.062295793605747364}}

# VideoSaliency_2019-07-03 17:36:37 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-05-14 17:13:16 model:resnext101 + R3Net + LSTM + motion enhancement + saliency guide block
# {'MCL': {'fmeasure': 0.7919051602536508, 'mae': 0.033465254438458325}}
# {'davis': {'fmeasure': 0.8739062844885623, 'mae': 0.03299489011624521}}
# {'FBMS': {'fmeasure': 0.8381842471006478, 'mae': 0.058207687328754}}

# VideoSaliency_2019-08-08 18:04:10 traning from original resnet101 model of torch parameter, using dataset:DUT-TR + DAVIS
# model:resnet101 + DSS
# {'davis': {'fmeasure': 0.8483, 'mae': 0.03583}}

# VideoSaliency_2019-08-08 23:13:19 traning from original resnet101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-08-08 18:04:10;  model:resnet101 + DSS + MEN
# {'davis': {'fmeasure': 0.8483, 'mae': 0.03472}}

# VideoSaliency_2019-08-09 23:13:08 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-26 00:07:16;  model:resnext101 + R3Net + GRU + motion enhancement + saliency guide block + PAM(self attention)
# {'davis': {'fmeasure': 0.8801, 'mae': 0.02831}} 30000
# {'davis': {'fmeasure': 0.8840, 'mae': 0.02761}} 20000


# VideoSaliency_2019-08-11 05:19:41 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-26 00:07:16;  model:resnext101 + R3Net + GRU + motion enhancement + saliency guide block + STA(self attention)
# {'davis': {'fmeasure': 0.8834, 'mae': 0.02841}} 30000
# {'davis': {'fmeasure': 0.8873, 'mae': 0.02731}} 25000
# {'davis': {'fmeasure': 0.8871, 'mae': 0.02630}} 20000
# {'davis': {'fmeasure': 0.8843, 'mae': 0.02849}} 10000

# VideoSaliency_2019-08-11 19:39:25 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-26 00:07:16;  model:resnext101 + R3Net + GRU + motion enhancement + saliency guide block + STA(self attention)
# {'davis': {'fmeasure': 0.8829, 'mae': 0.02765}} 30000
# {'davis': {'fmeasure': 0.8859, 'mae': 0.02593}} 20000

# VideoSaliency_2019-08-18 05:45:28 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-26 00:07:16;  model:resnext101 + R3Net + GRU + motion enhancement + saliency guide block + STA(self attention)
# {'DAVSOD': {'fmeasure': 0.5875760813843472, 'mae': 0.09476998692081383}} 20000
# {'SegTrackV2': {'mae': 0.02389494685134959, 'fmeasure': 0.8824181699450439}}
# {'VOS': {'mae': 0.07287949195973144, 'fmeasure': 0.7635497228309376}}
# {'ViSal': {'mae': 0.015052299736749774, 'fmeasure': 0.9492640942370092}}
# {'UVSD': {'fmeasure': 0.6981638464285685, 'mae': 0.037293701491150306}}
# {'MCL': {'fmeasure': 0.7802265038605428, 'mae': 0.034676702495845035}}

# VideoSaliency_2019-10-18 15:57:27 used for validate the LSTM senario
# finetune VideoSaliency_2019-06-26 00:07:16;  model:resnext101 + R3Net + LSTM + motion enhancement + saliency guide block + STA(self attention)
# {'davis': {'fmeasure': 0.8819791669234219, 'mae': 0.027835531901318836}}


# VideoSaliency_2019-10-18 16:27:15 used for replacing mutual attention with addition
# finetune VideoSaliency_2019-06-26 00:07:16;  model:resnext101 + R3Net + GRU + motion enhancement + saliency guide block + STA(self attention)
# {'davis': {'fmeasure': 0.884, 'mae': 0.0263}} 20000
# {'davis': {'fmeasure': 0.8830341793022286, 'mae': 0.02762751428279943}} 15000
# {'davis': {'fmeasure': 0.8818999449299334, 'mae': 0.028222377521135975}} 25000
# {'davis': {'fmeasure': 0.8822463482353269, 'mae': 0.027449925224086794}} 30000

# VideoSaliency_2019-10-18 22:15:32 used for replacing mutual attention with concat
# finetune VideoSaliency_2019-06-26 00:07:16;  model:resnext101 + R3Net + GRU + motion enhancement + saliency guide block + STA(self attention)
# {'davis': {'fmeasure': 0.8783563016062568, 'mae': 0.028543348960840154}} 30000

# VideoSaliency_2019-10-18 22:15:32 used for replacing mutual attention with multipy
# finetune VideoSaliency_2019-06-26 00:07:16;  model:resnext101 + R3Net + GRU + motion enhancement + saliency guide block + STA(self attention)
# {'davis': {'fmeasure': 0.880557849206916, 'mae': 0.027870731202063934}} 30000

# VideoSaliency_2019-08-25 18:55:50
# model:resnet50 + R3Net + GRU + motion enhancement + saliency guide block + STA(self attention)
# {'VOS': {'fmeasure': 0.7203863443068934, 'mae': 0.08559279067970993}}
# {'SegTrackV2': {'fmeasure': 0.8051851775515824, 'mae': 0.026605373713406655}}
# {'ViSal': {'fmeasure': 0.9259855810752392, 'mae': 0.026381948506201736}}
# {'FBMS': {'fmeasure': 0.8225627461536053, 'mae': 0.06362138353610897}}
# {'DAVSOD': {'fmeasure': 0.5397687767436796, 'mae': 0.09922930943792657}}

# VideoSaliency_2019-12-28 13:05:13 distill:0.3----- 50000: {'davis': {'fmeasure': 0.8810721035836645, 'mae': 0.022452792045184123}}
# VideoSaliency_2019-12-31 15:23:37 distill:0.5----- 50000: {'davis': {'fmeasure': 0.8782718878666532, 'mae': 0.023935075017862996}}
# VideoSaliency_2019-12-31 22:42:12 distill:0.7----- 70000: {'davis': {'fmeasure': 0.8812928597783427, 'mae': 0.022111233533872954}}
#                                     predict2 ----- 70000: {'davis': {'fmeasure': 0.8816722907747716, 'mae': 0.021336993669902524}}
#                                                           {'DAVSOD': {'fmeasure': 0.6134204987014114, 'mae': 0.0781959343906902}}
#                                                           {'VOS': {'fmeasure': 0.7443896748569679, 'mae': 0.07604991489270245}}

# VideoSaliency_2019-12-31 22:42:55 distill:0.1----- 70000: {'davis': {'fmeasure': 0.872654589151305, 'mae': 0.024006023195154613}}
# VideoSaliency_2020-01-01 10:11:15 distill:0.0----- 50000: {'davis': {'fmeasure': 0.8626085101794334, 'mae': 0.029987119009663856}}
#                                              ----- 60000: {'davis': {'fmeasure': 0.876864323046619, 'mae': 0.023919671771285032}}
# VideoSaliency_2020-01-07 22:22:22 distill:0.0----- 50000: {'davis': {'fmeasure': 0.8520706484386346, 'mae': 0.03443576234542937}}
#                                              ----- 60000: {'davis': {'fmeasure': 0.8720645511026007, 'mae': 0.023902326790021484}}
#                                              ----- 40000: {'davis': {'fmeasure': 0.8689895052201964, 'mae': 0.026248430934317772}}
#                                              ----- 30000: {'davis': {'fmeasure': 0.8596129612450853, 'mae': 0.0281209143096176}}
# VideoSaliency_2020-01-01 10:12:55 distill:0.9----- 70000: {'davis': {'fmeasure': 0.87781728386952, 'mae': 0.022213421130818912}}
# VideoSaliency_2020-01-03 16:14:20 distill:1

# VideoSaliency_2020-01-05 11:10:39 attention
# VideoSaliency_2020-01-05 11:13:53 concat ----- 70000:{'davis': {'fmeasure': 0.874450331579124, 'mae': 0.02862872375746112}}
# VideoSaliency_2020-01-05 22:11:28 addition ----- 80000: {'davis': {'fmeasure': 0.872093570105392, 'mae': 0.024995046238766545}}
# VideoSaliency_2020-01-05 22:12:22 multiply ----- 80000: {'davis': {'fmeasure': 0.8826515249806051, 'mae': 0.023024765630256083}}
#                                   predict2 ----- 80000: {'davis': {'fmeasure': 0.8821976390867491, 'mae': 0.023354431959553126}}
# VideoSaliency_2020-01-06 11:31:22 dual attention

# VideoSaliency_2020-01-06 15:09:37 no spatial and temporal distiall
# 80000: {'davis': {'fmeasure': 0.8640213544007063, 'mae': 0.02839769598587939}}
# 70000: {'davis': {'fmeasure': 0.8600419157804461, 'mae': 0.03167843389102999}}
# 60000: {'davis': {'fmeasure': 0.8623502822129258, 'mae': 0.03036184826043933}}
# 50000: {'davis': {'fmeasure': 0.858386289404193, 'mae': 0.03152209046305965}}
# 40000: {'davis': {'fmeasure': 0.8556782342746729, 'mae': 0.036276497675533204}}
# 30000: {'davis': {'fmeasure': 0.8600069372484436, 'mae': 0.03465928481551954}}
# 20000: {'davis': {'fmeasure': 0.8653211022240138, 'mae': 0.03323024623319115}}
# 10000: {'davis': {'fmeasure': 0.8209345686892037, 'mae': 0.04836898942506064}}
# VideoSaliency_2020-01-08 18:27:34 no spatial and temporal distiall
# 10000: {'davis': {'fmeasure': 0.8591739367530228, 'mae': 0.036552687581060936}}
# 15000: {'davis': {'fmeasure': 0.8636745673434826, 'mae': 0.03129528438011365}}
# 20000: {'davis': {'fmeasure': 0.8692893256323055, 'mae': 0.03159149273351484}}
# VideoSaliency_2020-01-08 21:21:28 no spatial and temporal distiall
# 9000: {'davis': {'fmeasure': 0.8593935396571305, 'mae': 0.03894153003684641}}
# 6000: {'davis': {'fmeasure': 0.8553050938266464, 'mae': 0.03874729345329752}}
# 3000: {'davis': {'fmeasure': 0.8381087691514476, 'mae': 0.04080485946642285}}
# VideoSaliency_2020-01-09 12:26:16 no spatial and temporal distiall
# 2000: {'davis': {'fmeasure': 0.842206969701165, 'mae': 0.04409032292327435}}
# 10000:{'davis': {'fmeasure': 0.8563717610817255, 'mae': 0.036414058222809416}}

# VideoSaliency_2020-01-06 15:09:37 no temporal distiall, have spatial temporal
# 60000: {'davis': {'fmeasure': 0.8582169284445543, 'mae': 0.036224856077541896}}
# 70000: {'davis': {'fmeasure': 0.8606752869050858, 'mae': 0.03147308904301015}}
# 80000: {'davis': {'fmeasure': 0.861378091587172, 'mae': 0.029567548031849366}}
# VideoSaliency_2020-01-08 15:45:02 no temporal distiall, have spatial temporal
# 10000: {'davis': {'fmeasure': 0.8543948851987682, 'mae': 0.03211852639414837}}
# 15000: {'davis': {'fmeasure': 0.8637713047061395, 'mae': 0.027165545309932034}}
# 20000: {'davis': {'fmeasure': 0.8713080615066242, 'mae': 0.024770326433407278}}

# VideoSaliency_2020-01-07 22:16:46 have spatial and temporal distiall, no mutual
# 80000: {'davis': {'fmeasure': 0.875925295857492, 'mae': 0.02309260167860989}}
# 70000: {'davis': {'fmeasure': 0.8776447358697721, 'mae': 0.022962033893014596}}
# 60000: {'davis': {'fmeasure': 0.87650231578808, 'mae': 0.023123045813350132}}
# 50000: {'davis': {'fmeasure': 0.8697937192108691, 'mae': 0.02638709378286041}}
# 40000: {'davis': {'fmeasure': 0.8580790523452594, 'mae': 0.027347441253037293}}

# VideoSaliency_2020-01-10 12:49:12 DUT
# {'VOS': {'fmeasure': 0.7547528782992113, 'mae': 0.06952151973728524}}
# {'DAVSOD': {'fmeasure': 0.5982167488429656, 'mae': 0.08669203875787684}}

# VideoSaliency_2020-01-10 23:30:18 DUT_TS
# {'DAVSOD': {'fmeasure': 0.6115132471036877, 'mae': 0.0790320177381459}}
# {'VOS': {'fmeasure': 0.7453595217469079, 'mae': 0.07865791524659582}}

# VideoSaliency_2020-01-11 14:56:54 DUT_TS + DUT finetune
# 20000: {'DAVSOD': {'fmeasure': 0.613637847210674, 'mae': 0.08187658859414605}}
# 20000: {'VOS': {'fmeasure': 0.7555144239844935, 'mae': 0.0713882826002036
# 20000: {'SegTrackV2': {'fmeasure': 0.8442080785171779, 'mae': 0.025581302479766355}}
# 20000: {'ViSal': {'fmeasure': 0.943108796516877, 'mae': 0.01782009968616656}}
# 20000: {'MCL': {'fmeasure': 0.7723660689897658, 'mae': 0.030471464104830384}}
# 20000: {'davis': {'fmeasure': 0.8807960657166323, 'mae': 0.021706777064460316}}
# 20000: {'FBMS': {'fmeasure': 0.8297292910986392, 'mae': 0.05413035079743894}}

# VideoSaliency_2020-01-11 22:29:06 DUT_TS + DUT finetune
# 50000: {'SegTrackV2': {'fmeasure': 0.8474721951491387, 'mae': 0.024745449083349934}}
# 50000: {'ViSal': {'fmeasure': 0.9432139322103857, 'mae': 0.017552129693787957}}
# 50000: {'VOS': {'fmeasure': 0.7521167957665365, 'mae': 0.07120552805611176}}
# 50000: {'DAVSOD': {'fmeasure': 0.6122169157403674, 'mae': 0.08443723174844969}}
# 50000: {'FBMS': {'fmeasure': 0.8312779584927306, 'mae': 0.054986611554781004}}
# 50000: {'davis': {'fmeasure': 0.882543029953261, 'mae': 0.02180807616404994}}
# 50000: seq=True {'davis': {'fmeasure': 0.8803435750454938, 'mae': 0.023505580464211664}}