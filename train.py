import torch

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
import numpy as np
import os
import cv2
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from dataload import Dataset_FFPN
from FFPN import FFPN
import torch.optim as optim
import json
from medpy import metric
import time

def d2c(dice):
    # dice /= 100
    if dice != 0:
        con = (3 * dice - 2) / dice
    else:
        con = -100.0
    return con*100

def fourier2contour(fourier, locations, samples=64, sampling=None):
    """

    Args:
        fourier: Array[..., order, 4]
        locations: Array[..., 2]
        samples: Number of samples.
        sampling: Array[samples] or Array[(fourier.shape[:-2] + (samples,)].
            Default is linspace from 0 to 1 with `samples` values.

    Returns:
        Contours.
    """
    order = fourier.shape[-2]
    sampling = np.linspace(0, 1.0, samples)
    samples = sampling.shape[-1]
    sampling = sampling[..., None, :]

    c = float(np.pi) * 2 * (np.arange(1, order + 1)[..., None]) * sampling

    c_cos = np.cos(c)
    c_sin = np.sin(c)

    con = np.zeros(fourier.shape[:-2] + (samples, 2))
    con += locations[..., None, :]
    con += (fourier[..., None, (1, 3)] * c_sin[(...,) + (None,) * 1]).sum(-3)
    con += (fourier[..., None, (0, 2)] * c_cos[(...,) + (None,) * 1]).sum(-3)
    return con


def update_masks(mask, label_id = [3]):
    def update_mask(labels):
        contours_1,_ = cv2.findContours(labels,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contour_1 = max(contours_1, key=len)
        mask_1 = np.zeros_like(labels)
        labels = cv2.drawContours(mask_1, [max_contour_1],-1,(1,1,1),-1)
        labels = (labels > 0).astype('uint8') 
        return labels
    labels = []
    n = 1
    for li in label_id:
        tmp_label = (mask == li).astype('uint8')
        tmp_label = update_mask(tmp_label) *n
        n += 1
        labels.append(tmp_label)
    return labels

def as_numpy(input):
    out_list = []
    for i in input:
        out_list.append(i.cpu().detach().numpy())
    return out_list
    

def eavl_results(model, test_loader, device, epoch, class_num = 1, label_ids = [3]):
    model.eval()
    run_time_all = []
    sum_iou = []
    sum_dice = []
    sum_conf = []
    sum_hd = []
    dst_result = {}
    metric_class = {}
    for n in range(class_num):
        metric_class[str(n+1)] = {'dice':[], 'hd':[], 'iou':[], 'conf':[]}

    # batch = cd.to_device(next(iter(test_loader)), device)
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Test Epoch %d" % epoch)):
        image, masks, obj_labels, fouriers, locations, contours, img_path_list, mask_path_list = batch
        image = image.to(device)
        masks = masks.to(device)
        # obj_labels = obj_labels.to(device)
        fouriers = fouriers.to(device)
        locations = locations.to(device)
        contours = contours.to(device)
        with torch.no_grad():
            torch.cuda.synchronize()
            time_start = time.time()
            _, outputs = model(image)
            torch.cuda.synchronize()
            time_end = time.time()
            run_time = time_end - time_start
            run_time_all.append(run_time)

        if outputs is None:
            continue

        pred_labels = as_numpy(outputs['batched_labels'])
        pred_contours = as_numpy(outputs['batched_contours'])
        pred_fouriers = as_numpy(outputs['batched_fouriers'])
        pred_locations = as_numpy(outputs['batched_locations'])

        gt_labels = obj_labels.numpy()
        bs_num = len(pred_fouriers)

        for idx in range(bs_num):
            img_path = img_path_list[idx]
            mask_path = mask_path_list[idx]
            img_name = img_path.split('/')[-1]
            dst_result[img_name] = {}
            if len(pred_fouriers[idx]) == 0:
                continue
            
            img = cv2.imread(img_path, 1)
            shape = [img.shape[1],img.shape[0]]
            radio = np.array([ s/416 for s in shape],dtype=np.float32)
            masks = cv2.imread(mask_path,0)
            lables = update_masks(masks, label_ids)
            pred_label = pred_labels[idx].astype(np.int64)
            
            
            for pred_num in range(len(gt_labels[idx])):
                gl = gt_labels[idx][pred_num]
                
                gt_mask = lables[gl - 1]
                
                if np.sum(gl == pred_label) < 1:
                    metric_class[str(gl)]['dice'].append(0.0)
                    metric_class[str(gl)]['iou'].append(0.0)
                    metric_class[str(gl)]['conf'].append(0.0)
                    dst_result[img_name][str(gl)] = []
                    sum_dice.append(0.0)
                    sum_iou.append(0.0)
                    sum_conf.append(0.0)
                else:
                    pred_idx = gl == pred_label
                    if len(pred_contours[idx] != 0):
                        contour = (pred_contours[idx][pred_idx] * radio).astype(int)[0]
                    else:
                        fourier = pred_fouriers[idx][pred_idx][0]
                        location = pred_locations[idx][pred_idx][0]
                        contour = fourier2contour(fourier.reshape(-1,7,4), location.reshape(-1,2), 128)
                        contour = contour.reshape(-1, 2)
                        contour = (contour*radio).astype(int)
                    
                    dst_result[img_name][str(gl)] = contour.tolist()
            
                    # 单独在图上画一个预测的contour，用来测iou和dice
                    test_img = np.zeros_like(gt_mask)
                    test_img = cv2.drawContours(test_img, [contour],-1,(1,1,1),-1)
                    # 计算iou
                    intersection = np.sum(np.logical_and(test_img,gt_mask))
                    union = np.sum(np.logical_or(test_img,gt_mask))
                    iou = intersection / union
                    sum_iou.append(iou)
                    m1 = np.sum((test_img * gt_mask > 0).astype(np.uint8))
                    m2 = np.sum(test_img) + np.sum(gt_mask/gl)
                    dice = 2 * m1 / m2
                    sum_dice.append(dice)
                    confm = d2c(dice)
                    sum_conf.append(confm)
                    # 计算HD
                    hd95 = metric.binary.hd95(test_img, gt_mask)
                    sum_hd.append(hd95)
                    metric_class[str(gl)]['dice'].append(dice)
                    metric_class[str(gl)]['iou'].append(iou)
                    metric_class[str(gl)]['conf'].append(confm)
                    metric_class[str(gl)]['hd'].append(hd95)
                
    
    for cls, metric_ in metric_class.items():
        for m_name, infor in metric_.items():
            print(cls, m_name, np.mean(infor), np.std(infor)) 
        print('#######################') 
                    
    average_dice = np.mean(sum_dice)
    var_dice = np.std(sum_dice)
    average_conf = np.mean(sum_conf)
    var_conf = np.std(sum_conf)
    average_iou = np.mean(sum_iou)
    var_iou = np.std(sum_iou)
    average_hd = np.mean(sum_hd)
    var_hd = np.std(sum_hd)
    print('run_time_all',len(run_time_all))
    print("time:",np.mean(run_time_all[3:]))
    print('average_iou', average_iou, 'average_conf', average_conf)
    print('average_dice', average_dice, 'average_hd', average_hd)
    
    print('max_memory_allocated:{:.2f} M'.format(torch.cuda.max_memory_allocated(device = device)/1024/1024))
    dst_result['metric'] = metric_class
    json.dump(dst_result, open(f'metric.json','w', encoding='utf8'), ensure_ascii=False, indent=4)

    return average_dice, average_hd


def train_epoch(model, train_loader, device, optimizer, epoch, scheduler=None):
    
    weights = {'ob':1.0,
        'four': 1.0,
        'loc': 1.0,
        'cont':1.0,
        'refine':0.2}
    
    loss_config = {'ob':.0,
            'four': .0,
            'loc':.0,
            'cont':.0,
            'refine':.0}
    model.train()
    train_losses = 0
    pbar = tqdm(train_loader)
    for batch_idx, batch in enumerate(pbar):

        image, masks, obj_labels, fouriers, locations, contours, img_path, mask_path = batch
        image = image.to(device)
        masks = masks.to(device)
        obj_labels = obj_labels.to(device)
        fouriers = fouriers.to(device)
        locations = locations.to(device)
        contours = contours.to(device)
        bs = image.shape[0]
        target_list = []
        for b in range(bs):
            target_list.append({'labels':obj_labels[b],'fouriers':fouriers[b],'locations':locations[b],
                                'contours':contours[b],'masks':masks[b]})
        optimizer.zero_grad()
        
        losses, _ = model(image, targets=target_list)
        loss = 0
        for k, v in losses.items():
            if v is not None:
                loss += v * weights[k]
                loss_config[k] += v.item() * weights[k]
        ob_loss = round(losses['ob'].item(),2)
        four_loss = round(losses['four'].item(),2)

        pbar.set_description(f'Epoch:{epoch}; ob_loss:{ob_loss}; four_loss:{four_loss}')
        train_losses += loss
        loss.backward()
        optimizer.step()

    if scheduler is not None:
        scheduler.step()
    for k, v in loss_config.items():
        loss_config[k] = round(v/len(train_loader), 3)
    return round(float(train_losses) / len(train_loader), 5), loss_config


def main_FFPN_CSR(device = 'cuda:0'):
    train_data = Dataset_FFPN(order=7, images_infor='Camus_dataset.json', class_num = 3, label_id = [255,200,150], type = 'train')
    val_data = Dataset_FFPN(order=7, images_infor='Camus_dataset.json', class_num = 3, label_id = [255,200,150], type = 'val')
    train_loader = DataLoader(train_data, batch_size=8, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, num_workers=2)
    model = FFPN(is_refinement=True, classes=4, order=7, sample_num=128, anchor_file_path='anchor_files/Camus_dataset_anchor.json')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr = 1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9)
    epochs=200
    mdice = 0.0
    
    for epoch in range(epochs):
        train_loss, loss_config = train_epoch(model, train_loader, device, optimizer, epoch, scheduler)  
        print('train_loss', train_loss)
        average_dice, average_hd = eavl_results(model, val_loader, device, epoch, class_num = 3, label_ids = [255,200,150])
        if average_dice >= mdice:
            mdice = average_dice
            torch.save({'model':model.state_dict(), 'dice':float(average_dice), 'hd':float(average_hd), 'epoch':epoch},
                    os.path.join('best_ffpn_csr.pth'))

def main_FFPN(device = 'cuda:0'):
    train_data = Dataset_FFPN(order=7, images_infor='2CH_dataset.json', class_num = 1, label_id = [3], type = 'train')
    val_data = Dataset_FFPN(order=7, images_infor='2CH_dataset.json', class_num = 1, label_id = [3], type = 'val')
    train_loader = DataLoader(train_data, batch_size=8, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=1)
    model = FFPN(is_refinement=False, classes=2, order=7, sample_num=128, anchor_file_path='anchor_files/2CH_dataset_anchor.json')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr = 1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9)
    epochs=200
    mdice = 0.0
    
    for epoch in range(epochs):
        train_loss, loss_config = train_epoch(model, train_loader, device, optimizer, epoch, scheduler)  
        print('train_loss', train_loss)
        average_dice, average_hd = eavl_results(model, val_loader, device, epoch, class_num = 1, label_ids = [3], is_inference = False)
        if average_dice >= mdice:
            mdice = average_dice
            torch.save({'model':model.state_dict(), 'dice':float(average_dice), 'hd':float(average_hd), 'epoch':epoch},
                    os.path.join('best_ffpn.pth'))


if __name__ =='__main__':
    main_FFPN_CSR()