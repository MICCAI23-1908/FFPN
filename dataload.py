import os
import glob
import json
import random
from turtle import forward
import cv2
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

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

def efd(contour, order=10, epsilon=1e-6):
    """Elliptic fourier descriptor.

    Computes elliptic fourier descriptors from contour data.

    Args:
        contour: Tensor of shape (..., num_points, 2). Should be set of `num_points` 2D points that describe the contour
            of an object. Based on each contour a descriptor of shape (order, 4) is computed. The result has thus
            a shape of (..., order, 4).
            As `num_points` may differ from one contour to another a list of (num_points, 2) arrays may be passed
            as a numpy array with `object` as its data type, i.e. `np.array(list_of_contours)`.
        order: Order of resulting descriptor. The higher the order, the more detail can be preserved. An order of 1
            produces ellipses.
        epsilon: Epsilon value. Used to avoid division by zero.

    Notes:
        Locations may contain NaN if `contour` only contains a single point.

    Returns:
        Tensor of shape (..., order, 4).
    """
    if isinstance(contour, np.ndarray) and contour.dtype == object:
        r = [efd(c, order=order, epsilon=epsilon) for c in contour]
        if all([isinstance(r_, tuple) and len(r_) == len(r[0]) for r_ in r]):
            res = [[] for _ in range(len(r[0]))]
            for r_ in r:
                for i in range(len(res)):
                    res[i].append(r_[i])
            return tuple(map(np.array, res))
    dxy = np.diff(contour, axis=-2)  # shape: (..., p, d)
    dt = np.sqrt(np.sum(np.square(dxy), axis=-1)) + epsilon  # shape: (..., p)
    cumsum = np.cumsum(dt, axis=-1)  # shape: (..., p)
    zero = np.zeros(cumsum.shape[:-1] + (1,))
    t = np.concatenate([zero, cumsum], axis=-1)  # shape: (..., p + 1)
    sampling = t[..., -1:]  # shape: (..., 1)
    T_ = t[..., -1]  # shape: (...,)
    phi = (2 * np.pi * t) / sampling  # shape: (..., p + 1)
    orders = np.arange(1, order + 1, dtype=phi.dtype)  # shape: (order,)
    constants = sampling / (2. * np.square(orders) * np.square(np.pi))
    phi = np.expand_dims(phi, -2) * np.expand_dims(orders, -1)
    d_cos_phi = np.cos(phi[..., 1:]) - np.cos(phi[..., :-1])
    d_sin_phi = np.sin(phi[..., 1:]) - np.sin(phi[..., :-1])

    dxy0_dt = np.expand_dims(dxy[..., 0] / dt, axis=-2)
    dxy1_dt = np.expand_dims(dxy[..., 1] / dt, axis=-2)
    coefficients = np.stack([
        constants * np.sum(dxy0_dt * d_cos_phi, axis=-1),
        constants * np.sum(dxy0_dt * d_sin_phi, axis=-1),
        constants * np.sum(dxy1_dt * d_cos_phi, axis=-1),
        constants * np.sum(dxy1_dt * d_sin_phi, axis=-1),
    ], axis=-1)

    xi = np.cumsum(dxy[..., 0], axis=-1) - (dxy[..., 0] / dt) * t[..., 1:]
    delta = np.cumsum(dxy[..., 1], axis=-1) - (dxy[..., 1] / dt) * t[..., 1:]
    t_diff = np.diff(t ** 2, axis=-1)
    dt2 = 2 * dt
    a0 = (1 / T_) * np.sum(((dxy[..., 0] / dt2) * t_diff) + xi * dt, axis=-1)
    c0 = (1 / T_) * np.sum(((dxy[..., 1] / dt2) * t_diff) + delta * dt, axis=-1)
    return np.array(coefficients), np.stack((contour[..., 0, 0] + a0, contour[..., 0, 1] + c0), axis=-1)


class Dataset_FFPN(Dataset):
    def __init__(self, order=7, images_infor='xxx.json', class_num = 1, label_id = [255], type = 'train'):
        super(Dataset_FFPN, self).__init__()
        self.order = order
        self.class_num = class_num
        self.label_id = label_id
        img_infor = json.load(open(images_infor, encoding='utf8'))
        self.images_list = img_infor[type]['img_list'][:16]
        self.masks_list = img_infor[type]['mask_list'][:16]
        print('img_list', len(self.images_list),'self.order', self.order)

    def __len__(self):
        return len(self.images_list)

    @staticmethod
    def map(image):
        image = image / 127.5
        image -= 1
        return image

    @staticmethod
    def unmap(image):
        image = (image + 1) * 127.5
        image = np.clip(image, 0, 255).astype('uint8')
        return image

    def update_mask(self, labels):
        contours_1,_ = cv2.findContours(labels,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contour_1 = max(contours_1, key=len)
        mask_1 = np.zeros_like(labels)
        labels = cv2.drawContours(mask_1, [max_contour_1],-1,(1,1,1),-1)
        labels = (labels > 0).astype('uint8') 
        return labels

    def generate_c_f_l(self, labels):
        contours, _ = cv2.findContours(labels,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        four, loc =  efd(contours[0].reshape(-1,2), self.order)
        sample_contour = fourier2contour(four, loc, 128)
        return four.reshape(-1, self.order, 4), loc.reshape(-1, 2), sample_contour.reshape(-1, 128,2)

    def __getitem__(self, item):

        img_path = self.images_list[item]
        mask_path = self.masks_list[item]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (416,416),interpolation=cv2.INTER_LINEAR)
        labels = cv2.imread(mask_path, 0)
        labels = cv2.resize(labels,(416,416),interpolation=cv2.INTER_NEAREST)
        
        
        obj_labels = []
        fouriers = []
        locations = []
        contours = []
        masks = np.zeros_like(labels)
        for n in range(self.class_num):
            tmp_mask = (labels == self.label_id[n]).astype('uint8')
            tmp_mask = self.update_mask(tmp_mask)
            tmp_foureier, tmp_loc, tmp_contours = self.generate_c_f_l(tmp_mask)
            fouriers.append(tmp_foureier)
            locations.append(tmp_loc)
            contours.append(tmp_contours)
            obj_labels.append(n+1)
            masks += tmp_mask * (n+1)
            
        
        fouriers = np.concatenate([fouriers], axis=0) # 1
        locations = np.concatenate([locations], axis=0)
        contours = np.concatenate([contours], axis=0)
        fouriers = torch.as_tensor(fouriers).squeeze(1).to(torch.float32)
        locations = torch.as_tensor(locations).squeeze(1).to(torch.float32)
        contours = torch.as_tensor(contours).squeeze(1).to(torch.float32)


        image = self.map(img)
        image = torch.as_tensor(image).permute(2,0,1).to(torch.float32)
        obj_labels = torch.tensor(obj_labels)
        return image, masks, obj_labels, fouriers, locations, contours, img_path, mask_path
        
        
    

if __name__ == '__main__':
    dataset = Dataset_FFPN(order=7, images_infor='Camus_dataset.json', class_num = 3, label_id = [255,200,150], type = 'train')
    train_loader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=True)
    for idx, data in enumerate(train_loader):
        image, masks, obj_labels, fouriers, locations, contours, img_path, mask_path = data
        print(image.shape)
        print(masks.shape)
        print(obj_labels.shape)
        print(fouriers.shape)
        print(locations.shape)
        print(contours.shape)
        print(len(img_path))