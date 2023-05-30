from resnet import ResNet_50_FPN, ReadOut, ReadOut_refine
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops
import torchvision.ops.roi_align as roi_f
import json
import numpy as np
from basemodule import *


class FFPN_Core(nn.Module):
    def __init__(
            self,
            anchor_num: int = 9,
            order = 7,
            classes: int = 2,
            is_refinement: bool = True):
        super(FFPN_Core, self).__init__()
        self.order = order
        self.is_refinement = is_refinement
        self.backbone = ResNet_50_FPN()
        backbone_out_channels = self.backbone.out_channels
        self.score_head_small = ReadOut(backbone_out_channels, classes*anchor_num,
                                        kernel_size=3, padding=1, channels_mid=backbone_out_channels, stride=2)
        self.score_head_middle = ReadOut(backbone_out_channels, classes*anchor_num,
                                        kernel_size=3, padding=1, channels_mid=backbone_out_channels, stride=2)
        self.score_head_large = ReadOut(backbone_out_channels, classes*anchor_num,
                                        kernel_size=3, padding=1, channels_mid=backbone_out_channels, stride=2)

        self.location_head_small = ReadOut(
            backbone_out_channels, 2*anchor_num,
            kernel_size=3,
            padding=1,
            channels_mid=backbone_out_channels,
            stride=2
        )
        self.location_head_middle = ReadOut(
            backbone_out_channels, 2*anchor_num,
            kernel_size=3,
            padding=1,
            channels_mid=backbone_out_channels,
            stride=2
        )
        self.location_head_large = ReadOut(
            backbone_out_channels, 2*anchor_num,
            kernel_size=3,
            padding=1,
            channels_mid=backbone_out_channels,
            stride=2
        )
        
        self.fourier_head_small = ReadOut(
            backbone_out_channels, (self.order - 3) * 4*anchor_num,
            kernel_size=3,
            padding=1,
            channels_mid=backbone_out_channels,
            stride=2
        )
        self.fourier_head_middle = ReadOut(
            backbone_out_channels, 4 * 2*anchor_num,
            kernel_size=3,
            padding=1,
            channels_mid=backbone_out_channels,
            stride=2
        )
        self.fourier_head_large = ReadOut(
            backbone_out_channels, 4 * 1*anchor_num,
            kernel_size=3,
            padding=1,
            channels_mid=backbone_out_channels,
            stride=2
        )
        if self.is_refinement:
            self.fuse_smooth = nn.Sequential(nn.Conv2d(backbone_out_channels*3,256,kernel_size=3,stride=1,padding=1, bias=False),\
                nn.BatchNorm2d(256),nn.ReLU(inplace=True))
            self.fuse_conv = ReadOut_refine(
                256, 64,
                kernel_size=3,
                padding=1,
                channels_mid = 256,
                stride=1
            )
    
    def forward(self, inputs):
        features = self.backbone(inputs)
        fouriers_feature_small = features[1] # 
        fouriers_feature_middle = features[2] # 
        fouriers_feature_large = features[3] # 
        base_shape = fouriers_feature_small.shape[-2:]
        fouriers_feature_middle = F.interpolate(fouriers_feature_middle, base_shape, mode='bilinear', align_corners=True)
        fouriers_feature_large = F.interpolate(fouriers_feature_large, base_shape, mode='bilinear', align_corners=True)
        if self.is_refinement:
            fuse_feature = torch.cat([fouriers_feature_small, fouriers_feature_middle, fouriers_feature_large], dim=1)
            fuse_refine_feature = self.fuse_conv(self.fuse_smooth(fuse_feature))
        else:
            fuse_refine_feature = None
        
        scores_small = self.score_head_small(fouriers_feature_small)
        scores_middle = self.score_head_middle(fouriers_feature_middle)
        scores_large = self.score_head_large(fouriers_feature_large)
        
        fouriers_small = self.fourier_head_small(fouriers_feature_small)
        fouriers_middle = self.fourier_head_middle(fouriers_feature_middle)
        fouriers_large = self.fourier_head_large(fouriers_feature_large)
        
        locations_small = self.location_head_small(fouriers_feature_small)
        locations_middle = self.location_head_middle(fouriers_feature_middle)
        locations_large = self.location_head_large(fouriers_feature_large)
        
        scores = [scores_large, scores_middle, scores_small]
        fourier = [fouriers_large, fouriers_middle, fouriers_small]
        locations = [locations_large, locations_middle, locations_small]
        
        return scores, locations, fourier, fuse_refine_feature

class ContourSampleRefineHead(nn.Module):
    """
    Standard heads for FPN-based models
    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
     1024  -> 512  -> 256
    """

    def __init__(self, in_channels, order):
        super().__init__()

        self.fc1 = nn.Linear(in_channels, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 4*order+2)

    def forward(self, x):
        x = torch.flatten(x, start_dim = 1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        reg_out = self.fc3(x)
        return reg_out

def fouriers2contours(fourier, locations, samples=64, sampling=None, cache: Dict[str, Tensor] = None,
                      cache_size: int = 16):
    """

    Args:
        fourier: Tensor[..., order, 4]
        locations: Tensor[..., 2]
        samples: Number of samples. Only used for default sampling, ignored otherwise.
        sampling: Sampling t. Default is linspace 0..1. Device should match `fourier` and `locations`.
        cache: Cache for initial zero tensors. When fourier shapes are consistent this can increase execution times.
        cache_size: Cache size.

    Returns:
        Contours.
    """

    
    order = fourier.shape[-2]
    d = fourier.device
    sampling_ = sampling
    if sampling is None:
        sampling = sampling_ = torch.linspace(0, 1.0, samples, device=d)
    samples = sampling.shape[-1]
    sampling = sampling[..., None, :]

    c = float(np.pi) * 2 * (torch.arange(1, order + 1, device=d)[..., None]) * sampling

    c_cos = torch.cos(c)
    c_sin = torch.sin(c)

    con = None
    con_shape = fourier.shape[:-2] + (samples, 2)
    con_key = str(tuple(con_shape) + (d,))
    if cache is not None:
        con = cache.get(con_key, None)
    if con is None:
        con = torch.zeros(fourier.shape[:-2] + (samples, 2), device=d)  # 40.1 ms for size (520, 696, 64, 2) to cuda
        if cache is not None:
            if len(cache) >= cache_size:
                del cache[next(iter(cache.keys()))]
            cache[con_key] = con
    con = con + locations[..., None, :]
    con += (fourier[..., None, (1, 3)] * c_sin[(...,) + (None,) * 1]).sum(-3)
    con += (fourier[..., None, (0, 2)] * c_cos[(...,) + (None,) * 1]).sum(-3)
    return con, sampling_

def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)#N,A,C,H,W
    layer = layer.permute(0, 3, 4, 1, 2)#
    layer = layer.reshape(N, -1, C)#N,H x W x A,C
    return layer

def concat_four_prediction_layers(four_cls, four_regression, loc_regression, order_list = [1,2,4]):
    four_cls_flattened = []
    four_regression_flattened = []
    loc_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for four_cls_per_level, four_regression_per_level, loc_regress_per_level, order in zip(
        four_cls, four_regression, loc_regression, order_list
    ):

        N, AxC, H, W = four_cls_per_level.shape#
        Ax4xorder = four_regression_per_level.shape[1]#
        A = Ax4xorder // (4*order) # 
        C = AxC // A # 
        four_cls_per_level = permute_and_flatten(
            four_cls_per_level, N, A, C, H, W#
        )
        four_cls_flattened.append(four_cls_per_level)

        four_regression_per_level = permute_and_flatten(
            four_regression_per_level, N, A, 4*order, H, W#
        )
        four_regression_flattened.append(four_regression_per_level)

        loc_regress_per_level = permute_and_flatten(
            loc_regress_per_level, N, A, 2, H, W#
        )
        
        loc_regression_flattened.append(loc_regress_per_level)
    order_sum = torch.as_tensor(order_list).sum()
    four_cls = torch.stack(four_cls_flattened, dim=0)#(fpn, b,h*w*a,numclass)
    four_regression = torch.cat(four_regression_flattened, dim=-1).reshape(-1, 4*order_sum)#
    loc_regress = torch.stack(loc_regression_flattened, dim=0).reshape(len(loc_regression_flattened), -1, 2)
    return four_cls, four_regression, loc_regress

def decode_refine_result(refine_off, fouiers_mean, locations_mean, order = 7, sampling_num = 128):
    all_num = refine_off.shape[0]
    # k,4*order+2, k,1
    ellipse_mean = fourier2ellipes_ab(fouiers_mean.reshape(-1,4)) # 
    ellipse_mean = ellipse_mean.reshape(-1, order, 2) +1e-5
    fouiers_mean = fouiers_mean.reshape(-1, order, 4)
    locations_mean = locations_mean.reshape(-1, 2)
    loc_offset = refine_off[:,:2]
    four_offset = refine_off[:, 2:].reshape(-1, order, 4)
    
    refine_locations = loc_offset * ellipse_mean[:,0,:] + locations_mean
    
    refine_fouriers_a =  four_offset[...,0] * ellipse_mean[...,0] + fouiers_mean[...,0]
    refine_fouriers_b =  four_offset[...,1] * ellipse_mean[...,0] + fouiers_mean[...,1]
    refine_fouriers_c =  four_offset[...,2] * ellipse_mean[...,1] + fouiers_mean[...,2]
    refine_fouriers_d =  four_offset[...,3] * ellipse_mean[...,1] + fouiers_mean[...,3]
    refine_fouriers = torch.stack([refine_fouriers_a,refine_fouriers_b,refine_fouriers_c,refine_fouriers_d], dim=-1)
    refine_contours, _ = fouriers2contours(refine_fouriers.reshape(all_num, -1, 4), \
        locations=refine_locations.reshape(all_num, 2), samples= sampling_num)
    return refine_fouriers, refine_locations, refine_contours

def encode_refine_target(fouiers_mean, locations_mean, target_fouriers, target_locations, order = 7):
    ellipse_mean = fourier2ellipes_ab(fouiers_mean.reshape(-1,4)) # 
    ellipse_mean = ellipse_mean.reshape(-1, order, 2) +1e-5
    fouiers_mean = fouiers_mean.reshape(-1, order, 4)
    locations_mean = locations_mean.reshape(-1, 2)
    target_fouriers = target_fouriers.reshape(-1, order, 4)
    target_locations = target_locations.reshape(-1, 2)
    
    delta_fouriers_a = (target_fouriers[..., 0] - fouiers_mean[...,0])/ellipse_mean[...,0]
    delta_fouriers_b = (target_fouriers[..., 1] - fouiers_mean[...,1])/ellipse_mean[...,0]
    delta_fouriers_c = (target_fouriers[..., 2] - fouiers_mean[...,2])/ellipse_mean[...,1]
    delta_fouriers_d = (target_fouriers[..., 3] - fouiers_mean[...,3])/ellipse_mean[...,1]
    
    delta_fouriers = torch.stack([delta_fouriers_a, delta_fouriers_b, delta_fouriers_c, delta_fouriers_d], dim=-1).reshape(-1, order*4)
    delta_location_x = (target_locations[:,0] - locations_mean[:,0]) / ellipse_mean[:,0,0]
    delta_location_y = (target_locations[:,1] - locations_mean[:,1]) / ellipse_mean[:,0,1]
    delta_location = torch.stack([delta_location_x, delta_location_y], dim=-1).reshape(-1, 2)
    return delta_fouriers, delta_location

def decoder(pred_four_deltas, pred_loc_deltas, fouiers_anchor):
    # pred_four_deltas bs*h*w*a, order*4
    # locations bs*h*w*a, 2
    # fouiers_anchor: list bs, h*w*a, 2+order*4
    fouiers_anchors = torch.cat(fouiers_anchor, dim=0) #bs*h*w*a, 2+order*4

    pred_four_deltas = pred_four_deltas.reshape(-1, 7, 4)
    pred_loc_deltas = pred_loc_deltas.reshape(-1, 2)
    location_anchors = fouiers_anchors[:,:2].reshape(-1, 2)# #bs*h*w*a, 2
    ellipse_anchors = fouiers_anchors[:,2:(2+7*2)].reshape(-1, 7, 2)
    fouiers_anchors = fouiers_anchors[:,(2+7*2)::].reshape(-1, 7, 4) # #bs*h*w*a, order*4
    pred_fouriers_a = pred_four_deltas[..., 0]*ellipse_anchors[...,0] + fouiers_anchors[...,0]
    pred_fouriers_b = pred_four_deltas[..., 1]*ellipse_anchors[...,0] + fouiers_anchors[...,1]
    pred_fouriers_c = pred_four_deltas[..., 2]*ellipse_anchors[...,1] + fouiers_anchors[...,2]
    pred_fouriers_d = pred_four_deltas[..., 3]*ellipse_anchors[...,1] + fouiers_anchors[...,3]
    pred_fouriers=  torch.stack([pred_fouriers_a, pred_fouriers_b, pred_fouriers_c, pred_fouriers_d], dim=-1).reshape(-1, 7*4)
    pred_locations_x = pred_loc_deltas[:,0]*ellipse_anchors[:,0,0] + location_anchors[:,0]
    pred_locations_y = pred_loc_deltas[:,1]*ellipse_anchors[:,0,1] + location_anchors[:,1]
    pred_locations =  torch.stack([pred_locations_x, pred_locations_y], dim=-1).reshape(-1, 2)

    return pred_fouriers, pred_locations

def encoder(gt_fouriers, gt_locations, fouiers_anchor):
    # gt_fouriers bs*h*w*a, order*4
    # locations bs*h*w*a, 2
    # fouiers_anchor: bs, h*w*a, 2+order*4
    fouiers_anchors = torch.cat(fouiers_anchor, dim=0).reshape(-1, 2+7*(4+2)) #bs*h*w*a, 2+order*4
    gt_fouriers = torch.cat(gt_fouriers, dim=0).reshape(-1, 7, 4)
    gt_locations = torch.cat(gt_locations, dim=0).reshape(-1, 2)
    locations_a = fouiers_anchors[:,:2].reshape(-1, 2)
    ellipse_a = fouiers_anchors[:,2:(2+7*2)].reshape(-1, 7, 2) + 1e-5
    fouiers_a = fouiers_anchors[:,(2+7*2):].reshape(-1, 7, 4)
    
    delta_fouriers_a = (gt_fouriers[..., 0] - fouiers_a[...,0])/ellipse_a[...,0]
    delta_fouriers_b = (gt_fouriers[..., 1] - fouiers_a[...,1])/ellipse_a[...,0]
    delta_fouriers_c = (gt_fouriers[..., 2] - fouiers_a[...,2])/ellipse_a[...,1]
    delta_fouriers_d = (gt_fouriers[..., 3] - fouiers_a[...,3])/ellipse_a[...,1]
    
    delta_fouriers = torch.stack([delta_fouriers_a, delta_fouriers_b, delta_fouriers_c, delta_fouriers_d], dim=-1).reshape(-1, 7*4)
    delta_location_x = (gt_locations[:,0] - locations_a[:,0]) / ellipse_a[:,0,0]
    delta_location_y = (gt_locations[:,1] - locations_a[:,1]) / ellipse_a[:,0,1]
    delta_location = torch.stack([delta_location_x, delta_location_y], dim=-1).reshape(-1, 2)
    return delta_fouriers, delta_location

def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[M, 4])
        boxes2 (Tensor[N, 4])

    Returns:
        iou (Tensor[M, N]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [M,N,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [M,N,2]

    wh = (rb - lt).clamp(min=0)  # [M,N,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [M,N]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def contour2box_iou(contour1, contour2):
    x11 = contour1[:,:,0].min(dim=1)[0]
    x12 = contour1[:,:,0].max(dim=1)[0]
    y11 = contour1[:,:,1].min(dim=1)[0]
    y12 = contour1[:,:,1].max(dim=1)[0]
    box1 = torch.stack([x11,y11,x12,y12], dim=-1)
    
    x21 = contour2[:,:,0].min(dim=1)[0]
    x22 = contour2[:,:,0].max(dim=1)[0]
    y21 = contour2[:,:,1].min(dim=1)[0]
    y22 = contour2[:,:,1].max(dim=1)[0]
    box2 = torch.stack([x21,y21,x22,y22], dim=-1)
    iou_matrix = box_iou(box1, box2)
    return iou_matrix

def contour_ious(gt_contours, anchor_contours, gt_locations=None, is_eye=False, is_trans = False):
    # gt_contours, M, 128 ,2
    # anchor_contours, N, 128, 2
    # is_eye: 表示输入的pred和gt的尺度一样(都为n,128,2)，且为一一对应关系，此时输出的结果则为
    # return M,N
    # return None

    gt_nums, points_num = gt_contours.shape[:2]
    anchor_num = anchor_contours.shape[0]
    
    box_iou_matrix = contour2box_iou(gt_contours, anchor_contours)
    if is_eye:
        eyes_tensor = torch.eye(anchor_num, device = box_iou_matrix.device)
        box_iou_matrix = (box_iou_matrix*eyes_tensor).sum(dim=1)
    
    if gt_locations is None:
        gt_cx = gt_contours[:, :, 0].mean(dim=1).reshape(gt_nums, 1)
        gt_cy = gt_contours[:, :, 1].mean(dim=1).reshape(gt_nums, 1)
    else:
        gt_cx = gt_locations[:,0].reshape(gt_nums, 1)
        gt_cy = gt_locations[:,1].reshape(gt_nums, 1)
    
    L_gt_x = (gt_cx - gt_contours[:, :, 0])**2
    L_gt_y = (gt_cy - gt_contours[:, :, 1])**2
    L_gt = torch.sqrt(L_gt_x + L_gt_y) #gt_nums, points_num

    
    if is_eye:
        L_gt = L_gt.reshape(gt_nums, points_num)
    else:
        L_gt = L_gt.reshape(gt_nums, 1, points_num).repeat(1, anchor_num, 1)

    if is_eye:
        L_anchor_x = (gt_cx - anchor_contours[:, :, 0])**2
        L_anchor_y = (gt_cy - anchor_contours[:, :, 1])**2
    else:
        L_anchor_x = (gt_cx.reshape(gt_nums, 1, 1) - anchor_contours[:, :, 0].unsqueeze(0))**2
        L_anchor_y = (gt_cy.reshape(gt_nums, 1, 1) - anchor_contours[:, :, 1].unsqueeze(0))**2

    
    L_pred = torch.sqrt(L_anchor_x + L_anchor_y) 
    total = torch.stack([L_gt, L_pred], dim=-1)
    l_max = total.max(dim=-1)[0]
    l_min = total.min(dim=-1)[0]
    iou_matrix = (l_max.sum(dim=-1) / l_min.sum(dim=-1)).log()
    if is_trans and not is_eye:
        iou_matrix_T  = iou_matrix.T # (N,M)
        iou_matrix = (iou_matrix + iou_matrix_T)/2 # (M,N)
        
    iou_matrix = iou_matrix.clamp(-1.0, 1.0)
    iou_matrix = 1 - torch.abs(iou_matrix)
    
    return iou_matrix * box_iou_matrix

def scale_mask2feature_shape(targets, ori_shape, feature_size):
    if ori_shape[0] / feature_size[0] != 1.0:
        for target in targets:
            mask = target['masks'].unsqueeze(0).unsqueeze(0)
            target['masks'] = torch.nn.functional.interpolate(mask, size=feature_size, mode='nearest').squeeze()
    return targets

class FFPN(nn.Module):
    def __init__(self, is_refinement = True, classes = 2,
                 order = 7, sample_num = 128, fg_iou_thresh = 0.25, bg_iou_thresh = 0.1,
                 anchor_file_path = '', top_k=20, nms_thresh= 0.1, score_thresh= 0.05):
        super(FFPN, self).__init__()
        self.is_refinement = is_refinement
        self.top_k = top_k
        self.order = order
        self.contour_sample_num = sample_num
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.classes = classes
        self.anchor_num = len(json.load(open(anchor_file_path)))
        print('self.anchor_num', self.anchor_num)
        
        self.core = FFPN_Core(anchor_num = self.anchor_num,
                              order=order,
                                classes = classes,
                                is_refinement = is_refinement)
        self.anchor_generater = FourierAnchorGenerater(order = self.order, sampling_num=self.contour_sample_num, fourier_file = anchor_file_path)

        if self.is_refinement:
            self.csr_sample_num = 16
            self.roi_align_shape = 2
            self.refine_head = ContourSampleRefineHead(64*self.roi_align_shape*self.roi_align_shape*(self.csr_sample_num + 1) , self.order)

        self.fl_loss = FocalLoss(class_num=classes, gamma=2, size_average=False)
        self.proposal_matcher = Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            pos_ratio = 1.0, neg_ratio=20.0
        )

    def get_refine_thres(self):
        if self.training:
            refine_iou_thres = 0.7
        else:
            refine_iou_thres = 0.7
        return refine_iou_thres

    def flilter_proposals(self, pred_fouriers, objectness, pred_locations, num_imgs, original_size, refinement_feature = None):
        # pred_fouriers bs*h*w*a, order*4
        # objectness (fpn, bs, h*w*a, c+1)
        # pred_locations (bs*h*w*a, 2)
        device = pred_fouriers.device
        num_classes = objectness.shape[-1]
        pred_fouriers = pred_fouriers.reshape(num_imgs, -1, self.order*4)# (bs, h*w*a, order*4)
        pred_locations = pred_locations.reshape(num_imgs, -1, 2)

        objectness = torch.softmax(objectness, dim=-1) # (bs, h*w*a, 2)
        objectness = objectness.mean(dim=0) # (bs, h*w*a, c+1)
        num_anchors = pred_fouriers.shape[1]
        top_k = int(min(self.top_k, num_anchors))
        top_n_scores, top_n_idx = objectness.topk(top_k, dim=1)
        # bs, n, c+1
        top_n_scores = top_n_scores[:, :, 1:].permute(0,2,1) # bs, c, 100
        # bs, c, n
        top_n_idx = top_n_idx[:, :, 1:].permute(0,2,1)  # bs, c, 100, remove background index(b,c,n)
        top_n_idx = top_n_idx.reshape(num_imgs, -1)
        batch_idx = torch.arange(num_imgs, device=device)[:, None]
        pred_fouriers = pred_fouriers[batch_idx, top_n_idx]
        pred_fouriers = pred_fouriers.reshape(num_imgs, -1, top_k, self.order*4)
        # bs, c, n, order*4
        pred_locations = pred_locations[batch_idx, top_n_idx]
        pred_locations = pred_locations.reshape(num_imgs, -1, top_k, 2)
        # bs, c, n, 2
        
        all_scores = []
        all_labels = []
        all_fouriers = []
        all_locations = []
        all_contours = []
        all_batch_infor = []
        batch = 0
        tmp_contours = []
        for scores, fouriers, loc in zip(top_n_scores, pred_fouriers, pred_locations):
            # scores (c,n)
            labels = torch.arange(1, num_classes, device=device)
            labels = labels.view(-1, 1).expand_as(scores)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            fouriers = fouriers.reshape(-1, self.order, 4)
            loc = loc.reshape(-1, 2)
            
            keep =  torch.where(scores >= self.score_thresh)[0]
            scores, labels, fouriers, loc = scores[keep], labels[keep], fouriers[keep], loc[keep]
            contours = fouriers2contours_anchor(fouriers, self.contour_sample_num, locations=loc)
            contours[..., 0].clamp_(0, original_size[1] - 1)
            contours[..., 1].clamp_(0, original_size[0] - 1)
            # n,128,2
            if not self.is_refinement:
                if contours.numel() > 0:
                    x1 = contours[:,:,0].min(dim=-1)[0]
                    x2 = contours[:,:,0].max(dim=-1)[0]
                    y1 = contours[:,:,1].min(dim=-1)[0]
                    y2 = contours[:,:,1].max(dim=-1)[0]
                    selected_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                else:
                    selected_boxes = torch.empty((0, 4), device=contours.device)
                
                keep = box_ops.batched_nms(selected_boxes, scores, labels, self.nms_thresh)
                scores, labels, fouriers, loc, contours = scores[keep], labels[keep], fouriers[keep], loc[keep], contours[keep]
            
            batch_num = fouriers.shape[0]
            batch_infor = torch.tensor(batch, device = device).repeat(batch_num)
            all_scores.append(scores)
            all_labels.append(labels)
            all_fouriers.append(fouriers)
            all_locations.append(loc)
            all_contours.append(contours)
            all_batch_infor.append(batch_infor)
            if len(contours) != 0:
                tmp_contours.append(contours[0])

            batch += 1
        
        if self.is_refinement:
            contours = torch.cat(all_contours, dim = 0)
            fouriers = torch.cat(all_fouriers, dim = 0)
            loc = torch.cat(all_locations, dim = 0)
            labels = torch.cat(all_labels, dim = 0).reshape(-1)
            all_scores =  torch.cat(all_scores, dim = 0).reshape(-1)
            bs_infor = torch.cat(all_batch_infor, dim = 0).reshape(-1)

            refine_off, fouriers_mean, locations_mean, bs_infors, merge_labels, scores = \
                self.refinement_method(refinement_feature, contours, fouriers, loc, \
                        bs_infor, labels, merge_iou_thres = self.refine_iou_thres, ori_shape = original_size, pred_scores=all_scores)

            all_scores = []
            all_labels = []
            all_fouriers = []
            all_locations = []
            all_contours = []
            if scores is not None: 
                refine_fouriers, refine_locations, refine_contours = \
                        decode_refine_result(refine_off, fouriers_mean, locations_mean, order = self.order)
                refine_contours[..., 0].clamp_(0, original_size[1] - 1)
                refine_contours[..., 1].clamp_(0, original_size[0] - 1)
                for b in torch.unique(bs_infors):
                    sel = bs_infors == b
                    all_scores.append(scores[sel])
                    all_labels.append(merge_labels[sel])
                    all_fouriers.append(refine_fouriers[sel])
                    all_locations.append(refine_locations[sel])
                    all_contours.append(refine_contours[sel])
        return all_scores, all_labels, all_fouriers, all_locations, all_contours, tmp_contours
    
    
    def assign_targets_to_anchors(self, shapes_anchor, targets):
        labels = []
        matched_gt_fouriers = []
        matched_gt_locations = []
        matched_gt_contours = []
        for shapes_per_image, targets_per_image in zip(shapes_anchor, targets):#看有几个BS
            # shapes_per_image bs,h*w*a, 128, 2
            gt_fouiers = targets_per_image["fouriers"] # n, order,4
            gt_locations = targets_per_image['locations'] # n, 2
            gt_contours = targets_per_image['contours'] # n, 128,2
            gt_labels = targets_per_image['labels'] # n,1
            gt_masks = targets_per_image['masks'] # (ori_h, ori_w)
            match_quality_matrix = contour_ious(gt_contours, shapes_per_image, gt_locations)
            # n, bs*h*w*a
            gt_masks = gt_masks.unsqueeze(-1).repeat(1, 1, self.anchor_num).reshape(-1) 
            
            matched_idxs = self.proposal_matcher(match_quality_matrix) # -1, -2, 0,1,2,3 ...

            
            new_between_thresholds = (matched_idxs >= 0) & (gt_masks == 0)
            matched_idxs[new_between_thresholds] = self.proposal_matcher.BETWEEN_THRESHOLDS

            
            clamped_matched_idxs = matched_idxs.clamp(min=0)
            matched_gt_fouriers_per_image = gt_fouiers[clamped_matched_idxs]#获取index=1时
            matched_gt_loctions_per_image = gt_locations[clamped_matched_idxs]
            matched_gt_contours_per_image = gt_contours[clamped_matched_idxs]
            labels_per_image = gt_labels[clamped_matched_idxs].to(dtype=torch.float32)
            bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD # -1
            labels_per_image[bg_indices] = 0

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS # -2
            labels_per_image[inds_to_discard] = -1
            labels.append(labels_per_image)
            matched_gt_fouriers.append(matched_gt_fouriers_per_image)
            matched_gt_locations.append(matched_gt_loctions_per_image)
            matched_gt_contours.append(matched_gt_contours_per_image)
            
        return labels, matched_gt_fouriers, matched_gt_locations, matched_gt_contours


    def refinement_method(self, refinement_feature, pred_contours, pred_fouiers, pred_locations, bs_infor, labels, merge_iou_thres = 0.7,\
                   gt_contours = None, gt_loactions = None, targets = None, ori_shape = [416,416],\
                   pred_scores = None):
        # refinement_feature, bs, 256, 52,52
        # pred_contours n, 128, 2
        # bs_infor n value(0->bs-1)
        # labels n value(1->class_num)
        # gt_contours n, 128, 2
        bs, _, fh, fw = refinement_feature.shape
        device = refinement_feature.device
        spatial_scale = fh / ori_shape[0]
        contours = []
        fouriers = []
        locations = []
        scores = []

        merge_labels = []
        bs_infors = []
        if self.training:
            iou_matrix = contour_ious(gt_contours, pred_contours, gt_loactions, is_eye=True)
            for b in range(bs):
                sel = bs_infor == b
                sel_labels = labels[sel]
                sel_ious = iou_matrix[sel]
                sel_pred_contours, sel_fouriers, sel_locations = pred_contours[sel], pred_fouiers[sel], pred_locations[sel]
                
                for l in torch.unique(sel_labels):
                    index = sel_labels == l
                    selected_contours = sel_pred_contours[index]
                    selected_ious = sel_ious[index]
                    selected_fouriers = sel_fouriers[index]
                    selected_locations = sel_locations[index]
                    keep =  torch.where(selected_ious >= merge_iou_thres)[0]
                    selected_contours, selected_fouriers, selected_locations  \
                            = selected_contours[keep], selected_fouriers[keep], selected_locations[keep]
                    
                    if len(selected_contours) == 0:

                        gt_label = targets[b]['labels'] #n
                        gt_fouier = targets[b]["fouriers"] # n, order,4
                        gt_location = targets[b]['locations'] # n, 2
                        label_index = gt_label == l
                        selected_fouriers = gt_fouier[label_index].reshape(-1, self.order, 4)
                        selected_locations = gt_location[label_index].reshape(-1, 2)
                        scores.append(1.0)
                    else:

                        selected_fouriers = selected_fouriers.mean(dim=0).reshape(-1, self.order, 4)
                        selected_locations = selected_locations.mean(dim=0).reshape(-1, 2)
                        scores.append(0.0)

                    sample_contour = fouriers2contours_anchor(selected_fouriers,  self.csr_sample_num, locations = selected_locations)
                    sample_contour.clamp_(0, ori_shape[1]-1.)
                    
                    contours.append(sample_contour)
                    fouriers.append(selected_fouriers)
                    locations.append(selected_locations)
                    merge_labels.append(l)    
                    bs_infors.append(b)    
        else:

            for b in range(bs):
                sel = bs_infor == b
                sel_labels = labels[sel]
                sel_pred_contours, sel_fouriers, sel_locations, sel_pred_scores = \
                    pred_contours[sel], pred_fouiers[sel], pred_locations[sel], pred_scores[sel]
                sel_ious = contour_ious(sel_pred_contours, sel_pred_contours, is_trans=True) #  gt_locations = sel_locations,

                # m, m
                for l in torch.unique(sel_labels):
                    index = sel_labels == l
                    in_iou_matrix = sel_ious[index][:,index] # m2,m2
                    selected_fouriers = sel_fouriers[index] # m2, 7x4
                    selected_locations = sel_locations[index] # m2,2
                    selected_scores = sel_pred_scores[index]
                    iou_mask = in_iou_matrix > merge_iou_thres
                    iou_num = iou_mask.sum(dim=1)
                    max_over_num = torch.max(iou_num)

                    if max_over_num > 1:

                        max_iou_index = torch.where(iou_num == max_over_num)[0] 
                        max_iou_index = torch.min(max_iou_index)
                        together_contour_index = torch.where(iou_mask[max_iou_index, :] == True)[0]
                        selected_fouriers= selected_fouriers[together_contour_index, ...].mean(dim=0).reshape(-1, self.order, 4)
                        selected_locations= selected_locations[together_contour_index, ...].mean(dim=0).reshape(-1, 2)
                        selected_scores = selected_scores[together_contour_index].mean()
                    else:
                        selected_fouriers= selected_fouriers[0].reshape(-1, self.order, 4)
                        selected_locations= selected_locations[0].reshape(-1, 2)
                        selected_scores = selected_scores[0]

                    sample_contour = fouriers2contours_anchor(selected_fouriers,  self.csr_sample_num, locations = selected_locations)

                    contours.append(sample_contour)
                    fouriers.append(selected_fouriers)
                    locations.append(selected_locations)
                    merge_labels.append(l)
                    bs_infors.append(b)
                    scores.append(selected_scores)

        if len(contours) != 0:
            contours = torch.cat(contours, dim=0).reshape(-1, self.csr_sample_num, 2).detach()
            fouriers = torch.stack(fouriers, dim=0).detach() # k, order*4
            locations = torch.stack(locations, dim=0).detach().reshape(-1, 2) # k ,2
            merge_labels = torch.as_tensor(merge_labels, device = device).reshape(-1)
            scores = torch.as_tensor(scores, device = device).reshape(-1)
            bs_infors = torch.as_tensor(bs_infors, device = device).reshape(-1)
            dist_x = contours[:,:,0].max(dim=1)[0] - contours[:,:,0].min(dim=1)[0]
            dist_y = contours[:,:,1].max(dim=1)[0] - contours[:,:,1].min(dim=1)[0]
            max_dist = torch.stack([dist_x, dist_y], dim=1)
            merge_points =  torch.cat([locations.reshape(-1,1,2), contours], dim=1) # n, 17, 2
            
            x1 = (merge_points[..., 0] - max_dist[:,0].reshape(-1, 1)/(2*self.csr_sample_num)).clamp(0., ori_shape[1]-1.) # n, self.csr_sample_num
            x2 = (merge_points[..., 0] + max_dist[:,0].reshape(-1, 1)/(2*self.csr_sample_num)).clamp(0., ori_shape[1]-1.)
            y1 = (merge_points[..., 1] - max_dist[:,1].reshape(-1, 1)/(2*self.csr_sample_num)).clamp(0., ori_shape[0]-1.)
            y2 = (merge_points[..., 1] + max_dist[:,1].reshape(-1, 1)/(2*self.csr_sample_num)).clamp(0., ori_shape[0]-1.)
            
            boxes = torch.stack([bs_infors.reshape(-1,1).repeat(1,self.csr_sample_num+1),x1,y1,x2,y2], dim=-1) #n,self.csr_sample_num+1,5
            boxes = boxes.reshape(-1,5)
            boxes[:,1:] *= spatial_scale
            bs_num = bs_infors.shape[0]
            roi_feature = roi_f(refinement_feature, boxes, self.roi_align_shape)
            roi_feature = roi_feature.reshape(bs_num, -1)
            refine_off = self.refine_head(roi_feature)
            
            return refine_off, fouriers, locations, bs_infors, merge_labels, scores
            
        else:
            return None, None, None, None, None, None, None#, None
    

    def computer_loss(self, objectness, pred_four_deltas, pred_loc_deltas,
                      encoder_fouriers, encoder_locations,
                      pred_fouriers, pred_locations, labels,
                      gt_fouriers, gt_locations, gt_contours, refinement_feature,
                      targets = None):
        # objectness (fpn, b, h*w*a, numclass)
        # pred_fouriers (b, h*w*a, order*4)
        # pred_locations (b, h*w*a, 2)
        # class_num = objectness.shape[-1]
        fpn_num, bs, anchor_num, class_num = objectness.shape
        device = objectness.device
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)#
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.reshape(fpn_num, -1, class_num)        
        labels = torch.cat(labels, dim=0)#
        labels = labels.to(torch.int64)
        if sampled_pos_inds.numel() == 0 :
            objectness_loss = 0
            for i in range(fpn_num):
                objectness_loss += F.cross_entropy(objectness[i, sampled_inds,...], labels[sampled_inds])
            objectness_loss /= fpn_num
            return objectness_loss, None, None, None, None
        else:
            gt_fouriers = torch.cat(gt_fouriers, dim=0)[sampled_pos_inds] # n,order, 4
            gt_locations = torch.cat(gt_locations, dim=0)[sampled_pos_inds]
            gt_contours = torch.cat(gt_contours, dim=0)[sampled_pos_inds] # n,128,2
            
            pred_pos_fouriers = pred_fouriers[sampled_pos_inds] # n, order, 4
            pred_pos_locations = pred_locations[sampled_pos_inds] # n, 2
            pred_pos_contours = fouriers2contours_anchor(fourier = pred_pos_fouriers, samples=self.contour_sample_num, locations=pred_pos_locations)
            # pred_pos_contours n, 128, 2
            locations_loss = F.smooth_l1_loss(pred_loc_deltas[sampled_pos_inds], encoder_locations[sampled_pos_inds], reduction='mean')
            fouriers_loss = F.smooth_l1_loss(pred_four_deltas[sampled_pos_inds], encoder_fouriers[sampled_pos_inds], reduction='none')
            fouriers_loss = fouriers_loss.mean()
            contour_iou = contour_ious(gt_contours, pred_pos_contours, gt_locations, is_eye=True)
            contours_loss = (1 - contour_iou).mean()
            

            if self.is_refinement:
                
                bs_infor = torch.arange(0, bs, device = device)
                bs_infor = bs_infor.reshape(-1,1).repeat(1, anchor_num).reshape(-1)
                pos_bs_infor = bs_infor[sampled_pos_inds]
                pos_labels = labels[sampled_pos_inds]

                refine_off, fouriers_mean, locations_mean, bs_infors, merge_labels, scores \
                    = self.refinement_method(refinement_feature, pred_pos_contours, pred_pos_fouriers, pred_pos_locations, \
                        pos_bs_infor, pos_labels, merge_iou_thres = self.refine_iou_thres, gt_contours = gt_contours, \
                            gt_loactions = gt_locations, targets = targets, ori_shape = [416,416])
                if scores is not None:
                    target_contours = []
                    target_fouriers = []
                    target_locations = []
                    for target in targets:
                        target_contours.append(target['contours'])
                        target_fouriers.append(target['fouriers'])
                        target_locations.append(target['locations'])
                    sample_index = bs_infors * (self.classes - 1) + (merge_labels - 1)
                    sample_index = sample_index.to(torch.int64)
                    target_contours = torch.cat(target_contours, dim=0).reshape(-1, self.contour_sample_num, 2)[sample_index]
                    target_fouriers = torch.cat(target_fouriers, dim=0).reshape(-1, self.order, 4)[sample_index]
                    target_locations = torch.cat(target_locations, dim=0).reshape(-1, 2)[sample_index]
                    
                    _, _, refine_contours = \
                        decode_refine_result(refine_off, fouriers_mean, locations_mean, order = self.order, sampling_num=self.contour_sample_num)
                        
                    delta_fouriers, delta_location = \
                        encode_refine_target(fouriers_mean, locations_mean, target_fouriers, target_locations, order = self.order)
                        
                    target_ious = contour_ious(target_contours, refine_contours.reshape(-1, self.contour_sample_num, 2), target_locations, is_eye = True)
                    
                    loc_offset = refine_off[:,:2].reshape(-1)
                    four_offset = refine_off[:, 2:].reshape(-1)
                    refine_fouriers_loss = F.smooth_l1_loss(four_offset, delta_fouriers.reshape(-1), reduction='mean')
                    refine_locations_loss = F.smooth_l1_loss(loc_offset, delta_location.reshape(-1), reduction='mean')
                    refine_contours_loss = (1 - target_ious).mean()
                    refine_loss = refine_fouriers_loss + refine_locations_loss + refine_contours_loss
                else:
                    refine_loss = None
            
            else:
                refine_loss = None
                
            objectness_loss_1 = 0
            objectness_loss_2 = 0
            for i in range(fpn_num):
                objectness_loss_1 += F.cross_entropy(objectness[i, sampled_inds,...], labels[sampled_inds])
                objectness_loss_2 += self.fl_loss(objectness[i, sampled_inds,...], labels[sampled_inds])/sampled_pos_inds.numel()
            objectness_loss_1 /= fpn_num
            objectness_loss_2 /= fpn_num
            objectness_loss = objectness_loss_1*0.25 + objectness_loss_2*0.75
            return objectness_loss, fouriers_loss, locations_loss, contours_loss, refine_loss
    

    def forward(self, inputs, targets = None):
        original_size = inputs.shape[-2:] # 416,416
        self.refine_iou_thres = self.get_refine_thres()
        scores, locations, fourier, refinement_feature = self.core(inputs)
        fourier_size= fourier[0].shape[-2:]
        num_imgs = inputs.shape[0]
        objectness, pred_four_deltas, pred_loc_deltas = \
            concat_four_prediction_layers(scores, fourier, locations)
        pred_loc_deltas = pred_loc_deltas.mean(dim=0)
        shapes_acnhor, fouiers_anchor = self.anchor_generater(inputs, fourier)
        proposals_fours, proposals_loc = decoder(pred_four_deltas, pred_loc_deltas, fouiers_anchor)        
        proposals_fours =  proposals_fours.reshape(-1, self.order, 4)

        losses = {}
        result = {}
        if self.training:
            
            targets = scale_mask2feature_shape(targets, original_size, fourier_size)
            
            labels, matched_gt_fouriers, matched_gt_locations, matched_gt_contours = \
                self.assign_targets_to_anchors(shapes_acnhor, targets)

            encoder_fouriers, encoder_locations = encoder(matched_gt_fouriers, matched_gt_locations, fouiers_anchor)
            objectness_loss, fouriers_loss, locations_loss, contours_loss, refine_contours_loss = \
                self.computer_loss(objectness, pred_four_deltas, pred_loc_deltas, encoder_fouriers, encoder_locations,
                                   proposals_fours, proposals_loc, labels, matched_gt_fouriers, matched_gt_locations,
                               matched_gt_contours, refinement_feature,  targets=targets)
            losses['ob'] = objectness_loss
            losses['four'] = fouriers_loss
            losses['loc'] = locations_loss
            losses['cont'] = contours_loss
            losses['refine'] = refine_contours_loss
        else:
            all_scores, all_labels, all_fouriers, all_locations, all_contours, tmp_contours = \
                self.flilter_proposals(proposals_fours.detach(), objectness.detach(), \
                    proposals_loc.detach(), num_imgs, original_size, refinement_feature)
            
            result['batched_scores'] = all_scores
            result['batched_labels'] = all_labels
            result['batched_fouriers'] = all_fouriers
            result['batched_locations'] = all_locations
            result['batched_contours'] = all_contours
            result['batched_tmp_contours'] = tmp_contours
        
        return losses, result