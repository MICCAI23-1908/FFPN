import torch
import torch.nn as nn
import numpy as np
import math
from torch import Tensor
from typing import Dict
import math
import json

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, use_alpha=False, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).float()

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average
        print('FL_size_average: ', self.size_average)
        print('self.class_num', self.class_num)
    def forward(self, pred, target):
        prob = self.softmax(pred.reshape(-1, self.class_num))
        prob = prob.clamp(min=1e-7, max=1.0-1e-7)

        target_ = torch.zeros(target.size(0), self.class_num, device=pred.device)
        target_.scatter_(1, target.view(-1, 1).long(), 1.)
        if self.use_alpha:
            self.alpha = self.alpha.to(pred.device)
            self.alpha = self.alpha[target.reshape(-1).long()].reshape(-1, 1)
            batch_loss = - self.alpha.float() * torch.pow(1 - prob.detach(), self.gamma).float() * prob.log().float() * target_.float()
        else:
            batch_loss = - torch.pow(1 - prob.detach(), self.gamma).float() * prob.log().float() * target_.float()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:#判断其中元素的个数，若为0则其中一个维度是0，报错
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is 2 (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)#
        if self.allow_low_quality_matches:
            all_matches = matches.clone()

        # print('matches',torch.max(matches))
        # print(torch.max(matches))
        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        
        matches[between_thresholds] = Matcher.BETWEEN_THRESHOLDS#

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)
        matches[below_low_threshold] = Matcher.BELOW_LOW_THRESHOLD#
        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]

class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, pos_ratio=0.75, neg_ratio = 10):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.pos_ratio = pos_ratio
        self.neg_ratio = neg_ratio

    def __call__(self, matched_idxs):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)


            num_pos = int(positive.numel() * self.pos_ratio)
            if num_pos !=0:
                num_neg = int(min(negative.numel(), num_pos*self.neg_ratio))
            else:
                num_neg = int(negative.numel()/self.neg_ratio)



            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx

def fouriers2contours_anchor(fourier, samples=64, cache: Dict[str, Tensor] = None, cache_size: int = 128,
                      locations=None):
    """
    """
    
    order = fourier.shape[-2]
    d = fourier.device
    sampling = torch.linspace(0, 1.0, samples, device=d)
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
    if locations is not None:
        con = con + locations[..., None, :]
    con += (fourier[..., None, (1, 3)] * c_sin[(...,) + (None,) * 1]).sum(-3)
    con += (fourier[..., None, (0, 2)] * c_cos[(...,) + (None,) * 1]).sum(-3)
    return con

def fourier2ellipes_ab(fourier):
    # fourier :   torch.tensor[order ,4]
    device = fourier.device
    final_result = torch.zeros([fourier.shape[0], 2],device=device)
    for kk_ in range(fourier.shape[0]):
        fourier_in = fourier[kk_]

        A, B, C, D = fourier_in[0], fourier_in[1], fourier_in[2], fourier_in[3]

        a = torch.sqrt((A ** 2 + B ** 2))
        b = torch.sqrt((C ** 2 + D ** 2))
        if a > b:
            theta =  torch.arctan(A/C)
        else:
            theta = torch.arctan(C / A)
            temp = a
            a = b
            b = temp
        theta = 180*theta/math.pi
        if abs(theta) > 45:
            first_out = b
            second_out = a
        else:
            first_out  = a
            second_out = b
        current_result = torch.stack([first_out,second_out]).view(1,2)
        final_result[kk_] = current_result
    return  final_result.to(device)

class FourierAnchorGenerater(nn.Module):
    def __init__(self, order = 7, fourier_file = '', sampling_num = 128):
        super().__init__()
        self.order = order
        base_fourier = json.load(open(fourier_file))
        self.base_fourier = torch.as_tensor(base_fourier).reshape(-1,order,4)

        
        self.sampling_num = sampling_num
        self.shape_cache = {}
        self._cache = {}
        self.cell_cache = {}
        self.cell_fours = None
        self.cell_shapes = None
        self.cell_ellipse_ab = None
        
       
    def generate_fourier_anchors(self, base_fourier, device="cpu"):
        order = base_fourier.shape[1]
        num = base_fourier.shape[0]
        base_shape = fouriers2contours_anchor(base_fourier, self.sampling_num, self.shape_cache)
        base_ellipse_ab = fourier2ellipes_ab(base_fourier.reshape(-1, 4)).reshape(num, order, 2) # x, y (order, 2)
        four_list = []
        shape_list = []
        ellipse_ab_list = []

        ratio = 1.0
        four_list.append(base_fourier * ratio)
        shape_list.append(base_shape * ratio)
        ellipse_ab_list.append(base_ellipse_ab*ratio)
        four_list = torch.cat(four_list, dim=0).reshape(-1, order, 4).to(device)
        ellipse_ab_list = torch.cat(ellipse_ab_list, dim=0).reshape(-1, order, 2).to(device)
        shape_list = torch.cat(shape_list, dim=0).reshape(-1, self.sampling_num, 2).to(device)
        return four_list, shape_list, ellipse_ab_list
 
    def set_cell_fourier_shape(self, feature_num,  device):
        if self.cell_fours is not None:
            return self.cell_fours, self.cell_shapes
        cell_fours = []
        cell_shapes = []
        cell_ellipse_abs = []
        cell_four, cell_shape, cell_ellipse_ab = self.generate_fourier_anchors(base_fourier =self.base_fourier, device=device)
        cell_fours.append(cell_four)
        cell_shapes.append(cell_shape)
        cell_ellipse_abs.append(cell_ellipse_ab)

        self.cell_fours = cell_fours
        self.cell_shapes = cell_shapes
        self.cell_ellipse_ab = cell_ellipse_abs

    def grid_anchors(self, grid_sizes, strides):
        shapes = []
        fouriers = []
        for size, stride, base_fourier, base_shape, base_ellipse in zip(
            grid_sizes, strides, self.cell_fours, self.cell_shapes, self.cell_ellipse_ab
        ):

            grid_height, grid_width = size# size=（50，20）
            stride_height, stride_width = stride# = （20，40）
            f_num, order = base_fourier.shape[:2]
            device = base_fourier.device
            shifts_x = torch.arange(
                1, grid_width+1, dtype=torch.float32, device=device
            ) * stride_width 
            shifts_y = torch.arange(
                1, grid_height+1, dtype=torch.float32, device=device
            ) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)#
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            centers = torch.stack((shift_x, shift_y), dim=1) # (-1, 2)
            # print(centers.reshape(grid_height, grid_width, 2))
            num_center = centers.shape[0]
            shapes.append((centers.view(-1,1,1,2) + base_shape.view(1,-1,self.sampling_num,2)).reshape(-1,self.sampling_num,2))
            base_ellipse = base_ellipse.view(1,-1, order*2).repeat(num_center,1,1)
            cat_four = torch.cat([centers.view(-1,1,2).repeat(1,f_num,1), base_ellipse, base_fourier.view(1,-1, order*4).repeat(num_center,1,1)], dim=-1)
            fouriers.append(cat_four.reshape(-1, order*(4+2) + 2))

        return shapes, fouriers

    def cached_grid_anchors(self, grid_sizes, strides):
        key = tuple(grid_sizes) + tuple(strides)#
        if key in self._cache:
            return self._cache[key]
        shapes, fouriers = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = [shapes, fouriers]#
        return shapes, fouriers #
    
    def forward(self, image, feature_maps):
        feature_num = len(feature_maps)
        grid_sizes = tuple([feature_map.shape[-2:] for feature_map in feature_maps])#
        image_size = image.shape[-2:]#
        img_num = int(image.shape[0])
        strides = tuple((image_size[0] / g[0], image_size[1] / g[1]) for g in grid_sizes)#
        self.set_cell_fourier_shape(feature_num, feature_maps[0].device)#
        shapes_over_all_feature_maps, fouriers_over_all_feature_maps \
            = self.cached_grid_anchors(grid_sizes, strides)#
        shapes = []
        fouiers = []
        for i in range(img_num):
            shapes_in_image = []
            fouriers_in_image = []
            for shapes_per_feature_map, fouriers_per_feature_map in zip(shapes_over_all_feature_maps, fouriers_over_all_feature_maps):
                shapes_in_image.append(shapes_per_feature_map)#
                fouriers_in_image.append(fouriers_per_feature_map)
            
            shapes.append(shapes_in_image)
            fouiers.append(fouriers_in_image)
        shapes = [torch.cat(shapes_per_image) for shapes_per_image in shapes]#
        fouiers = [torch.cat(fouriers_per_image) for fouriers_per_image in fouiers]#
        return shapes, fouiers

