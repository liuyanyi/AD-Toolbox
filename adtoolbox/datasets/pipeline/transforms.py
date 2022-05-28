import copy

import cv2
import mmcv
import numpy as np
import torch
from mmdet.core import find_inside_bboxes
from mmdet.core.visualization import palette_val
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import Mosaic, RandomAffine, MixUp
from mmdet.utils import log_img_scale
from mmrotate.core.bbox.transforms import get_best_begin_point
from mmrotate.core.visualization.palette import get_palette
from mmrotate.core.bbox import obb2poly_np, poly2obb_np, poly2obb
from mmrotate.core.visualization.image import draw_rbboxes
from numpy import random


@PIPELINES.register_module()
class RMosaic(Mosaic):
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:

         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (height, width).
            Default to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Default to (0.5, 1.5).
        min_bbox_size (int | float): The minimum pixel for filtering
            invalid bboxes after the mosaic pipeline. Default to 0.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` is invalid. Default to True.
        pad_val (int): Pad value. Default to 114.
        prob (float): Probability of applying this transformation.
            Default to 1.0.
    """

    def __init__(self,
                 angle_version,
                 img_scale=(640, 640),
                 center_ratio_range=(0.5, 1.5),
                 min_bbox_size=0,
                 bbox_clip_border=True,
                 skip_filter=True,
                 pad_val=114,
                 prob=1.0):
        self.angle_version = angle_version
        super().__init__(img_scale,
                         center_ratio_range,
                         min_bbox_size,
                         bbox_clip_border,
                         skip_filter,
                         pad_val,
                         prob)

    def _mosaic_transform(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        mosaic_labels = []
        mosaic_bboxes = []
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_polygon_i = results_patch['gt_polygons']
            gt_labels_i = results_patch['gt_labels']

            if gt_polygon_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_polygon_i[:, 0::2] = \
                    scale_ratio_i * gt_polygon_i[:, 0::2] + padw
                gt_polygon_i[:, 1::2] = \
                    scale_ratio_i * gt_polygon_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_polygon_i)
            mosaic_labels.append(gt_labels_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)

            # remove outside polygon
            inside_inds = find_inside_polygons(mosaic_bboxes, 2 * self.img_scale[0],
                                               2 * self.img_scale[1])
            mosaic_bboxes = mosaic_bboxes[inside_inds]
            mosaic_labels = mosaic_labels[inside_inds]

            if self.bbox_clip_border:
                mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0,
                                                 2 * self.img_scale[1])
                mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0,
                                                 2 * self.img_scale[0])

            if not self.skip_filter:
                mosaic_bboxes, mosaic_labels = \
                    self._filter_box_candidates(mosaic_bboxes, mosaic_labels)

        # show_img(mosaic_img, mosaic_bboxes)
        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_polygons'] = mosaic_bboxes
        results['gt_labels'] = mosaic_labels

        return results

    def _mosaic_combine(self, loc, center_position_xy, img_shape_wh):
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                    y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def _filter_box_candidates(self, bboxes, labels):
        """Filter out bboxes too small after Mosaic."""
        valid_ind = []
        for i in range(len(bboxes)):
            bbox = poly2obb_np(bboxes[i], self.angle_version)
            bbox_w = bbox[2]
            bbox_h = bbox[3]
            valid_ind.append((bbox_w > self.min_bbox_size) & \
                             (bbox_h > self.min_bbox_size))

        valid_inds = np.array(valid_ind)
        valid_inds = np.nonzero(valid_inds)[0]
        return bboxes[valid_inds], labels[valid_inds]


@PIPELINES.register_module()
class OBB2Poly:

    def __init__(self, angle_version='le90'):
        self.angle_version = angle_version

    def __call__(self, results):
        """Convert bbox to polygon."""
        results['bbox_fields'].remove('gt_bboxes')
        results['bbox_fields'].append('gt_polygons')
        bboxes = results['gt_bboxes']
        scores = np.zeros((bboxes.shape[0], 1), dtype=np.float32)
        bboxes = np.concatenate([bboxes, scores], axis=1)
        polygons = obb2poly_np(bboxes, version=self.angle_version)[:, :8]
        results['gt_polygons'] = polygons
        return results


@PIPELINES.register_module()
class Poly2OBB:

    def __init__(self, angle_version='le90'):
        self.angle_version = angle_version

    def __call__(self, results):
        """Convert polygon to bbox."""
        polygons = results['gt_polygons']
        bbox = poly2obb(torch.Tensor(polygons), version=self.angle_version).numpy()
        results['gt_bboxes'] = bbox
        results['bbox_fields'].remove('gt_polygons')
        results['bbox_fields'].append('gt_bboxes')
        return results


@PIPELINES.register_module()
class RRandomAffine(RandomAffine):

    def __init__(self, angle_version='le90', **kwargs):
        super(RRandomAffine, self).__init__(**kwargs)
        self.angle_version = angle_version

    def __call__(self, results):
        img = results['img']
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2

        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = random.uniform(-self.max_translate_ratio,
                                 self.max_translate_ratio) * width
        trans_y = random.uniform(-self.max_translate_ratio,
                                 self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = (
                translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)

        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)
        results['img'] = img
        results['img_shape'] = img.shape

        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            num_bboxes = len(bboxes)
            if num_bboxes:
                # homogeneous coordinates
                xs = bboxes[:, ::2].reshape(num_bboxes * 4)
                ys = bboxes[:, 1::2].reshape(num_bboxes * 4)
                ones = np.ones_like(xs)
                points = np.vstack([xs, ys, ones])

                warp_points = warp_matrix @ points
                warp_points = warp_points[:2] / warp_points[2]
                xs = warp_points[0].reshape(num_bboxes, 4)
                ys = warp_points[1].reshape(num_bboxes, 4)

                warp_bboxes = np.zeros_like(bboxes)
                warp_bboxes[:, ::2] = xs
                warp_bboxes[:, 1::2] = ys

                # remove outside bbox
                valid_index = find_inside_polygons(warp_bboxes, height, width)

                if self.bbox_clip_border:
                    warp_bboxes[:, ::2] = \
                        warp_bboxes[:, ::2].clip(0, width)
                    warp_bboxes[:, 1::2] = \
                        warp_bboxes[:, 1::2].clip(0, height)

                if not self.skip_filter:
                    raise NotImplementedError
                    # # filter bboxes
                    # filter_index = self.filter_gt_bboxes(
                    #     bboxes * scaling_ratio, warp_bboxes)
                    # valid_index = valid_index & filter_index

                results[key] = warp_bboxes[valid_index]
                if key in ['gt_polygons']:
                    if 'gt_labels' in results:
                        results['gt_labels'] = results['gt_labels'][
                            valid_index]

                if 'gt_masks' in results:
                    raise NotImplementedError(
                        'RandomAffine only supports bbox.')
        # show_img(img, warp_bboxes[valid_index])
        return results


@PIPELINES.register_module()
class RMixUp(MixUp):

    def _mixup_transform(self, results):
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        assert len(
            results['mix_results']) == 1, 'MixUp only support 2 images now !'

        if results['mix_results'][0]['gt_bboxes'].shape[0] == 0:
            # empty bbox
            return results

        retrieve_results = results['mix_results'][0].copy()
        retrieve_img = retrieve_results['img'].copy()
        # show_img(retrieve_img, retrieve_results['gt_polygons'])

        jit_factor = random.uniform(*self.ratio_range)
        is_filp = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = np.ones(
                (self.dynamic_scale[0], self.dynamic_scale[1], 3),
                dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(
                self.dynamic_scale, dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[0],
                          self.dynamic_scale[1] / retrieve_img.shape[1])
        retrieve_img = mmcv.imresize(
            retrieve_img, (int(retrieve_img.shape[1] * scale_ratio),
                           int(retrieve_img.shape[0] * scale_ratio)))

        # 2. paste
        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor),
                                          int(out_img.shape[0] * jit_factor)))

        # 4. flip
        if is_filp:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results['img']
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w,
                                          target_w), 3)).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                             x_offset:x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results['gt_polygons']
        retrieve_gt_bboxes[:, 0::2] = retrieve_gt_bboxes[:, 0::2] * scale_ratio
        retrieve_gt_bboxes[:, 1::2] = retrieve_gt_bboxes[:, 1::2] * scale_ratio
        if self.bbox_clip_border:
            retrieve_gt_bboxes[:, 0::2] = np.clip(retrieve_gt_bboxes[:, 0::2],
                                                  0, origin_w)
            retrieve_gt_bboxes[:, 1::2] = np.clip(retrieve_gt_bboxes[:, 1::2],
                                                  0, origin_h)

        if is_filp:
            ss = retrieve_gt_bboxes.copy()
            retrieve_gt_bboxes[:, 0::2] = (
                    origin_w - retrieve_gt_bboxes[:, 0::2])
            score = np.zeros((retrieve_gt_bboxes.shape[0], 1))
            retrieve_gt_bboxes = np.concatenate([retrieve_gt_bboxes, score], axis=1)
            retrieve_gt_bboxes = get_best_begin_point(retrieve_gt_bboxes)
            retrieve_gt_bboxes = retrieve_gt_bboxes[:, :8]

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.copy()
        cp_retrieve_gt_bboxes[:, 0::2] = \
            cp_retrieve_gt_bboxes[:, 0::2] - x_offset
        cp_retrieve_gt_bboxes[:, 1::2] = \
            cp_retrieve_gt_bboxes[:, 1::2] - y_offset

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_labels = retrieve_results['gt_labels']
        if not self.skip_filter:
            keep_list = self._filter_box_candidates(retrieve_gt_bboxes.T,
                                                    cp_retrieve_gt_bboxes.T)

            retrieve_gt_labels = retrieve_gt_labels[keep_list]
            cp_retrieve_gt_bboxes = cp_retrieve_gt_bboxes[keep_list]

        mixup_gt_bboxes = np.concatenate(
            (results['gt_polygons'], cp_retrieve_gt_bboxes), axis=0)
        mixup_gt_labels = np.concatenate(
            (results['gt_labels'], retrieve_gt_labels), axis=0)

        # remove outside bbox
        inside_inds = find_inside_polygons(mixup_gt_bboxes, target_h, target_w)
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_labels = mixup_gt_labels[inside_inds]

        if self.bbox_clip_border:
            mixup_gt_bboxes[:, 0::2] = np.clip(
                mixup_gt_bboxes[:, 0::2], 0, target_w)
            mixup_gt_bboxes[:, 1::2] = np.clip(
                mixup_gt_bboxes[:, 1::2], 0, target_h)

        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape
        results['gt_polygons'] = mixup_gt_bboxes
        results['gt_labels'] = mixup_gt_labels
        # show_img(mixup_img.astype(np.uint8), mixup_gt_bboxes)
        return results


def find_inside_polygons(polygons, img_shape_x, img_shape_y):
    """Find inside polygons."""
    polygons_ctr_x = polygons[:, ::2].sum(axis=1) / 4
    polygons_ctr_y = polygons[:, 1::2].sum(axis=1) / 4
    polygons_inside_x = (polygons_ctr_x > 0) & (polygons_ctr_x < img_shape_x)
    polygons_inside_y = (polygons_ctr_y > 0) & (polygons_ctr_y < img_shape_y)
    inside_ind = np.nonzero(polygons_inside_x & polygons_inside_y)[0]
    return inside_ind


def show_img(img, bbox):
    import matplotlib.pyplot as plt
    fig = plt.figure('a', frameon=False)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')
    plt.imshow(img)
    bbox_color = 'green',
    num_bboxes = bbox.shape[0]

    boxes = []
    for i in range(num_bboxes):
        boxes.append(poly2obb_np(bbox[i].astype(np.float32), 'le90'))
    boxes = np.array(boxes)

    bbox_palette = palette_val(get_palette('green', 2))
    colors = [bbox_palette[0] for _ in range(num_bboxes)]
    draw_rbboxes(ax, boxes, colors, alpha=0.8, thickness=2)
    plt.show()
