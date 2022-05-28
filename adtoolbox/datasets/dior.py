import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np
from mmcv import print_log
from mmdet.datasets import XMLDataset
from mmdet.datasets.builder import DATASETS
from mmrotate.core import poly2obb_np

from ..core import ad_eval_map


@DATASETS.register_module()
class DIORDataset(XMLDataset):
    CLASSES = ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
               'chimney', 'expressway-service-area', 'expressway-toll-station',
               'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
               'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
               'windmill')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 img_subdir='JPEGImages',
                 ann_subdir='Annotations',
                 bbox_type='hbb',
                 angle_version='oc',
                 **kwargs):
        self.bbox_type = bbox_type
        self.angle_version = angle_version
        super(DIORDataset, self).__init__(ann_file=ann_file,
                                          pipeline=pipeline,
                                          img_subdir=img_subdir,
                                          ann_subdir=ann_subdir,
                                          **kwargs)

    def get_class_name(self, name):
        return name.lower()

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, self.ann_subdir, f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bbox_len = 5 if self.bbox_type == 'obb' else 4
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            name = self.get_class_name(name)
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            if self.bbox_type == 'hbb':
                bnd_box = obj.find('bndbox')
                # TODO: check whether it is necessary to use int
                # Coordinates may be float type
                if bnd_box is None:
                    # Error in DIOR Annotations
                    bnd_box = obj.find('robndbox')
                bbox = [
                    int(float(bnd_box.find('xmin').text)),
                    int(float(bnd_box.find('ymin').text)),
                    int(float(bnd_box.find('xmax').text)),
                    int(float(bnd_box.find('ymax').text))
                ]
                ignore = False
                if self.min_size:
                    assert not self.test_mode
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w < self.min_size or h < self.min_size:
                        ignore = True
            elif self.bbox_type == 'obb':
                bnd_box = obj.find('robndbox')
                polygon = np.array([
                    float(bnd_box.find('x_left_top').text),
                    float(bnd_box.find('y_left_top').text),
                    float(bnd_box.find('x_right_top').text),
                    float(bnd_box.find('y_right_top').text),
                    float(bnd_box.find('x_right_bottom').text),
                    float(bnd_box.find('y_right_bottom').text),
                    float(bnd_box.find('x_left_bottom').text),
                    float(bnd_box.find('y_left_bottom').text),
                ], dtype=np.float32)

                bbox = poly2obb_np(polygon, self.angle_version)

                if bbox is None:
                    continue

                ignore = False
                if self.min_size:
                    assert not self.test_mode
                    w = bbox[2]
                    h = bbox[3]
                    if w < self.min_size or h < self.min_size:
                        ignore = True
            else:
                raise ValueError(f'Unknown bbox type: {self.bbox_type}')

            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, bbox_len))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, bbox_len))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2, dtype=np.float32)
            labels_ignore = np.array(labels_ignore, dtype=np.int64)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def evaluate(
            self,
            results,
            metric='mAP',
            logger=None,
            proposal_nums=(100, 300, 1000),
            iou_thr=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            scale_ranges=None,
            use_07_metric=True,
            nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            use_07_metric (bool): Whether to use the voc07 metric.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')

                mean_ap, _ = ad_eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=use_07_metric,
                    dataset=self.CLASSES,
                    logger=logger,
                    nproc=nproc)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        elif metric == 'recall':
            raise NotImplementedError

        return eval_results
