# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict, defaultdict

import json_tricks as json
import numpy as np
from mmcv import Config, deprecated_api_warning
from xtcocotools.cocoeval import COCOeval

from ....core.post_processing import oks_nms, soft_oks_nms
from ...builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class TopDownCocoDataset(Kpt2dSviewRgbImgTopDownDataset):
    """CocoDataset dataset for top-down pose estimation.

    "Microsoft COCO: Common Objects in Context", ECCV'2014.
    More details can be found in the `paper
    <https://arxiv.org/abs/1405.0312>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    COCO keypoint indexes::

        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/coco.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        if (not self.test_mode) or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db = []
        for img_id in self.img_ids:
            gt_db.extend(self._load_coco_keypoint_annotation_kernel(img_id))
        return gt_db

    def _load_coco_keypoint_annotation_kernel(self, img_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]

        Args:
            img_id: coco image id

        Returns:
            dict: db entry
        """
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        num_joints = self.ann_info['num_joints']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w))
            y2 = min(height - 1, y1 + max(0, h))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        bbox_id = 0
        rec = []
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            image_file = osp.join(self.img_prefix, self.id2name[img_id])
            rec.append({
                'image_file': image_file,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1

        return rec

    def _load_coco_person_detection_results(self):
        """Load coco person detection results."""
        num_joints = self.ann_info['num_joints']
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            raise ValueError('=> Load %s fail!' % self.bbox_file)

        print(f'=> Total boxes: {len(all_boxes)}')

        kpt_db = []
        bbox_id = 0
        for det_res in all_boxes:
            if det_res['category_id'] != 1:
                continue

            # print('-' * 30, type(det_res['image_id']))
            # print(self.id2name)
            image_file = osp.join(self.img_prefix,
                                  self.id2name[det_res['image_id']])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.det_bbox_thr:
                continue

            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.ones((num_joints, 3), dtype=np.float32)
            kpt_db.append({
                'image_file': image_file,
                'rotation': 0,
                'bbox': box[:4],
                'bbox_score': score,
                'dataset': self.dataset_name,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1
        print(f'=> Total boxes after filter '
              f'low score@{self.det_bbox_thr}: {bbox_id}')
        return kpt_db

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='mAP', **kwargs):
        """Evaluate coco keypoint results. The pose prediction results will be
        saved in ``${res_folder}/result_keypoints.json``.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            results (list[dict]): Testing results containing the following
                items:

                - preds (np.ndarray[N,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - boxes (np.ndarray[N,6]): [center[0], center[1], scale[0], \
                    scale[1],area, score]
                - image_paths (list[str]): For example, ['data/coco/val2017\
                    /000000393226.jpg']
                - heatmap (np.ndarray[N, K, H, W]): model output heatmap
                - bbox_id (list(int)).
            res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
            metric (str | list[str]): Metric to be performed. Defaults: 'mAP'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['mAP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        kpts = defaultdict(list)

        for result in results:
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]
                kpts[image_id].append({
                    'keypoints': preds[i],
                    'center': boxes[i][0:2],
                    'scale': boxes[i][2:4],
                    'area': boxes[i][4],
                    'score': boxes[i][5],
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)

        # rescoring and oks nms
        num_joints = self.ann_info['num_joints']
        vis_thr = self.vis_thr
        oks_thr = self.oks_thr
        valid_kpts = []
        for image_id in kpts.keys():
            img_kpts = kpts[image_id]
            for n_p in img_kpts:
                box_score = n_p['score']
                if kwargs.get('rle_score', False):
                    pose_score = n_p['keypoints'][:, 2]
                    n_p['score'] = float(box_score + np.mean(pose_score) +
                                         np.max(pose_score))
                else:
                    kpt_score = 0
                    valid_num = 0
                    for n_jt in range(0, num_joints):
                        t_s = n_p['keypoints'][n_jt][2]
                        if t_s > vis_thr:
                            kpt_score = kpt_score + t_s
                            valid_num = valid_num + 1
                    if valid_num != 0:
                        kpt_score = kpt_score / valid_num
                    # rescoring
                    n_p['score'] = kpt_score * box_score

            if self.use_nms:
                nms = soft_oks_nms if self.soft_nms else oks_nms
                keep = nms(img_kpts, oks_thr, sigmas=self.sigmas)
                valid_kpts.append([img_kpts[_keep] for _keep in keep])
            else:
                valid_kpts.append(img_kpts)

        self._write_coco_keypoint_results(valid_kpts, res_file)

        # do evaluation only if the ground truth keypoint annotations exist
        if 'annotations' in self.coco.dataset:
            info_str = self._do_python_keypoint_eval(res_file)
            name_value = OrderedDict(info_str)

            if tmp_folder is not None:
                tmp_folder.cleanup()
        else:
            warnings.warn(f'Due to the absence of ground truth keypoint'
                          f'annotations, the quantitative evaluation can not'
                          f'be conducted. The prediction results have been'
                          f'saved at: {osp.abspath(res_file)}')
            name_value = {}

        return name_value

    def _write_coco_keypoint_results(self, keypoints, res_file):
        """Write results into a json file."""
        data_pack = [{
            'cat_id': self._class_to_coco_ind[cls],
            'cls_ind': cls_ind,
            'cls': cls,
            'ann_type': 'keypoints',
            'keypoints': keypoints
        } for cls_ind, cls in enumerate(self.classes)
                     if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """Get coco keypoint results."""
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1,
                                             self.ann_info['num_joints'] * 3)

            result = [{
                'image_id': img_kpt['image_id'],
                'category_id': cat_id,
                'keypoints': key_point.tolist(),
                'score': float(img_kpt['score']),
                'center': img_kpt['center'].tolist(),
                'scale': img_kpt['scale'].tolist()
            } for img_kpt, key_point in zip(img_kpts, key_points)]

            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_det, 'keypoints', self.sigmas)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts



# =========================================================================================

def pre_torso_judge(kpts, data_type='coco'):
    assert kpts.shape == (4, 3)
    
    vis_kpts_num = np.sum(kpts[:, 2] > 0)
    if vis_kpts_num <= 1:
        return True
    return False

def calculate_body(kpts):
    assert len(kpts.shape) == 2

    all_num = kpts.shape[0]
    unvis_kpts = np.sum(kpts[:, 2] == 0)

    return unvis_kpts / all_num

def calculate_torso(kpts):
    return calculate_body(kpts)

def calculate_head(kpts):
    return calculate_body(kpts)

def calculate_arm(kpts):
    return calculate_body(kpts)

def calculate_leg(kpts):
    return calculate_body(kpts)

def calculate_all(complete_torso, complete_body : list):
    ans = 0.5 * complete_torso
    for item in complete_body:
        ans += 0.1 * item
    
    return ans

def caluculate_crowdpose_complete(kpts, data_type='crowdpose'):
    # if pre_torso_judge(np.vstack((kpts[0], kpts[1], kpts[6], kpts[7]))):
    #     return 1

    complete_torso = calculate_torso(np.vstack((kpts[0], kpts[1], kpts[6], kpts[7])))
    complete_head = calculate_head(np.vstack((kpts[12], kpts[13])))
    complete_left_arm = calculate_arm(np.vstack((kpts[3], kpts[5])))
    complete_right_arm = calculate_arm(np.vstack((kpts[2], kpts[4])))
    complete_left_leg = calculate_leg(np.vstack((kpts[9], kpts[11])))
    complete_right_leg = calculate_leg(np.vstack((kpts[8], kpts[10])))

    complete_index = calculate_all(
        complete_torso,
        [complete_head, complete_left_arm, complete_right_arm, complete_left_leg, complete_right_leg]    
    )

    return complete_index

def calculate_coco_complete(kpts, data_type='coco'):
    # if pre_torso_judge(np.vstack((kpts[5], kpts[6], kpts[11], kpts[12]))):
    #     return 1

    complete_torso = calculate_torso(np.vstack((kpts[5], kpts[6], kpts[11], kpts[12])))
    complete_head = calculate_head(np.vstack((kpts[0], kpts[1], kpts[2], kpts[3], kpts[4])))
    complete_left_arm = calculate_arm(np.vstack((kpts[7], kpts[9])))   
    complete_right_arm = calculate_arm(np.vstack((kpts[8], kpts[10])))
    complete_left_leg = calculate_leg(np.vstack((kpts[13], kpts[15])))
    complete_right_leg = calculate_leg(np.vstack((kpts[14], kpts[16])))

    complete_index = calculate_all(
        complete_torso, 
        [complete_head, complete_left_arm, complete_right_arm, complete_left_leg, complete_right_leg],
    )

    return complete_index


def calculate_crowd(ann_list, data_type='coco'):
    all_crowd = 0
    person_num = 0

    for ann_ori in ann_list:
        if data_type == 'coco':
            per_kpt = np.array(ann_ori['keypoints']).reshape(17, 3) 
        else:
            per_kpt = np.array(ann_ori['keypoints']).reshape(14, 3) 
        ori_all_kpt_num = np.sum(per_kpt[:, 2] > 0)

        if ori_all_kpt_num == 0:
            continue  

        x0, y0 = ann_ori['bbox'][0], ann_ori['bbox'][1]
        x1, y1 = x0 + ann_ori['bbox'][2], y0 + ann_ori['bbox'][3]
        other_kpt_num = 0
        for ann_other in ann_list:
            if ann_ori == ann_other:
                continue

            if data_type == 'coco':
                other_kpt = np.array(ann_other['keypoints']).reshape(17, 3)
            else:
                other_kpt = np.array(ann_other['keypoints']).reshape(14, 3)

            # for k in range(other_length):
            other_kpt_num += np.sum(((x0 <= other_kpt[:, 0]).astype(int) + (other_kpt[:, 0] <= x1).astype(int) \
                + (y0 <= other_kpt[:, 1]).astype(int) + (other_kpt[:, 1] <= y1).astype(int) + (other_kpt[:, 2] > 0).astype(int)) == 5)

        # if other_kpt_num / ori_all_kpt_num > 1:
        #     print(other_kpt_num / ori_all_kpt_num)
        all_crowd += other_kpt_num / ori_all_kpt_num
        person_num += 1

    if person_num == 0:
        return -1

    mean_crowd = all_crowd / person_num
    mean_crowd = min(mean_crowd, 1)
    return mean_crowd


def calculate_mean_complete(ann_list, data_type='coco'):
    person_num = 0
    mean_complete_index = 0
    for ann_ori in ann_list:
        if data_type == 'coco':
            per_kpt = np.array(ann_ori['keypoints']).reshape(17, 3) 
        else:
            per_kpt = np.array(ann_ori['keypoints']).reshape(14, 3) 
        ori_all_kpt_num = np.sum(per_kpt[:, 2] > 0)

        if ori_all_kpt_num == 0:
            continue  

        if data_type == 'coco':
            complete_index = calculate_coco_complete(per_kpt.copy(), data_type = 'coco')
        elif data_type == 'crowdpose':
            complete_index = caluculate_crowdpose_complete(per_kpt.copy(), data_type = 'crowdpose')

        mean_complete_index += complete_index
        person_num += 1

    if person_num == 0:
        return -1

    mean_complete_index /= person_num
    
    return mean_complete_index


def calculate_cover(ann_list, data_type='coco'):
    person_per_img = 0
    occlusion_per_img = 0
    for ann in ann_list:
        if data_type == 'coco':
            per_kpt = np.array(ann['keypoints']).reshape(17, 3) 
        else:
            per_kpt = np.array(ann['keypoints']).reshape(14, 3) 
        all_kpt_num = np.sum(per_kpt[:, 2] > 0)
        if all_kpt_num == 0:
            continue
        person_per_img += 1
        occlusion_num = np.sum(per_kpt[:, 2] == 1)
        occlusion_per_img += occlusion_num / all_kpt_num

    # break      
    if person_per_img == 0:
        return -1
    return occlusion_per_img / person_per_img

def calculate_single_mean_complete(ann_origin, data_type='coco'):
    person_num = 0
    mean_complete_index = 0
    ann_list = [ann_origin]
    for ann_ori in ann_list:
        if data_type == 'coco':
            per_kpt = np.array(ann_ori['keypoints']).reshape(17, 3) 
        else:
            per_kpt = np.array(ann_ori['keypoints']).reshape(14, 3) 
        ori_all_kpt_num = np.sum(per_kpt[:, 2] > 0)

        if ori_all_kpt_num == 0:
            continue  

        if data_type == 'coco':
            complete_index = calculate_coco_complete(per_kpt.copy(), data_type = 'coco')
        elif data_type == 'crowdpose':
            complete_index = caluculate_crowdpose_complete(per_kpt.copy(), data_type = 'crowdpose')

        mean_complete_index += complete_index
        person_num += 1

    if person_num == 0:
        return -1

    mean_complete_index /= person_num
    
    return mean_complete_index


def calculate_total_max_mean(ann_list, data_type='coco'):
    cover_index = calculate_cover(ann_list, data_type)
    crowd_index = calculate_crowd(ann_list, data_type)
    complete_index = calculate_mean_complete(ann_list, data_type)
    # print(cover_index, crowd_index, complete_index)

    if cover_index == -1:
        return -1

    final_index = max(cover_index, crowd_index, complete_index)

    return final_index

def calculate_single_cover(ann, data_type='coco'):
    occlusion_per_img = 0

    if data_type == 'coco':
        per_kpt = np.array(ann['keypoints']).reshape(17, 3) 
    else:
        per_kpt = np.array(ann['keypoints']).reshape(14, 3) 
    all_kpt_num = np.sum(per_kpt[:, 2] > 0)
    if all_kpt_num == 0:
        return -1
    occlusion_num = np.sum(per_kpt[:, 2] == 1)
    occlusion_per_img += occlusion_num / all_kpt_num

    return occlusion_per_img


def calculate_single_crowd(ann_origin, ann_list, data_type='coco'):
    all_crowd = 0
    person_num = 0
    ori_ann_list = [ann_origin]

    for ann_ori in ori_ann_list:
        if data_type == 'coco':
            per_kpt = np.array(ann_ori['keypoints']).reshape(17, 3) 
        else:
            per_kpt = np.array(ann_ori['keypoints']).reshape(14, 3) 
        ori_all_kpt_num = np.sum(per_kpt[:, 2] > 0)

        if ori_all_kpt_num == 0:
            continue  

        x0, y0 = ann_ori['bbox'][0], ann_ori['bbox'][1]
        x1, y1 = x0 + ann_ori['bbox'][2], y0 + ann_ori['bbox'][3]
        other_kpt_num = 0
        for ann_other in ann_list:
            if ann_ori == ann_other:
                continue

            if data_type == 'coco':
                other_kpt = np.array(ann_other['keypoints']).reshape(17, 3)
            else:
                other_kpt = np.array(ann_other['keypoints']).reshape(14, 3)

            # for k in range(other_length):
            other_kpt_num += np.sum(((x0 <= other_kpt[:, 0]).astype(int) + (other_kpt[:, 0] <= x1).astype(int) \
                + (y0 <= other_kpt[:, 1]).astype(int) + (other_kpt[:, 1] <= y1).astype(int) + (other_kpt[:, 2] > 0).astype(int)) == 5)

        # if other_kpt_num / ori_all_kpt_num > 1:
        #     print(other_kpt_num / ori_all_kpt_num)
        all_crowd += other_kpt_num / ori_all_kpt_num
        person_num += 1

    if person_num == 0:
        return -1

    mean_crowd = all_crowd / person_num
    mean_crowd = min(mean_crowd, 1)
    return mean_crowd



def calculate_single_total_max_mean(single_ann, ann_list, data_type='coco'):
    cover_index = calculate_single_cover(single_ann, data_type)
    crowd_index = calculate_single_crowd(single_ann, ann_list, data_type)
    complete_index = calculate_single_mean_complete(single_ann, data_type)
    # print(cover_index, crowd_index, complete_index)

    if cover_index == -1:
        return -1

    final_index = max(cover_index, crowd_index, complete_index)

    return final_index


@DATASETS.register_module()
class CLPickTopDownCocoDataset(Kpt2dSviewRgbImgTopDownDataset):
    """CocoDataset dataset for top-down pose estimation.

    "Microsoft COCO: Common Objects in Context", ECCV'2014.
    More details can be found in the `paper
    <https://arxiv.org/abs/1405.0312>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    COCO keypoint indexes::

        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/coco.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']

        self.all_db = self._get_db()
        self.all_db = sorted(self.all_db, key=lambda temp_var : temp_var['complex_index'])
        # print(self.all_db)
        # for db in self.all_db:
        #     print(db['complex_index'], end=' ')
        # print()
        self.last_thr = 0.4
        self.db = []
        self.modify_db(self.last_thr)


        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    # def modify_db(self, complex_index_thr, choose_num=1000000):
    #     self.db = []
    #     count = 0
    #     for i, db in enumerate(self.all_db):
    #         complex_index = db['complex_index']
    #         if (complex_index <= complex_index_thr and count < choose_num) or self.all_db[i]['choosed_flag'] == True:
    #             if self.all_db[i]['choosed_flag'] == False:
    #                 count += 1
    #             self.all_db[i]['choosed_flag'] = True
    #             self.db.append(self.all_db[i])

    def modify_db(self, complex_index_thr, choose_num=10000000):
        if complex_index_thr > 0.5:
            complex_index_thr = 1
        pre_length = len(self.db)
        # print('pre self.db length : ', len(self.db))
        self.db = []
        count = 0
        for i, db in enumerate(self.all_db):
            complex_index = db['complex_index']
            trained_count = db['trained_count']
            if trained_count > 210:
                continue
            if (complex_index <= complex_index_thr and count < choose_num) or self.all_db[i]['choosed_flag'] == True:
                if self.all_db[i]['choosed_flag'] == False:
                    count += 1
                self.all_db[i]['choosed_flag'] = True
                self.all_db[i]['trained_count'] += 1
                self.db.append(self.all_db[i])
        # print('count : ', count)
        # print('updated self.db length : ', len(self.db), '   all length : ', len(self.all_db))
        # print('changed length : ', len(self.db) - pre_length, '  count : ', count)


    def _get_db(self):
        """Load dataset."""
        if (not self.test_mode) or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db = []
        for i, img_id in enumerate(self.img_ids):
            # print(f"{i} / {len(self.img_ids)}", end='\r')
            gt_db.extend(self._load_coco_keypoint_annotation_kernel(img_id))
        return gt_db

    def _load_coco_keypoint_annotation_kernel(self, img_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]

        Args:
            img_id: coco image id

        Returns:
            dict: db entry
        """
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        num_joints = self.ann_info['num_joints']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)
        # ---------------------------------------------------------------
        complex_index = calculate_total_max_mean(objs, 'coco')
        # ---------------------------------------------------------------

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w))
            y2 = min(height - 1, y1 + max(0, h))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        bbox_id = 0
        rec = []
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            image_file = osp.join(self.img_prefix, self.id2name[img_id])
            rec.append({
                'image_file': image_file,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': bbox_id,
                'complex_index': complex_index,
                'choosed_flag' : False,
                'trained_count' : 0
            })
            bbox_id = bbox_id + 1

        return rec

    def _load_coco_person_detection_results(self):
        """Load coco person detection results."""
        num_joints = self.ann_info['num_joints']
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            raise ValueError('=> Load %s fail!' % self.bbox_file)

        print(f'=> Total boxes: {len(all_boxes)}')

        kpt_db = []
        bbox_id = 0
        for det_res in all_boxes:
            if det_res['category_id'] != 1:
                continue

            image_file = osp.join(self.img_prefix,
                                  self.id2name[det_res['image_id']])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.det_bbox_thr:
                continue

            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.ones((num_joints, 3), dtype=np.float32)
            kpt_db.append({
                'image_file': image_file,
                'rotation': 0,
                'bbox': box[:4],
                'bbox_score': score,
                'dataset': self.dataset_name,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1
        print(f'=> Total boxes after filter '
              f'low score@{self.det_bbox_thr}: {bbox_id}')
        return kpt_db

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='mAP', **kwargs):
        """Evaluate coco keypoint results. The pose prediction results will be
        saved in ``${res_folder}/result_keypoints.json``.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            results (list[dict]): Testing results containing the following
                items:

                - preds (np.ndarray[N,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - boxes (np.ndarray[N,6]): [center[0], center[1], scale[0], \
                    scale[1],area, score]
                - image_paths (list[str]): For example, ['data/coco/val2017\
                    /000000393226.jpg']
                - heatmap (np.ndarray[N, K, H, W]): model output heatmap
                - bbox_id (list(int)).
            res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
            metric (str | list[str]): Metric to be performed. Defaults: 'mAP'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['mAP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        kpts = defaultdict(list)

        for result in results:
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]
                kpts[image_id].append({
                    'keypoints': preds[i],
                    'center': boxes[i][0:2],
                    'scale': boxes[i][2:4],
                    'area': boxes[i][4],
                    'score': boxes[i][5],
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)

        # rescoring and oks nms
        num_joints = self.ann_info['num_joints']
        vis_thr = self.vis_thr
        oks_thr = self.oks_thr
        valid_kpts = []
        for image_id in kpts.keys():
            img_kpts = kpts[image_id]
            for n_p in img_kpts:
                box_score = n_p['score']
                if kwargs.get('rle_score', False):
                    pose_score = n_p['keypoints'][:, 2]
                    n_p['score'] = float(box_score + np.mean(pose_score) +
                                         np.max(pose_score))
                else:
                    kpt_score = 0
                    valid_num = 0
                    for n_jt in range(0, num_joints):
                        t_s = n_p['keypoints'][n_jt][2]
                        if t_s > vis_thr:
                            kpt_score = kpt_score + t_s
                            valid_num = valid_num + 1
                    if valid_num != 0:
                        kpt_score = kpt_score / valid_num
                    # rescoring
                    n_p['score'] = kpt_score * box_score

            if self.use_nms:
                nms = soft_oks_nms if self.soft_nms else oks_nms
                keep = nms(img_kpts, oks_thr, sigmas=self.sigmas)
                valid_kpts.append([img_kpts[_keep] for _keep in keep])
            else:
                valid_kpts.append(img_kpts)

        self._write_coco_keypoint_results(valid_kpts, res_file)

        # do evaluation only if the ground truth keypoint annotations exist
        if 'annotations' in self.coco.dataset:
            info_str = self._do_python_keypoint_eval(res_file)
            name_value = OrderedDict(info_str)

            if tmp_folder is not None:
                tmp_folder.cleanup()
        else:
            warnings.warn(f'Due to the absence of ground truth keypoint'
                          f'annotations, the quantitative evaluation can not'
                          f'be conducted. The prediction results have been'
                          f'saved at: {osp.abspath(res_file)}')
            name_value = {}

        return name_value

    def _write_coco_keypoint_results(self, keypoints, res_file):
        """Write results into a json file."""
        data_pack = [{
            'cat_id': self._class_to_coco_ind[cls],
            'cls_ind': cls_ind,
            'cls': cls,
            'ann_type': 'keypoints',
            'keypoints': keypoints
        } for cls_ind, cls in enumerate(self.classes)
                     if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """Get coco keypoint results."""
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1,
                                             self.ann_info['num_joints'] * 3)

            result = [{
                'image_id': img_kpt['image_id'],
                'category_id': cat_id,
                'keypoints': key_point.tolist(),
                'score': float(img_kpt['score']),
                'center': img_kpt['center'].tolist(),
                'scale': img_kpt['scale'].tolist()
            } for img_kpt, key_point in zip(img_kpts, key_points)]

            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_det, 'keypoints', self.sigmas)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts



@DATASETS.register_module()
class MultiCLPickTopDownCocoDataset(Kpt2dSviewRgbImgTopDownDataset):
    """CocoDataset dataset for top-down pose estimation.

    "Microsoft COCO: Common Objects in Context", ECCV'2014.
    More details can be found in the `paper
    <https://arxiv.org/abs/1405.0312>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    COCO keypoint indexes::

        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 start_index=0,
                 end_index=0,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/coco.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']

        self.all_db = self._get_db()
        self.all_db = sorted(self.all_db, key=lambda temp_var : temp_var['complex_index'])[::-1]
        self.db = self.all_db[start_index : end_index]
        # print(self.all_db)
        # for db in self.all_db:
        #     print(db['complex_index'], end=' ')
        # print()
        # self.last_thr = 0.4
        # self.db = []
        # self.modify_db(self.last_thr)


        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    # def modify_db(self, complex_index_thr, choose_num=1000000):
    #     self.db = []
    #     count = 0
    #     for i, db in enumerate(self.all_db):
    #         complex_index = db['complex_index']
    #         if (complex_index <= complex_index_thr and count < choose_num) or self.all_db[i]['choosed_flag'] == True:
    #             if self.all_db[i]['choosed_flag'] == False:
    #                 count += 1
    #             self.all_db[i]['choosed_flag'] = True
    #             self.db.append(self.all_db[i])

    def modify_db(self, complex_index_thr, choose_num=10000000):
        if complex_index_thr > 0.5:
            complex_index_thr = 1
        pre_length = len(self.db)
        # print('pre self.db length : ', len(self.db))
        self.db = []
        count = 0
        for i, db in enumerate(self.all_db):
            complex_index = db['complex_index']
            trained_count = db['trained_count']
            if trained_count > 210:
                continue
            if (complex_index <= complex_index_thr and count < choose_num) or self.all_db[i]['choosed_flag'] == True:
                if self.all_db[i]['choosed_flag'] == False:
                    count += 1
                self.all_db[i]['choosed_flag'] = True
                self.all_db[i]['trained_count'] += 1
                self.db.append(self.all_db[i])
        # print('count : ', count)
        # print('updated self.db length : ', len(self.db), '   all length : ', len(self.all_db))
        # print('changed length : ', len(self.db) - pre_length, '  count : ', count)


    def _get_db(self):
        """Load dataset."""
        if (not self.test_mode) or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db = []
        for i, img_id in enumerate(self.img_ids):
            # print(f"{i} / {len(self.img_ids)}", end='\r')
            gt_db.extend(self._load_coco_keypoint_annotation_kernel(img_id))
        return gt_db

    def _load_coco_keypoint_annotation_kernel(self, img_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]

        Args:
            img_id: coco image id

        Returns:
            dict: db entry
        """
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        num_joints = self.ann_info['num_joints']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)
        # ---------------------------------------------------------------
        # complex_index = calculate_total_max_mean(objs, 'coco')
        # complex_index = calculate_single_total_max_mean(objs, 'coco')
        # ---------------------------------------------------------------

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w))
            y2 = min(height - 1, y1 + max(0, h))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        bbox_id = 0
        rec = []
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            # ---------------------------------------------------------------
            complex_index = calculate_single_total_max_mean(obj, objs, 'coco')
            # ---------------------------------------------------------------

            image_file = osp.join(self.img_prefix, self.id2name[img_id])
            rec.append({
                'image_file': image_file,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': bbox_id,
                'complex_index': complex_index,
                'choosed_flag' : False,
                'trained_count' : 0
            })
            bbox_id = bbox_id + 1

        return rec

    def _load_coco_person_detection_results(self):
        """Load coco person detection results."""
        num_joints = self.ann_info['num_joints']
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            raise ValueError('=> Load %s fail!' % self.bbox_file)

        print(f'=> Total boxes: {len(all_boxes)}')

        kpt_db = []
        bbox_id = 0
        for det_res in all_boxes:
            if det_res['category_id'] != 1:
                continue

            image_file = osp.join(self.img_prefix,
                                  self.id2name[det_res['image_id']])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.det_bbox_thr:
                continue

            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.ones((num_joints, 3), dtype=np.float32)
            kpt_db.append({
                'image_file': image_file,
                'rotation': 0,
                'bbox': box[:4],
                'bbox_score': score,
                'dataset': self.dataset_name,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1
        print(f'=> Total boxes after filter '
              f'low score@{self.det_bbox_thr}: {bbox_id}')
        return kpt_db

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='mAP', **kwargs):
        """Evaluate coco keypoint results. The pose prediction results will be
        saved in ``${res_folder}/result_keypoints.json``.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            results (list[dict]): Testing results containing the following
                items:

                - preds (np.ndarray[N,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - boxes (np.ndarray[N,6]): [center[0], center[1], scale[0], \
                    scale[1],area, score]
                - image_paths (list[str]): For example, ['data/coco/val2017\
                    /000000393226.jpg']
                - heatmap (np.ndarray[N, K, H, W]): model output heatmap
                - bbox_id (list(int)).
            res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
            metric (str | list[str]): Metric to be performed. Defaults: 'mAP'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['mAP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        kpts = defaultdict(list)

        for result in results:
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]
                kpts[image_id].append({
                    'keypoints': preds[i],
                    'center': boxes[i][0:2],
                    'scale': boxes[i][2:4],
                    'area': boxes[i][4],
                    'score': boxes[i][5],
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)

        # rescoring and oks nms
        num_joints = self.ann_info['num_joints']
        vis_thr = self.vis_thr
        oks_thr = self.oks_thr
        valid_kpts = []
        for image_id in kpts.keys():
            img_kpts = kpts[image_id]
            for n_p in img_kpts:
                box_score = n_p['score']
                if kwargs.get('rle_score', False):
                    pose_score = n_p['keypoints'][:, 2]
                    n_p['score'] = float(box_score + np.mean(pose_score) +
                                         np.max(pose_score))
                else:
                    kpt_score = 0
                    valid_num = 0
                    for n_jt in range(0, num_joints):
                        t_s = n_p['keypoints'][n_jt][2]
                        if t_s > vis_thr:
                            kpt_score = kpt_score + t_s
                            valid_num = valid_num + 1
                    if valid_num != 0:
                        kpt_score = kpt_score / valid_num
                    # rescoring
                    n_p['score'] = kpt_score * box_score

            if self.use_nms:
                nms = soft_oks_nms if self.soft_nms else oks_nms
                keep = nms(img_kpts, oks_thr, sigmas=self.sigmas)
                valid_kpts.append([img_kpts[_keep] for _keep in keep])
            else:
                valid_kpts.append(img_kpts)

        self._write_coco_keypoint_results(valid_kpts, res_file)

        # do evaluation only if the ground truth keypoint annotations exist
        if 'annotations' in self.coco.dataset:
            info_str = self._do_python_keypoint_eval(res_file)
            name_value = OrderedDict(info_str)

            if tmp_folder is not None:
                tmp_folder.cleanup()
        else:
            warnings.warn(f'Due to the absence of ground truth keypoint'
                          f'annotations, the quantitative evaluation can not'
                          f'be conducted. The prediction results have been'
                          f'saved at: {osp.abspath(res_file)}')
            name_value = {}

        return name_value

    def _write_coco_keypoint_results(self, keypoints, res_file):
        """Write results into a json file."""
        data_pack = [{
            'cat_id': self._class_to_coco_ind[cls],
            'cls_ind': cls_ind,
            'cls': cls,
            'ann_type': 'keypoints',
            'keypoints': keypoints
        } for cls_ind, cls in enumerate(self.classes)
                     if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """Get coco keypoint results."""
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1,
                                             self.ann_info['num_joints'] * 3)

            result = [{
                'image_id': img_kpt['image_id'],
                'category_id': cat_id,
                'keypoints': key_point.tolist(),
                'score': float(img_kpt['score']),
                'center': img_kpt['center'].tolist(),
                'scale': img_kpt['scale'].tolist()
            } for img_kpt, key_point in zip(img_kpts, key_points)]

            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_det, 'keypoints', self.sigmas)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts

# =========================================================================================