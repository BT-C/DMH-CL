from concurrent.futures import process
from pyexpat import model
from pycocotools.coco import COCO
import numpy as np
import json
import skimage.io as io
import cv2
from skimage.transform import resize
import os
from copy import deepcopy
from multiprocessing import Process, Queue, set_start_method
from torch.multiprocessing import Process, Queue
import warnings

import pycocotools
from pycocotools import mask as mask_util
import torch.multiprocessing as mp
import torch

import sys
import numpy as np 
from PIL import ImageEnhance
from PIL import Image

import copy
import random
import time

import datetime

from syn_utils import get_oks_mean_list, accumulation_pick_data, equal_split_data, get_oks_mean_dict, get_no_peron
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

mp = mp.get_context('spawn')
# try:
#     set_start_method('forkserver')
# except RuntimeError:
#     pass

transform_type_dict = dict(
    brightness=ImageEnhance.Brightness, contrast=ImageEnhance.Contrast,
    sharpness=ImageEnhance.Sharpness,   color=ImageEnhance.Color
)

import pycocotools.mask as mask_utils

'''
coco.dataset: dict
    images: list[dict]
    annotations: list[dict]
'''


class ColorJitter(object):
    def __init__(self, transform_dict):
        self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]
    
    def __call__(self, img):
        out = Image.fromarray(img)
        rand_num = np.random.uniform(0, 1, len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (rand_num[i]*2.0 - 1.0) + 1   # r in [1-alpha, 1+alpha)
            out = transformer(out).enhance(r)
        
        return np.array(out)


max_orig_annid = 900100581904
transform_dict = {'brightness':0.1026, 'contrast':0.0935, 'sharpness':0.8386, 'color':0.1592}
colorJitter = ColorJitter(transform_dict)


def deepcopy_dict(x):
    y = {}
    for key, value in x.items():
        y[deepcopy(key)] = deepcopy(value)
    return y

def deepcopy_list(x):
    y = []
    for a in x:
        y.append(deepcopy_dict(a))
    return y

def computeIoU(moddified_bbox, orig_bbox):
    cx1 = moddified_bbox[0]
    cy1 = moddified_bbox[1]
    cx2 = cx1 + moddified_bbox[2]
    cy2 = cy1 + moddified_bbox[3]

    gx1 = orig_bbox[0]
    gy1 = orig_bbox[1]
    gx2 = gx1 + orig_bbox[2]
    gy2 = gy1 + orig_bbox[3]

    carea = (cx2 - cx1) * (cy2 - cy1)
    garea = (gx2 - gx1) * (gy2 - gy1)

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h

    if carea * garea == 0:
        return 0

    return max(area / carea, area / garea)

def computeMaskIoU(mask_add, mask_orig):
    return np.sum((mask_add+mask_orig)>1.5) / np.sum((mask_add+mask_orig)>=0.9)

def IoUsLowerThanThres(moddified_bbox, orig_bboxes, up_thr=0.6, down_thr=0.1):
    for orig_bbox in orig_bboxes:
        if computeIoU(moddified_bbox, orig_bbox) > up_thr :
            return False
    return True

def IoUsLowerThanThresBackUp(moddified_bbox, orig_bboxes, up_thr=0.6, down_thr=0.1):
    ious_list = []
    for orig_bbox in orig_bboxes:
        ious_list.append(computeIoU(moddified_bbox, orig_bbox))
    if len(ious_list) == 0:
        return True
    max_iou = max(ious_list)
    if down_thr <= max_iou <= up_thr:
        return True
    return False

def modify_ann(ann, x_shift, y_shift, scale_w, scale_h):
    x_orig, y_orig, w_orig, h_orig = ann['bbox']
    ann['bbox'] = [x_shift, y_shift, int(w_orig*scale_w), int(h_orig*scale_h)]
    for i in range(len(ann['segmentation'])):
        segm = np.array(ann['segmentation'][i])
        segm[::2] = (segm[::2] - x_orig) * scale_w + x_shift
        segm[1::2] = (segm[1::2] - y_orig) * scale_h + y_shift
        ann['segmentation'][i] = segm.tolist()
    kpts = np.array(ann['keypoints'])
    kpts[::3] = (kpts[::3] - x_orig) * scale_w + x_shift
    kpts[1::3] = (kpts[1::3] - y_orig) * scale_h + y_shift
    ann['keypoints'] = kpts.tolist()
    
def image_occlusion(occlusion_image, ann_list, random_occlusion_thr=0.5):
    occlusion_image_backup = copy.deepcopy(occlusion_image)
    result_ann_list = []
    for ann in ann_list:
        occlusion_ann = copy.deepcopy(ann)
        area = occlusion_ann['area'] / 2e3
        kpts = np.array(occlusion_ann['keypoints']).reshape(17, 3)
        for kpt_id, kpt in enumerate(kpts):
            if kpt[-1] == 2 and random.random() < random_occlusion_thr:
            # if True:
                kpts[kpt_id][-1] = 1
                h = random.random() * 3 + area
                w = random.random() * 3 + area
                temp_img = occlusion_image[int(kpt[1]-h):int(kpt[1]+h), int(kpt[0]-w):int(kpt[0]+w), :]
                if temp_img.shape[0] * temp_img.shape[1] == 0:
                    continue 
                corner_id = np.random.randint(0, 4, (1)).item()
                if corner_id == 0:
                    bg_img = occlusion_image_backup[:temp_img.shape[0], :temp_img.shape[1], :]
                elif corner_id == 1:
                    bg_img = occlusion_image_backup[:temp_img.shape[0], -temp_img.shape[1]:, :]
                elif corner_id == 2:
                    bg_img = occlusion_image_backup[-temp_img.shape[0]:, :temp_img.shape[1], :]
                elif corner_id == 3:
                    bg_img = occlusion_image_backup[-temp_img.shape[0]:, -temp_img.shape[1]:, :]
                occlusion_image[int(kpt[1]-h):int(kpt[1]+h), int(kpt[0]-w):int(kpt[0]+w), :] = bg_img

        occlusion_ann['keypoints'] = kpts.reshape(-1).tolist()
        result_ann_list.append(occlusion_ann)
        
    return occlusion_image, result_ann_list


def evaluateImg(ious, gt):
    '''
    perform evaluation for single category and image
    :return: dict (single image results)
    '''
    
    if len(ious) == 0 or len(gt) == 0:
        return -1

    for g in gt:
        if g['iscrowd']:
            g['_ignore'] = 1
        else:
            g['_ignore'] = 0

    # sort dt highest score first, sort gt ignore last
    gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
    gt = [gt[i] for i in gtind]
    iscrowd = [int(o['iscrowd']) for o in gt]
    # load computed ious
    ious = np.array(ious)
    ious = ious[:, gtind] if len(ious) > 0 else ious
    
    G = len(gt)
    gtm  = np.zeros((G))
    gtm[:] = -1
    gtIg = np.array([g['_ignore'] for g in gt])

    iou_sum = -1
    if not len(ious) == 0:
        iou_sum = 0
        match_count = 0
        for dind, d in enumerate(ious):
            # information about best match so far (m=-1 -> unmatched)
            iou = 0
            m   = -1
            for gind, g in enumerate(gt):
                # if this gt already matched, and not a crowd, continue
                if gtm[gind]>-1 and not iscrowd[gind]:
                    continue
                # if dt matched to reg gt, and on ignore gt, stop
                if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                    break
                # continue to next gt unless better match made
                if ious[dind,gind] < iou:
                    continue
                # if match successful and best so far, store appropriately
                iou=ious[dind,gind]
                m=gind
            # if match made store id of match for both dt and gt
            if m ==-1:
                continue
            iou_sum += iou
            match_count += 1
            # print('iou_sum :', iou_sum)
            gtm[m]     = dind
        iou_sum /= match_count
            # dtIg[tind,dind] = gtIg[m]
            # dtm[tind,dind]  = gt[m]['id']
            
    return iou_sum


def evaluateImg_every_pose(ious_item, coco, img_id):
    '''
    perform evaluation for single category and image
    :return: dict (single image results)
    '''
    # coco = COCO(ann_file)
    ann_ids = coco.getAnnIds(imgIds=img_id)
    gt_list = coco.loadAnns(ids=ann_ids)
    
    ious = ious_item[0]
    gt_ann_id = ious_item[1]
    gt = [0 for _ in range(len(gt_list))]
    for order_id, temp_ann_id in enumerate(gt_ann_id):
        for temp_gt in gt_list:
            if temp_gt['id'] == temp_ann_id:
                gt[order_id] = temp_gt
                break
    
        
    if len(ious) == 0 or len(gt) == 0:
        return -1

    for g in gt:
        if g['iscrowd']:
            g['_ignore'] = 1
        else:
            g['_ignore'] = 0

    # sort dt highest score first, sort gt ignore last
    gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
    gt = [gt[i] for i in gtind]
    iscrowd = [int(o['iscrowd']) for o in gt]
    # load computed ious
    
    # print(gtind)
    ious = np.array(ious)
    ious = ious[:, gtind] if len(ious) > 0 else ious
    
    
    G = len(gt)
    gtm  = np.zeros((G))
    gtm[:] = -1
    gtIg = np.array([g['_ignore'] for g in gt])

    result_ann_oks = {g['id'] : 0 for g in gt}
    if not len(ious) == 0:
        for dind, d in enumerate(ious):
            # information about best match so far (m=-1 -> unmatched)
            iou = 0
            m   = -1
            for gind, g in enumerate(gt):
                # if this gt already matched, and not a crowd, continue
                if gtm[gind]>-1 and not iscrowd[gind]:
                    continue
                # if dt matched to reg gt, and on ignore gt, stop
                if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                    break
                # continue to next gt unless better match made
                if ious[dind,gind] < iou:
                    continue
                # if match successful and best so far, store appropriately
                iou=ious[dind,gind]
                m=gind
            # if match made store id of match for both dt and gt
            if m ==-1:
                continue
            if iou > result_ann_oks[gt[m]['id']]:
                result_ann_oks[gt[m]['id']] = iou
            gtm[m]     = dind
    
    # result_ann_list = [
    #     {
    #         'ann_id' : temp_id,
    #         'oks'    : result_ann_oks[temp_id],
    #     } for temp_id in result_ann_oks   
    # ]
    result_ann_list = []
    for temp_gt in gt:
        if temp_gt['id'] not in result_ann_oks:
            continue
        if temp_gt['num_keypoints'] == 0:
            continue
        
        result_ann_list.append({
            'ann_id'  : temp_gt['id'],
            'oks'     : result_ann_oks[temp_gt['id']]
        })
    return result_ann_oks, result_ann_list



def computeOks(dts, gts):
    # dimention here should be Nxm
    if len(dts) > 20:
        dts = dts[0:20]
    # if len(gts) == 0 and len(dts) == 0:
    if len(gts) == 0 or len(dts) == 0:
        return []
    ious = np.zeros((len(dts), len(gts)))
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    vars = (sigmas * 2)**2
    k = len(sigmas)
    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):
        # create bounds for ignore regions(double the gt bbox)
        g = np.array(gt['keypoints'])

        xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
        k1 = np.count_nonzero(vg > 0)
        bb = gt['bbox']
        x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
        y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
        for i, dt in enumerate(dts):
            d = np.array(dt['keypoints'])
            # xd = d[0::3]; yd = d[1::3]
            xd = d[:, 0]; yd = d[:, 1]
            if k1>0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                z = np.zeros((k))
                dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
            e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
            if k1 > 0:
                e=e[vg > 0]
            ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
    return ious


def computeOks_backup(dts, gts):
    # dimention here should be Nxm
    # inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
    # dts = [dts[i] for i in inds]
    if len(dts) > 20:
        dts = dts[0:20]
    # if len(gts) == 0 and len(dts) == 0:
    if len(gts) == 0 or len(dts) == 0:
        return []
    ious = np.zeros((len(dts), len(gts)))
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    vars = (sigmas * 2)**2
    k = len(sigmas)
    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):
        # create bounds for ignore regions(double the gt bbox)
        g = np.array(gt['keypoints'])
        xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
        k1 = np.count_nonzero(vg > 0)
        bb = gt['bbox']
        x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
        y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
        for i, dt in enumerate(dts):
            d = np.array(dt)
            xd = d[0::3]; yd = d[1::3]
            if k1>0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                z = np.zeros((k))
                dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
            e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
            if k1 > 0:
                e=e[vg > 0]
            ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
    return ious

def model_eval_oks(pose_model, target_img_name, anns):
    
    # result = inference_detector(model, target_img_name)
    
    
    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    gts = []
    person_results = []
    for ann in anns:
        person = {}
        # bbox format is 'xywh'
        # person['bbox'] = ann['bbox']
        person['bbox'] = [*ann['bbox'], 1]
        kpts = ann['keypoints']
        gts.append(ann)
        person_results.append(person)

    # test a single image, with a list of bboxes
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        target_img_name,
        person_results,
        bbox_thr=None,
        format='xywh',
        dataset='TopDownCocoDataset',
        dataset_info=dataset_info,
        return_heatmap=False,
        outputs=None)
    
    
    dts = [t.reshape(-1) for t in result[1][0]]
    ious = computeOks(dts, gts) # (len(gts), len(dts))
    if type(ious) == list:
        return -1
    # ious = ious.T
    
    avg_iou = evaluateImg(ious, anns)
    
    return avg_iou
        
def final_syn_function(
        process_id,
        start_i, end_i,
        up_condition_thr, down_condition_thr,
        syn_imgDir, temp_buffer_img_dir, temp_buffer_ann_dir,
        coco_synthesize,
        coco, bg_imgs, buffers,
        id_used, 
        pose_num, up_iou_thr, down_iou_thr, random_occlusion_thr,
        # model,
        oks_mean_dict,
        syn_interval_id,
        queue
    ):
    np.random.seed(process_id * (int(time.time()) % 24817))
    model = init_pose_model(
        ".py", #args.config, 
        ".pth", #args.checkpoint, 
        device=f"cuda:{process_id % 2}"
    )
    
    
    # mid_buffer_ann_dir = os.path.join(temp_buffer_ann_dir, f"ann_{process_id}")
    mid_buffer_ann_dir = temp_buffer_ann_dir + f"_{process_id}"
    if not os.path.exists(mid_buffer_ann_dir):
        os.mkdir(mid_buffer_ann_dir)
            
    print('process_i : ', process_id)
    # max_img_id = 1000000 * (int(time.time()) + process_id)
    max_img_id = 1000000 * (int(time.time()) + process_id)
    i = start_i
    coco_synthesize_list = [{
            'images'         : [],
            'annotations'    : [],
            'categories'     : [coco.dataset['categories'][0]]
        } for _ in range(5)
    ]
    # for i in range(start_i, end_i):
    while i < end_i:
        bg_img_index = i % len(bg_imgs)
        bg_img = copy.deepcopy(bg_imgs[bg_img_index])
        # if i >= len(bg_imgs):
        bg_img['file_name'] = str(max_img_id).zfill(12) + '.jpg'
        bg_img['id'] = max_img_id
        max_img_id += 1
        
        # print(f"{i} / {len(bg_imgs)}  time: {time.time()}", end='\r')
        if len(buffers) < 10:
            break
        # if i > 1500:
        #     break
        # test
        # if i == 64115 - len(non_occlusion_img_ids):
        #     break
               

        I_bg = io.imread("%s" %bg_imgs[bg_img_index]['file_name'])
        if len(I_bg.shape) == 0:
            continue
        if len(I_bg.shape) == 2:
            I_bg = np.expand_dims(I_bg, -1).repeat(3, -1)
   

        num_add = np.random.randint(pose_num, pose_num + 4)
        bboxes_keypoints = []
        added_ann_list = []
        temp_ann_oks_list = []
        real_num_add = 0
        for j in range(num_add):
            index_buf = np.random.randint(len(buffers))
            img_add, ann_add, img_info = deepcopy(buffers[index_buf])
            # print(type(img_add))
            # print("{} image is used".format(img_info['file_name']))
            h_img, w_img, _ = I_bg.shape
            x0, y0, w_bbox_add, h_bbox_add = [int(xywh) for xywh in ann_add['bbox']]
            x1 = x0 + w_bbox_add
            y1 = y0 + h_bbox_add

            # For w,h of bbox rescale
            if max(w_bbox_add, h_bbox_add) < 100:  # small
                scale_w = np.random.uniform(1.5, min(2, (w_img - 100) / w_bbox_add - 0.1))
                scale_h = np.random.uniform(0.98 * scale_w, 1.02 * scale_w)
            elif max(w_bbox_add, h_bbox_add) < 200:  # medium
                scale_w = np.random.uniform(1.2, min(1.8, (w_img - 50) / w_bbox_add - 0.05))
                scale_h = np.random.uniform(0.98 * scale_w, 1.02 * scale_w)
            elif max(w_bbox_add, h_bbox_add) < 300:
                scale_w = np.random.uniform(0.9, min(1.5, (w_img - 10) / w_bbox_add - 0.05))
                scale_h = np.random.uniform(0.98 * scale_w, 1.02 * scale_w)
            elif w_bbox_add > w_img - 5:
                scale_w = np.random.uniform(0.7, (w_img - 10) / w_bbox_add - 0.05)
                scale_h = np.random.uniform(0.98 * scale_w, 1.02 * scale_w)
            elif h_bbox_add > h_img - 5:
                scale_h = np.random.uniform(0.7, (h_img - 10) / h_bbox_add - 0.05)
                scale_w = np.random.uniform(0.98 * scale_h, 1.02 * scale_h)
            else:
                scale_h = np.random.uniform(0.8, (h_img - 10) / h_bbox_add - 0.1)
                scale_w = np.random.uniform(0.98 * scale_h, 1.02 * scale_h)

            if w_bbox_add * scale_w * h_bbox_add * scale_h == 0:
                continue
            # For (x0, y0) shift
            try:
                x_shifts = [np.random.randint(5, w_img - w_bbox_add * scale_w) for _ in range(10000)]
                y_shifts = [np.random.randint(5, h_img - h_bbox_add * scale_h) for _ in range(10000)]
            except ValueError:
                continue

            add_flag = False
            for x_shift, y_shift in zip(x_shifts, y_shifts):
                modified_bbox = [x_shift, y_shift, w_bbox_add * scale_w, h_bbox_add * scale_h]
                if IoUsLowerThanThres(modified_bbox, bboxes_keypoints, up_thr=up_iou_thr, down_thr=down_iou_thr) \
                   or \
                   len(bboxes_keypoints) == 0:
                       
                    add_flag = True
                    real_num_add += 1
                    break

            # if this ann_add not satisfies IoU with original bbox < 0.5, just continue
            if not add_flag:
                continue

            mask_add = coco.annToMask(ann_add)

            id_used[ann_add['id']] += 1
            ann_add['id'] = ann_add['id'] + max_orig_annid * (id_used[ann_add['id']])

            temp_ann_oks_list.append(oks_mean_dict[str(ann_add['image_id'])])
            ann_add['image_id'] = bg_img['id']
            ann_add['area'] = int(ann_add['area'] * scale_w * scale_h)
            modify_ann(ann_add, x_shift, y_shift, scale_w, scale_h)
            added_ann_list.append(ann_add)
            bboxes_keypoints.append(ann_add['bbox'])
            
            h_img_bbox = int(h_bbox_add * scale_h)
            w_img_bbox = int(w_bbox_add * scale_w)
            img_add = colorJitter(img_add)
            img_bbox_add = cv2.resize(img_add[y0: y1, x0: x1], (w_img_bbox, h_img_bbox))
            mask_bbox_add = cv2.resize(mask_add[y0: y1, x0: x1], (w_img_bbox, h_img_bbox))
            mask_bbox_add = np.expand_dims(mask_bbox_add, -1)
            I_bg[y_shift: y_shift + h_img_bbox, x_shift: x_shift + w_img_bbox] = \
                I_bg[y_shift: y_shift + h_img_bbox, x_shift: x_shift + w_img_bbox] * (1 - mask_bbox_add) + \
                img_bbox_add * mask_bbox_add
                
        if random_occlusion_thr > 0:
            I_bg, added_ann_list = image_occlusion(I_bg, added_ann_list, random_occlusion_thr=random_occlusion_thr)

       
        buffer_img_file_name = os.path.join(temp_buffer_img_dir, bg_img['file_name'])
        io.imsave(buffer_img_file_name, I_bg)
        with torch.no_grad():
            img_oks = model_eval_oks(
                model, 
                buffer_img_file_name, 
                added_ann_list
            )
        torch.cuda.empty_cache()
        
        if img_oks < 0:
            print('-' * 30, 'oks = -1')
            os.remove(buffer_img_file_name)
            continue
        
        target_interval_id = min(int(img_oks * 10 / 2), 4)
        
        if target_interval_id != syn_interval_id:
            continue
        
    
        i += 1
        print("[{}/{}] add {} annotations on image {}".format(i, len(bg_imgs), real_num_add, bg_img['file_name']), end='\r')
        coco_synthesize_list[target_interval_id]['images'].append(bg_img)
        coco_synthesize_list[target_interval_id]['annotations'].extend(added_ann_list)
        os.popen(f"cp {buffer_img_file_name} {os.path.join(syn_imgDir, f'epoch_{target_interval_id}', bg_img['file_name'])}")

        
        if (i - start_i) % 200 == 0:
            print('write to json', end='\r')
            for dir_id in range(5):
                temp_buffer_ann_dir_name = os.path.join(mid_buffer_ann_dir, f"ann_{dir_id}.json")
                with open(temp_buffer_ann_dir_name, 'w') as fd:
                    json.dump(coco_synthesize_list[dir_id], fd)
            print('finish write to json', end='\r')
                
    print('write to json', end='\r')
    for dir_id in range(5):
        temp_buffer_ann_dir_name = os.path.join(mid_buffer_ann_dir, f"ann_{dir_id}.json")
        with open(temp_buffer_ann_dir_name, 'w') as fd:
            json.dump(coco_synthesize_list[dir_id], fd)
    print('finish write to json', end='\r')
        
    del model
    torch.cuda.empty_cache()
    print(f"{process_id} finish")

def syn_kpts(
        syn_imgDir, syn_json_file, 
        up_condition_thr, down_condition_thr,
        choose_img_ids, other_img_ids, occlusion_num=2000,
        pose_num=4, random_occlusion_thr=0.5,
        up_iou_thr=0.3, down_iou_thr=0.2, 
        temp_buffer_img_dir=None,
        temp_buffer_ann_dir=None,
        oks_mean_dict=None,
        pose_oks_dict=None, 
        pose_oks_list=None,
        process_num=6,
        all_syn_img_num=64115,
        syn_interval_id=4,
        interval_id=0
    ):
    
    coco_synthesize = {}
    annTrainFile = ""
    if not os.path.exists(syn_imgDir):
        print(syn_imgDir)
        os.makedirs(syn_imgDir)

    coco = COCO(annTrainFile)
    coco_synthesize['images'] = []
    coco_synthesize['categories'] = [coco.dataset['categories'][0]]
    coco_synthesize['annotations'] = []

    InstanceAnnFile = ""
    coco_instances = COCO(InstanceAnnFile)
    person_catIds = coco_instances.getCatIds(catNms=['person'])
    person_imgIds = coco_instances.getImgIds(catIds=person_catIds)
    all_imgIds = coco_instances.getImgIds()
    bg_imgIds = list(set(all_imgIds) - set(person_imgIds))
    bg_imgs = coco_instances.loadImgs(bg_imgIds)  # 54172 images

    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    imgs = coco.loadImgs(imgIds)  # list[dict] 64115 images
    all_annIds = coco.getAnnIds(imgIds, catIds)
    all_anns = coco.loadAnns(all_annIds)
    id_used = {deepcopy(ann['id']): 0 for ann in all_anns}
    # ------------------------------------------------------------------------------
    random.shuffle(choose_img_ids)
    
    non_occlusion_img_ids = choose_img_ids[:]
    choose_img_ids = person_imgIds[:]
    if len(choose_img_ids) == 0:
        choose_img_ids = other_img_ids
        
    
    # ------------------------------------------------------------------------------

    buffers = []
    print(f"interval_id : {interval_id} choose_img_ids {len(choose_img_ids)}")
    for temp_id, temp_img_id in enumerate(choose_img_ids):
        # ==========================================================================
        if 0.2 * syn_interval_id < oks_mean_dict[str(temp_img_id)] < 0.2 * (syn_interval_id + 1):
            continue
        # ==========================================================================
        print(f"{temp_id} / {len(choose_img_ids)}", end='\r')
        img = coco.loadImgs(ids=temp_img_id)[0]
        try:
            I = io.imread("%s%s" %('', img['file_name']))
        except FileNotFoundError:
            I = io.imread("%s%s" %('', img['file_name']))
        if len(I.shape) == 0:
            continue
        if len(I.shape) == 2:
            I = np.expand_dims(I, -1).repeat(3, -1)
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        try:
            anns = coco.loadAnns(annIds)
        except KeyError:
            continue
        anns_keyopints = [ann for ann in anns if 'keypoints' in ann.keys()]
        useful_anns_keyopints = [ann for ann in anns_keyopints if ann['num_keypoints'] > 1 and len(ann['segmentation']) == 1]
        for ann_keypoints in useful_anns_keyopints:
            # ==========================================================================
            temp_ann_id = ann_keypoints['id']
            if pose_oks_dict[temp_ann_id] < 0:
                continue
            # ==========================================================================
            buffers.append([I, deepcopy(ann_keypoints), img])
    print("**" * 20)
    print(f"interval_id : {interval_id} has {len(buffers)} keypoints annotations can be used")
    single_process_img_num = all_syn_img_num // process_num
    
    
    process_list = []
    queue = mp.Queue()
    dir_time_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    for process_id in range(process_num):
        start_i = process_id * single_process_img_num
        end_i = (process_id + 1) * single_process_img_num
        if process_id == process_num - 1:
            end_i = all_syn_img_num
        temp_coco_synthesize = {
            'images'         : [],
            'annotations'    : [],
            'categories'     : [coco.dataset['categories'][0]]
        }
        temp_id_used = {deepcopy(ann['id']): (int(time.time()) + process_id) * 1400000 for ann in all_anns}
        

        p = mp.Process(
            target=final_syn_function, 
            args=(
                process_id,
                start_i, end_i,
                up_condition_thr, down_condition_thr,
                syn_imgDir, temp_buffer_img_dir, os.path.join(temp_buffer_ann_dir, f"ann_{dir_time_name}"),
                temp_coco_synthesize,
                coco, bg_imgs, buffers,
                temp_id_used, 
                pose_num, up_iou_thr, down_iou_thr, random_occlusion_thr,
                oks_mean_dict,
                syn_interval_id,
                queue
            )
        )
        p.start()
        process_list.append(p)
      
    for p in process_list:
        # print('-' * 30)
        while p.is_alive():
            while False == queue.empty():
                result = queue.get()
                coco_synthesize['images'].extend(result['images'])
                coco_synthesize['annotations'].extend(result['annotations'])
          
    for p in process_list:
        p.join()
    print('finish join')
            
    print("Save annotations into {}".format(syn_json_file))
    with open(syn_json_file, 'w') as fp:
        json.dump(coco_synthesize, fp)
    print("num of annotations: ", len(coco_synthesize["annotations"]))
    print("num of images: ", len(coco_synthesize["images"]))

def split_last_four_data(oks_mean_sort_list, interval_ids, start_thr, interval_thr):
    current_thr = start_thr - interval_ids * interval_thr
    start_index = 58496
    end_index = start_index + (interval_ids - 5) * 1405
    current_img_ids_list = []
    choose_oks_list = oks_mean_sort_list[-end_index:]
    for temp_oks in choose_oks_list:
        current_img_ids_list.append(temp_oks['img_id'])
    print('current thr :', current_thr, ' length : ', len(current_img_ids_list))
    return current_img_ids_list

def get_larger_oks(oks_mean_sort_list, oks_thr):
    result_list = []
    for temp_oks in oks_mean_sort_list:
        if temp_oks['oks'] >= oks_thr:
            result_list.append(temp_oks['img_id'])
            
    return result_list

def get_condition_thr(interval_ids):
    down_condition_thr = 0.8 - interval_ids * 0.2
    up_condition_thr = down_condition_thr + 0.2
    
    return up_condition_thr, down_condition_thr


def get_every_pose_oks(coco, oks_dict=None):
    oks_pose_dict = {}
    oks_pose_list = []
    for key in oks_dict:
        result_dict, result_list = evaluateImg_every_pose(oks_dict[key], coco, int(key))
        oks_pose_dict.update(result_dict)
        oks_pose_list.extend(result_list)
        
    return oks_pose_dict, oks_pose_list


def get_pose_oks_dict_and_list(coco, ann_file, oks_file):
    oks_ann = json.load(open(oks_file, 'r'))
    
    no_person_list, person_list = get_no_peron(coco, ann_file)
    for img_id in no_person_list:
        del oks_ann[str(img_id)]
    assert len(oks_ann) == len(person_list)
    oks_pose_dict, oks_pose_list = get_every_pose_oks(coco, oks_dict=oks_ann)
    
    oks_pose_sort_list = sorted(oks_pose_list, key=lambda temp_var : temp_var['oks'])
    count_arr = [0 for _ in range(11)]
    for temp_oks in oks_pose_sort_list:
        count_arr[int(temp_oks['oks'] * 10)] += 1
    print(count_arr)
    return oks_pose_dict, oks_pose_sort_list

def cvpr_syn_kpts(
        coco, 
        oks_mean_sort_list, oks_mean_dict,
        pose_oks_dict, pose_oks_list,
        fun_for_pick_img, fun_for_condition_thr,
        start_thr = 0.6, interval_num = 16,
        img_target_dir=None,
        ann_target_dir=None,
        img_dir_name=None,
        temp_buffer_img_dir=None,
        temp_buffer_ann_dir=None,
        process_num=6,
        pose_num=6,
        all_syn_img_num=64115,
        syn_interval_id=4,
        up_iou_thr=0.6,
        random_occlusion_thr=0.02,
        json_file_name=None,
    ):

    if not os.path.exists(img_target_dir):
        os.mkdir(img_target_dir)
    if not os.path.exists(ann_target_dir):
        os.mkdir(ann_target_dir)
    if not os.path.exists(temp_buffer_img_dir):
        os.mkdir(temp_buffer_img_dir)
    if not os.path.exists(temp_buffer_ann_dir):
        os.mkdir(temp_buffer_ann_dir)
        
    for i in range(5):
        t_path = os.path.join(img_target_dir, f"epoch_{i}")
        if not os.path.exists(t_path):
            os.mkdir(t_path)

    interval_thr = start_thr / interval_num
    start_thr -= interval_thr

    interval_ids_list = [0]
    process_list = []
    for interval_ids in range(interval_num):
        if interval_ids not in interval_ids_list:
            continue
        
        pick_img_ids = fun_for_pick_img(oks_mean_sort_list, interval_ids, start_thr, interval_thr)
        other_img_ids = get_larger_oks(oks_mean_sort_list, max(0, 1 - (interval_ids + 2) * 0.2))
        up_condition_thr, down_condition_thr = fun_for_condition_thr(interval_ids)
        print('larger oks : ', max(0, 1 - (interval_ids + 4) * 0.2))
        print(f'interval_ids : {interval_ids}, pick_img_length : {len(pick_img_ids)}')
        print(f'{img_dir_name}_{interval_ids}')
        print(f"{json_file_name}_{interval_ids}.json")
        print(f"up_confition_thr : {up_condition_thr} down_condition_thr : {down_condition_thr}")

        down_iou_thr = up_iou_thr - 0.1
        
        
        syn_kpts(
            syn_imgDir=img_target_dir, 
            syn_json_file=os.path.join(ann_target_dir, f"{json_file_name}_{interval_ids}.json"),
            choose_img_ids=pick_img_ids,
            other_img_ids=other_img_ids,
            up_condition_thr=up_condition_thr,
            down_condition_thr=down_condition_thr,
            occlusion_num=40000 - interval_ids * 2000,
            pose_num=pose_num,
            random_occlusion_thr=random_occlusion_thr,
            up_iou_thr=up_iou_thr,
            down_iou_thr=down_iou_thr,
            temp_buffer_img_dir=temp_buffer_img_dir,
            temp_buffer_ann_dir=temp_buffer_ann_dir,
            oks_mean_dict=oks_mean_dict,
            pose_oks_dict=pose_oks_dict, 
            pose_oks_list=pose_oks_list,
            process_num=process_num,
            all_syn_img_num=all_syn_img_num,
            interval_id=interval_ids,
            syn_interval_id=syn_interval_id,
        )

    for p in process_list:
        p.join()
    
    
if __name__ == '__main__':
    ann_file = ''
    oks_file = ''
    coco = COCO(ann_file)
    oks_mean_sort_list = get_oks_mean_list(coco, ann_file=ann_file, oks_file=oks_file)
    oks_mean_dict = get_oks_mean_dict(coco, ann_file=ann_file, oks_file=oks_file)
    oks_file = ''
    pose_oks_dict, pose_oks_list = get_pose_oks_dict_and_list(coco, ann_file=ann_file, oks_file=oks_file)
    
    
    val_ann_file = ''
    oks_val_file = ''
    coco = COCO(val_ann_file)
    oks_mean_sort_list = get_oks_mean_list(coco, ann_file=val_ann_file, oks_file=oks_val_file)
    oks_mean_dict = get_oks_mean_dict(coco, ann_file=val_ann_file, oks_file=oks_val_file)
    oks_val_file = ""
    pose_oks_dict, pose_oks_list = get_pose_oks_dict_and_list(coco, ann_file=val_ann_file, oks_file=oks_val_file)

    