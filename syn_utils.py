import os
import json
from pycocotools.coco import COCO
import cv2
import numpy as np

import random
import json
import copy


def get_no_peron(coco, file_ann):
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = set(coco.getImgIds(catIds=cat_ids))
    person_ids = coco.getImgIds(catIds=cat_ids)
    imgs = coco.loadImgs(ids=person_ids)
    img_ids = set()
    for img in imgs:
        ann_ids = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(ids=ann_ids)
        kpts_sum = 0
        for ann in anns:
            kpts_sum += ann['num_keypoints']
        if kpts_sum > 0:
            img_ids.add(img['id'])
    
    all_img_ids = set(coco.getImgIds())
    return list(all_img_ids - img_ids), list(img_ids)

def choose_background_ids_random(coco, background_img_ids, pose_img_height, pose_img_width):
    import random
    choose_background_id = -1
    random.shuffle(background_img_ids)
    shuffle_background_img_ids = background_img_ids
    for background_id in shuffle_background_img_ids:
        background_img_ann = coco.loadImgs(ids=background_id)[0]
        background_img_height = background_img_ann['height']
        background_img_width = background_img_ann['width']
        if pose_img_height <= background_img_height and pose_img_width <= background_img_width:
            choose_background_id = background_id
            break
    
    return choose_background_id

def choose_background_ids(coco, background_img_ids, pose_img_height, pose_img_width):
    import random
    choose_background_id = -1
    random.shuffle(background_img_ids)
    choose_background_id = background_img_ids[0]
    
    return choose_background_id

def syn_data(
        ann_file, coco, origin_img_ids, background_img_ids, syn_img_num, 
        origin_img_dir, img_target_dir, 
        ann_target_dir, ann_json_name
    ):
    if not os.path.exists(img_target_dir):
        os.mkdir(img_target_dir)
    if not os.path.exists(ann_target_dir):
        os.mkdir(ann_target_dir)

    file = json.load(open(ann_file, 'r'))

    random.shuffle(origin_img_ids)

    result_json = {
        'images' : [],
        'annotations' : [],
        'categories' : copy.deepcopy(file['categories']),
    }

    ''' add origin image '''
    # max_ann_ids : 900100581904    min_ann_ids : 183014
    # max_img_ids : 581929          min_img_ids : 9
    global_ann_id = 1000000000000
    global_img_id = 1000000

    for ori_ids, ori_img_ids in enumerate(origin_img_ids):
        print('ori img ', ori_ids, end='\r')
        img = coco.loadImgs(ids=ori_img_ids)[0]
        ann_ids = coco.getAnnIds(imgIds=[ori_img_ids])
        ann_list = coco.loadAnns(ids=ann_ids)
        os.popen(f"cp {os.path.join(origin_img_dir, img['file_name'])} {os.path.join(img_target_dir, img['file_name'])}")
        for ann in ann_list:
            result_json['images'].append(copy.deepcopy(img))
            result_json['annotations'].append(copy.deepcopy(ann))

    img_ids = -1
    while img_ids < syn_img_num:
        img_ids += 1
        print('syn image ids : ', img_ids, end='\r')
        ori_img_id = origin_img_ids[img_ids % len(origin_img_ids)]
        pose_img_ann = coco.loadImgs(ids=ori_img_id)[0]
        pose_img_height = pose_img_ann['height']
        pose_img_width = pose_img_ann['width']
        pose_ann_ids = coco.getAnnIds(imgIds=[ori_img_id])
        pose_ann_list = coco.loadAnns(ids=pose_ann_ids)
        all_kpts_num = 0
        for ann in pose_ann_list:
            all_kpts_num += ann['num_keypoints']
        if all_kpts_num == 0:
            syn_img_num += 1
            continue
        
        
        choose_background_id = choose_background_ids(coco, background_img_ids, pose_img_height, pose_img_width)
        if choose_background_id == -1:
            continue
        background_img_ann = coco.loadImgs(ids=choose_background_id)[0]

        pose_image = cv2.imread(os.path.join(origin_img_dir, pose_img_ann['file_name']))
        background_image = cv2.imread(os.path.join(origin_img_dir, background_img_ann['file_name']))
        background_image = cv2.resize(background_image, (pose_img_width, pose_img_height))
        final_image = np.zeros_like(background_image)
        final_image += background_image
        for pose_ids, pose_ann in enumerate(pose_ann_list):
            pose_mask = coco.annToMask(pose_ann)
            if pose_image.shape != pose_mask.shape:
                pose_mask = pose_mask[:, :, None]

            final_image[:pose_img_height, :pose_img_width, :] = pose_image * pose_mask + final_image[:pose_img_height, :pose_img_width, :] * (1 - pose_mask)
            temp_pose_ann = copy.deepcopy(pose_ann)
            temp_pose_ann['id'] = global_ann_id
            global_ann_id += 1
            temp_pose_ann['image_id'] = global_img_id
            result_json['annotations'].append(temp_pose_ann)
        
        final_img_file_name = str(global_img_id).zfill(12) + '.jpg'
        final_img_ann = copy.deepcopy(background_img_ann)
        final_img_ann['id'] = global_img_id
        final_img_ann['file_name'] = final_img_file_name
        final_img_ann['height'] = pose_img_height
        final_img_ann['width'] = pose_img_width
        result_json['images'].append(final_img_ann)
        global_img_id += 1
        cv2.imwrite(os.path.join(img_target_dir, final_img_file_name), final_image)
        
    with open(os.path.join(ann_target_dir, ann_json_name), 'w') as fd:
        json.dump(result_json, fd)

def get_oks_mean_list(coco, ann_file, oks_file):
    oks_ann = json.load(open(oks_file, 'r'))
    
    no_person_list, person_list = get_no_peron(coco, ann_file)
    for img_id in no_person_list:
        del oks_ann[str(img_id)]
    assert len(oks_ann) == len(person_list)
    oks_mean_dict, oks_mean_list = get_img_ids_according_oks(coco, oks_dict=oks_ann)
    oks_mean_sort_list = sorted(oks_mean_list, key=lambda temp_var : temp_var['oks'])
    count_arr = [0 for _ in range(11)]
    for temp_oks in oks_mean_sort_list:
        count_arr[int(temp_oks['oks'] * 10)] += 1
    return oks_mean_sort_list

def get_cross_oks_mean_list():
    oks_file_dir = ''
    ann_file_dir = ''
    oks_dict_list = [
        json.load(open(
            os.path.join(oks_file_dir, f'all_train_oks_gt_ann_id_{model_id}.json'), 'r'
        )) for model_id in range(5)
    ]
    
    oks_mean_list = []
    for train_data_id in range(5):
        ann_file = os.path.join(ann_file_dir, f'corss_model_train_{train_data_id}.json')
        coco = COCO(ann_file)
        cat_ids = coco.getCatIds(catNms=['person'])
        img_ids = coco.getImgIds(catIds=cat_ids)
        
        for img_id in img_ids:
            single_img_oks = 0
            for model_id in range(5):
                if model_id == train_data_id:
                    continue
                temp_ious = oks_dict_list[model_id][str(img_id)]
                temp_oks = evaluateImg(temp_ious, coco, img_id)
                single_img_oks += temp_oks
            single_img_oks /= len(oks_dict_list) - 1
            oks_mean_list.append(
                {
                    'img_id' : int(img_id),  
                    'oks'    : single_img_oks
                }
            )
    oks_mean_sort_list = sorted(oks_mean_list, key=lambda temp_var : temp_var['oks'])
    count_arr = [0 for _ in range(11)]
    for temp_oks in oks_mean_sort_list:
        count_arr[int(temp_oks['oks'] * 10)] += 1
            
    return oks_mean_sort_list

def get_val_cross_oks_mean_list():
    oks_file_dir = ''
    oks_dict_list = [
        json.load(open(
            os.path.join(oks_file_dir, f'all_val_oks_gt_ann_id_{model_id}.json'), 'r'
        )) for model_id in range(5)
    ]
    
    oks_mean_list = []
    
    ann_file = ''
    coco = COCO(ann_file)
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)
    
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ids=ann_ids)
        kpt_sum = 0
        for ann in anns:
            kpt_sum += ann['num_keypoints']
        if kpt_sum == 0:
            continue
        
        single_img_oks = 0
        for model_id in range(5):
            temp_ious = oks_dict_list[model_id][str(img_id)]
            
            temp_oks = evaluateImg(temp_ious, coco, img_id)
            single_img_oks += temp_oks
        single_img_oks /= len(oks_dict_list)
        oks_mean_list.append(
            {
                'img_id' : int(img_id),  
                'oks'    : single_img_oks
            }
        )
    oks_mean_sort_list = sorted(oks_mean_list, key=lambda temp_var : temp_var['oks'])
    count_arr = [0 for _ in range(11)]
    for temp_oks in oks_mean_sort_list:
        count_arr[int(temp_oks['oks'] * 10)] += 1
            
    return oks_mean_sort_list


def get_higherhrnet_cross_oks_mean_list():
    oks_file_dir = ''
    ann_file_dir = ''
    oks_dict_list = [
        json.load(open(
            os.path.join(oks_file_dir, f'all_train_oks_gt_ann_id_{model_id}.json'), 'r'
        )) for model_id in range(5)
    ]
    
    oks_mean_list = []
    for train_data_id in range(5):
        ann_file = os.path.join(ann_file_dir, f'corss_model_train_{train_data_id}.json')
        coco = COCO(ann_file)
        cat_ids = coco.getCatIds(catNms=['person'])
        img_ids = coco.getImgIds(catIds=cat_ids)
        
        for img_id in img_ids:
            # print(img_id)
            single_img_oks = 0
            for model_id in range(5):
                if model_id == train_data_id:
                    continue
                temp_ious = oks_dict_list[model_id][str(img_id)]
                temp_oks = evaluateImg(temp_ious, coco, img_id)
                single_img_oks += temp_oks
            single_img_oks /= len(oks_dict_list) - 1
            oks_mean_list.append(
                {
                    'img_id' : int(img_id),  
                    'oks'    : single_img_oks
                }
            )
    oks_mean_sort_list = sorted(oks_mean_list, key=lambda temp_var : temp_var['oks'])
    count_arr = [0 for _ in range(11)]
    for temp_oks in oks_mean_sort_list:
        count_arr[int(temp_oks['oks'] * 10)] += 1
            
    return oks_mean_sort_list

def get_higherhrnet_cross_oks_dict():
    oks_file_dir = ''
    ann_file_dir = ''
    oks_dict_list = [
        json.load(open(
            os.path.join(oks_file_dir, f'all_train_oks_gt_ann_id_{model_id}.json'), 'r'
        )) for model_id in range(5)
    ]
    
    oks_mean_dict = {}
    # oks_mean_list = []
    for train_data_id in range(5):
        print('train_data_id :', train_data_id)
        ann_file = os.path.join(ann_file_dir, f'corss_model_train_{train_data_id}.json')
        coco = COCO(ann_file)
        cat_ids = coco.getCatIds(catNms=['person'])
        img_ids = coco.getImgIds(catIds=cat_ids)
        
        for img_id in img_ids:
            single_img_oks = 0
            for model_id in range(5):
                if model_id == train_data_id:
                    continue
                temp_ious = oks_dict_list[model_id][str(img_id)]
                temp_oks = evaluateImg(temp_ious, coco, img_id)
                single_img_oks += temp_oks
            single_img_oks /= len(oks_dict_list) - 1
            oks_mean_dict[str(img_id)] = single_img_oks
            
    return oks_mean_dict

def crowdpose_get_higherhrnet_cross_oks_mean_list():
    oks_file_dir = ''
    ann_file_dir = ''
    oks_dict_list = [
        json.load(open(
            os.path.join(oks_file_dir, f'all_train_oks_gt_ann_id_{model_id}.json'), 'r'
        )) for model_id in range(5)
    ]
    
    oks_mean_list = []
    for train_data_id in range(5):
        ann_file = os.path.join(ann_file_dir, f'corss_model_train_{train_data_id}.json')
        coco = COCO(ann_file)
        cat_ids = coco.getCatIds(catNms=['person'])
        img_ids = coco.getImgIds(catIds=cat_ids)
        
        for img_id in img_ids:
            single_img_oks = 0
            for model_id in range(5):
                if model_id == train_data_id:
                    continue
                temp_ious = oks_dict_list[model_id][str(img_id)]
                temp_oks = evaluateImg(temp_ious, coco, img_id)
                single_img_oks += temp_oks
            single_img_oks /= len(oks_dict_list) - 1
            oks_mean_list.append(
                {
                    'img_id' : int(img_id),  
                    'oks'    : single_img_oks
                }
            )
    oks_mean_sort_list = sorted(oks_mean_list, key=lambda temp_var : temp_var['oks'])
    count_arr = [0 for _ in range(11)]
    for temp_oks in oks_mean_sort_list:
        count_arr[int(temp_oks['oks'] * 10)] += 1
            
    return oks_mean_sort_list


def crowdpose_get_higherhrnet_cross_oks_dict():
    oks_file_dir = ''
    ann_file_dir = ''
    oks_dict_list = [
        json.load(open(
            os.path.join(oks_file_dir, f'all_train_oks_gt_ann_id_{model_id}.json'), 'r'
        )) for model_id in range(5)
    ]
    
    oks_mean_dict = {}
    for train_data_id in range(5):
        print('train_data_id :', train_data_id)
        ann_file = os.path.join(ann_file_dir, f'corss_model_train_{train_data_id}.json')
        coco = COCO(ann_file)
        cat_ids = coco.getCatIds(catNms=['person'])
        img_ids = coco.getImgIds(catIds=cat_ids)
        
        for img_id in img_ids:
            single_img_oks = 0
            for model_id in range(5):
                if model_id == train_data_id:
                    continue
                temp_ious = oks_dict_list[model_id][str(img_id)]
                temp_oks = evaluateImg(temp_ious, coco, img_id)
                single_img_oks += temp_oks
            single_img_oks /= len(oks_dict_list) - 1
            oks_mean_dict[str(img_id)] = single_img_oks
            
    return oks_mean_dict



def crowdpose_get_val_higherhrnet_cross_oks_mean_list():
    oks_file_dir = ''
    oks_dict_list = [
        json.load(open(
            os.path.join(oks_file_dir, f'all_val_oks_gt_ann_id_{model_id}.json'), 'r'
        )) for model_id in range(5)
    ]
    
    oks_mean_list = []
    
    ann_file = ''
    coco = COCO(ann_file)
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)
    
    for img_id in img_ids:
        # print(img_id)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ids=ann_ids)
        kpt_sum = 0
        for ann in anns:
            kpt_sum += ann['num_keypoints']
        if kpt_sum == 0:
            continue
        
        single_img_oks = 0
        for model_id in range(5):
            if str(img_id) not in oks_dict_list[model_id]:
                print('-' * 30)
                continue
            temp_ious = oks_dict_list[model_id][str(img_id)]
            
            temp_oks = evaluateImg(temp_ious, coco, img_id)
            single_img_oks += temp_oks
        single_img_oks /= len(oks_dict_list)
        oks_mean_list.append(
            {
                'img_id' : int(img_id),  
                'oks'    : single_img_oks
            }
        )
    oks_mean_sort_list = sorted(oks_mean_list, key=lambda temp_var : temp_var['oks'])
    count_arr = [0 for _ in range(11)]
    for temp_oks in oks_mean_sort_list:
        count_arr[int(temp_oks['oks'] * 10)] += 1
            
    return oks_mean_sort_list



def get_higherhrnet_val_cross_oks_mean_list():
    oks_file_dir = ''
    oks_dict_list = [
        json.load(open(
            os.path.join(oks_file_dir, f'all_val_oks_gt_ann_id_{model_id}.json'), 'r'
        )) for model_id in range(5)
    ]
    
    oks_mean_list = []
    
    ann_file = ''
    coco = COCO(ann_file)
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)
    
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ids=ann_ids)
        kpt_sum = 0
        for ann in anns:
            kpt_sum += ann['num_keypoints']
        if kpt_sum == 0:
            continue
        
        single_img_oks = 0
        for model_id in range(5):
            temp_ious = oks_dict_list[model_id][str(img_id)]
            
            temp_oks = evaluateImg(temp_ious, coco, img_id)
            single_img_oks += temp_oks
        single_img_oks /= len(oks_dict_list)
        oks_mean_list.append(
            {
                'img_id' : int(img_id),  
                'oks'    : single_img_oks
            }
        )
    oks_mean_sort_list = sorted(oks_mean_list, key=lambda temp_var : temp_var['oks'])
    count_arr = [0 for _ in range(11)]
    for temp_oks in oks_mean_sort_list:
        count_arr[int(temp_oks['oks'] * 10)] += 1
    print(count_arr)
            
    return oks_mean_sort_list

def get_oks_mean_dict(coco, ann_file, oks_file):
    oks_ann = json.load(open(oks_file, 'r'))
    
    no_person_list, person_list = get_no_peron(coco, ann_file)
    for img_id in no_person_list:
        del oks_ann[str(img_id)]
    assert len(oks_ann) == len(person_list)
    oks_mean_dict, oks_mean_list = get_img_ids_according_oks(coco, oks_dict=oks_ann)

    return oks_mean_dict

def get_img_ids_according_oks(coco, oks_dict):
    oks_mean_dict = {}
    oks_mean_list = []
    for key in oks_dict:
        oks_mean_dict[key] = evaluateImg(oks_dict[key], coco, int(key))
        oks_mean_list.append(
            {
                'img_id' : int(key),  
                'oks'    : oks_mean_dict[key]
            }
        )
    return oks_mean_dict, oks_mean_list

def evaluateImg(ious_item, coco, img_id):
    '''
    perform evaluation for single category and image
    :return: dict (single image results)
    '''
    if len(ious_item) == 0:
        return 0
    
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
        if type(g) == int:
            print('*' * 30)
            return 0
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
    # =============================================================
    if len(ious) > 0:
        ious = ious[:, gtind] if len(ious) > 0 else ious
        gt_index = []
        temp_gt_list = []
        for temp_index, temp_gt in enumerate(gt):
            if temp_gt['num_keypoints'] > 0:
                gt_index.append(temp_index)
                temp_gt_list.append(temp_gt)
        gt_index = np.array(gt_index)
        gt = temp_gt_list
        ious = ious[:, gt_index]
    
    if len(gt) != ious.shape[1]:
        print(len(gt), ious.shape[1])
    assert len(gt) == ious.shape[1]
    
    if ious.shape[0] < ious.shape[1]:
        ious = np.concatenate([ious, np.zeros((ious.shape[1] - ious.shape[0], len(gt)))], axis=0)
    # =============================================================
    
    G = len(gt)
    gtm  = np.zeros((G))
    gtm[:] = -1
    gtIg = np.array([g['_ignore'] for g in gt])

    iou_sum = -1
    if not len(ious)==0:
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


def evaluateImg_backup(ious, coco, img_id):
    '''
    perform evaluation for single category and image
    :return: dict (single image results)
    '''
    # coco = COCO(ann_file)
    ann_ids = coco.getAnnIds(imgIds=img_id)
    gt = coco.loadAnns(ids=ann_ids)
    
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
    ious = np.array(ious)
    
    G = len(gt)
    gtm  = np.zeros((G))
    gtIg = np.array([g['_ignore'] for g in gt])

    gt_result_oks = [0 for _ in range(len(gt))]
    if not len(ious) == 0:
        for dind, d in enumerate(ious):
            # information about best match so far (m=-1 -> unmatched)
            iou = 0
            m   = -1
            for gind, g in enumerate(gt):
                # if this gt already matched, and not a crowd, continue
                if gtm[gind]>0 and not iscrowd[gind]:
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
            gt_result_oks[m] = max(gt_result_oks[m], iou)
            gtm[m]     = dind
        iou_avg = np.array(gt_result_oks).mean()
            
    return iou_avg



def syn_data_according_oks(
        coco, oks_mean_sort_list, fun_for_pick_img, start_thr = 0.6, interval_num = 16,
        img_target_dir=None,
        ann_target_dir=None,
        img_dir_name=None,
        json_file_name=None,
    ):
    ann_file = ''
    origin_img_dir = ''
    if not os.path.exists(img_target_dir):
        os.mkdir(img_target_dir)
    if not os.path.exists(ann_target_dir):
        os.mkdir(ann_target_dir)

    no_person_ids, person_ids = get_no_peron(coco, ann_file)
    train_dataset_num = len(person_ids)
    background_img_ids = no_person_ids

    interval_thr = start_thr / interval_num
    start_thr -= interval_thr
    for interval_ids in range(interval_num):
        pick_img_ids = fun_for_pick_img(oks_mean_sort_list, interval_ids, start_thr, interval_thr)
        print(f'interval_ids : {interval_ids}, pick_img_length : {len(pick_img_ids)}')
        print(f'{img_dir_name}_{interval_ids}')
        print(f"{json_file_name}_{interval_ids}.json")

        syn_data(
            ann_file, coco, pick_img_ids, background_img_ids, train_dataset_num - len(pick_img_ids), 
            origin_img_dir, img_target_dir=os.path.join(img_target_dir, f'{img_dir_name}_{interval_ids}'), 
            ann_target_dir=ann_target_dir, ann_json_name=f"{json_file_name}_{interval_ids}.json"
        )

        
def accumulation_pick_data(oks_mean_sort_list, interval_ids, start_thr, interval_thr):
    current_thr = start_thr - interval_ids * interval_thr
    current_img_ids_list = []
    for temp_oks in oks_mean_sort_list:
        if temp_oks['oks'] >= current_thr:
            current_img_ids_list.append(temp_oks['img_id'])
    return current_img_ids_list

def equal_split_data(oks_mean_sort_list, interval_ids, start_thr, interval_thr):
    current_start = start_thr - (interval_ids) * interval_thr
    current_end = current_start + interval_thr
    current_img_ids_list = []
    for temp_oks in oks_mean_sort_list:
        if current_start <= temp_oks['oks'] < current_end:
            current_img_ids_list.append(temp_oks['img_id'])
        if current_end == 1 and current_end == temp_oks['oks']:
            current_img_ids_list.append(temp_oks['img_id'])
    
    return current_img_ids_list

def wide_equal_split_data(oks_mean_sort_list, interval_ids, start_thr, interval_thr, interval_length=0.4):
    current_start = start_thr - interval_ids * interval_thr
    current_end = current_start + interval_length
    if current_end >= 0.94:
        current_end = 1
    current_img_ids_list = []
    for temp_oks in oks_mean_sort_list:
        if current_start <= temp_oks['oks'] < current_end:
            current_img_ids_list.append(temp_oks['img_id'])
        if current_end == 1 and current_end == temp_oks['oks']:
            current_img_ids_list.append(temp_oks['img_id'])

    return current_img_ids_list

def direct_split_data(oks_mean_sort_list, interval_ids, start_thr, interval_thr):
    current_start = 0.5
    current_end = 1
    current_img_ids_list = []
    for temp_oks in oks_mean_sort_list:
        if current_start <= temp_oks['oks'] < current_end:
            current_img_ids_list.append(temp_oks['img_id'])
        if current_end == 1 and current_end == temp_oks['oks']:
            current_img_ids_list.append(temp_oks['img_id'])
    
    return current_img_ids_list


if __name__ == '__main__':
    crowdpose_get_higherhrnet_cross_oks_mean_list()
