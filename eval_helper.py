#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import numpy as np
import paddle.fluid as fluid
import math
import box_utils
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from config import cfg
import six
from colormap import colormap
import cv2

#def clip_box(bbox, im_info):
#
#    h = im_info[0]
#    w = im_info[1]
#    print("########", bbox)
#    pts = bbox.reshape(4, 2)
#    pts[np.where(pts < 0)] = 1
#    pts[np.where(pts[:, 0] > w), 0] = w - 1
#    pts[np.where(pts[:, 1] > h), 1] = h - 1
#    pts = pts.reshape(-1)
#    pts /= im_info[2]

#    return pts

def get_dict(out, data, key):
    res = {}
    for i in range(len(key)):
        if i== 0:
            res[key[i]] = out
        else:
             res[key[i]] = data[i]
    return res

import numpy as np
import cv2
import Polygon as plg
import logging
logger = logging.getLogger(__name__)



def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    res_boxes = np.empty([1, 8], dtype='int32')
    res_boxes[0, 0] = int(points[0])
    res_boxes[0, 4] = int(points[1])
    res_boxes[0, 1] = int(points[2])
    res_boxes[0, 5] = int(points[3])
    res_boxes[0, 2] = int(points[4])
    res_boxes[0, 6] = int(points[5])
    res_boxes[0, 3] = int(points[6])
    res_boxes[0, 7] = int(points[7])
    point_mat = res_boxes[0].reshape([2, 4]).T
    return plg.Polygon(point_mat)


def clip_box(bbox, im_info):

    h = im_info[0]
    w = im_info[1]
    res = []
    for b in bbox:
        pts = b.reshape(4, 2)
        pts[np.where(pts < 0)] = 1
        pts[np.where(pts[:, 0] > w), 0] = w - 1
        pts[np.where(pts[:, 1] > h), 1] = h - 1
        pts = pts.reshape(-1)
        pts /= im_info[2]
        res.append(pts)

    return np.array(res)


def get_union(det, gt):
    area_det = det.area()
    area_gt = gt.area()
    return area_det + area_gt - get_intersection(det, gt)


def get_intersection_over_union(det, gt):
    try:
        return get_intersection(det, gt) / get_union(det, gt)
    except:
        return 0


def get_intersection(det, gt):
    inter = det & gt
    if len(inter) == 0:
        return 0
    return inter.area()


def parse_gt(result, im_id):
    #   objects = []
    for res in result:
        if res['im_id'][0][0][0] == im_id:
            gt_boxes = list(res['gt_box'][0])
            gt_class = res['gt_label'][0]
            is_difficult = res['is_difficult'][0].reshape(-1)
            objects = []
            for i in range(len(gt_boxes)):
                object_struct = {}
                object_struct['bbox'] = gt_boxes[i]
                object_struct['class'] = gt_class[i][0]
                if is_difficult[i] == 1:
                    object_struct['difficult'] = 1
                else:
                    object_struct['difficult'] = 0
                object_struct['im_id'] = im_id
                objects.append(object_struct)
            return objects


def caluc_ap(rec, prec):
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.
    return ap


def icdar_map(result, class_name, ovthresh):
    im_ids = []
    for res in result:
        im_ids.append(res['im_id'][0][0][0])
    recs = {}

    for i, im_id in enumerate(im_ids):
        recs[str(im_id)] = parse_gt(result, im_id)
    class_recs = {}
    npos = 0
    for k in im_ids:
        R = [obj for obj in recs[str(k)] if obj['class'] == class_name]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[k] = {'bbox': bbox, 'difficult': difficult, 'det': det}
    image_ids = []
    confidence = []
    BB = []
    for res in result:
        im_info = res['im_info']
        pred_boxes = res['bbox']
        for box in pred_boxes:
            if box[0] == class_name:
                image_ids.append(res['im_id'][0][0][0])
                confidence.append(box[1])
                clipd_box = clip_box(box[2:].reshape(-1, 8), im_info)
                BB.append(clipd_box[0])
    confidence = np.array(confidence)
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = np.array(BB)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        if BBGT.size > 0:
            # compute overlaps
            BBGT_xmin = np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni
            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]

            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):
                    p_g = polygon_from_points(BBGT_keep[index])
                    p_d = polygon_from_points(bb)
                    overlap = get_intersection_over_union(p_d, p_g)
                    overlaps.append(overlap)
                return overlaps

            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = caluc_ap(rec, prec)
    return rec, prec, ap


def icdar_map_eval(result, num_class):
    map = 0
    for i in range(num_class - 1):
        rec, prec, ap = icdar_map(result, i + 1, ovthresh=0.5)
        map = map + ap
    map = map / (num_class - 1)
    logger.info('mAP {}'.format(map))


def icdar_box_eval(result, thresh):

    matched_sum = 0
    num_global_care_gt = 0
    num_global_care_det = 0
    for res in result:
        #print(res['im_info'])
        im_info = res['im_info']
        h = im_info[1]
        w = im_info[2]
        gt_boxes = res['gt_box']
        pred_boxes = res['bbox']
        #print(pred_boxes)
        pred_boxes = pred_boxes[np.where(pred_boxes[:, 1] > thresh)]
        pred_boxes = pred_boxes[:, 2:]
        pred_boxes = clip_box(pred_boxes, im_info)

        is_difficult = res['is_difficult']
        det_matched = 0

        iou_mat = np.empty([1, 1])

        gt_pols = []
        det_pols = []

        gt_pol_points = []
        det_pol_points = []

        gt_dont_care_pols_num = []
        det_dont_care_pols_num = []

        det_matched_nums = []

        points_list = list(gt_boxes)

        dony_care = is_difficult.reshape(-1)
        for i, points in enumerate(points_list):
            gt_pol = polygon_from_points(list(points))
            gt_pols.append(gt_pol)
            gt_pol_points.append(list(points))
            if dony_care[i] == 1:
                gt_dont_care_pols_num.append(len(gt_pols) - 1)
        for i, points in enumerate(pred_boxes):
            points = list(points.reshape(8).astype(np.int32))
            det_pol = polygon_from_points(points)
            det_pols.append(det_pol)
            det_pol_points.append(points)
            if len(gt_dont_care_pols_num) > 0:
                for dont_care_pol in gt_dont_care_pols_num:
                    dont_care_pol = gt_pols[dont_care_pol]
                    intersected_area = get_intersection(dont_care_pol, det_pol)
                    pd_dimensions = det_pol.area()
                    precision = 0 if pd_dimensions == 0 else intersected_area / pd_dimensions
                    if (precision > 0.5):
                        det_dont_care_pols_num.append(len(det_pols) - 1)
                        break
        if len(gt_pols) > 0 and len(det_pols) > 0:
            # Calculate IoU and precision matrixs
            output_shape = [len(gt_pols), len(det_pols)]
            iou_mat = np.empty(output_shape)
            gt_rect_mat = np.zeros(len(gt_pols), np.int8)
            det_rect_mat = np.zeros(len(det_pols), np.int8)
            for gt_num in range(len(gt_pols)):
                for det_num in range(len(det_pols)):
                    p_d = gt_pols[gt_num]
                    p_g = det_pols[det_num]
                    iou_mat[gt_num, det_num] = get_intersection_over_union(p_d,
                                                                           p_g)

            for gt_num in range(len(gt_pols)):
                for det_num in range(len(det_pols)):
                    if gt_rect_mat[gt_num] == 0 and det_rect_mat[
                            det_num] == 0 and gt_num not in gt_dont_care_pols_num and det_num not in det_dont_care_pols_num:
                        if iou_mat[gt_num, det_num] > 0.5:
                            gt_rect_mat[gt_num] = 1
                            det_rect_mat[det_num] = 1
                            det_matched += 1
                            det_matched_nums.append(det_num)
        num_gt_care = (len(gt_pols) - len(gt_dont_care_pols_num))
        num_det_care = (len(det_pols) - len(det_dont_care_pols_num))
        matched_sum += det_matched
        num_global_care_gt += num_gt_care
        num_global_care_det += num_det_care
    method_recall = 0 if num_global_care_gt == 0 else float(
        matched_sum) / num_global_care_gt
    method_precision = 0 if num_global_care_det == 0 else float(
        matched_sum) / num_global_care_det
    method_hmean = 0 if method_recall + method_precision == 0 else 2 * method_recall * method_precision / (
        method_recall + method_precision)
    logger.info('Recall {}'.format(method_recall))
    logger.info('Precision {}'.format(method_precision))
    logger.info('F1 {}'.format(method_hmean))
    print('Recall {}'.format(method_recall))
    print('Precision {}'.format(method_precision))
    print('F1 {}'.format(method_hmean))

def icdar_eval(result, num_class):
    if num_class == 2:
        icdar_box_eval(result, 0.8)
    else:
        icdar_map_eval(result, num_class)
