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

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.initializer import Normal
from paddle.fluid.initializer import MSRA
from paddle.fluid.regularizer import L2Decay
from config import cfg
from models.ext_op.rrpn_lib import *

class RCNN(object):
    def __init__(self,
                 add_conv_body_func=None,
                 add_roi_box_head_func=None,
                 mode='train',
                 use_pyreader=True,
                 use_random=True):
        self.add_conv_body_func = add_conv_body_func
        self.add_roi_box_head_func = add_roi_box_head_func
        self.mode = mode
        self.use_pyreader = use_pyreader
        self.use_random = use_random

    def build_model(self, image_shape):
        self.build_input(image_shape)
        #fluid.layers.Print(self.image)
        body_conv = self.add_conv_body_func(self.image)
        # RPN
        #fluid.layers.Print(body_conv)
        self.rpn_heads(body_conv)
        # Fast RCNN
        self.fast_rcnn_heads(body_conv)
        if self.mode != 'train':
            self.eval_bbox()
        # Mask RCNN
        if cfg.MASK_ON:
            self.mask_rcnn_heads(body_conv)

    def loss(self):
        losses = []
        # Fast RCNN loss
        loss_cls, loss_bbox = self.fast_rcnn_loss()
        # RPN loss
        rpn_cls_loss, rpn_reg_loss = self.rpn_loss()
        losses = [loss_cls, loss_bbox, rpn_cls_loss, rpn_reg_loss]
        rkeys = ['loss', 'loss_cls', 'loss_bbox', \
                 'loss_rpn_cls', 'loss_rpn_bbox',]
        if cfg.MASK_ON:
            loss_mask = self.mask_rcnn_loss()
            losses = losses + [loss_mask]
            rkeys = rkeys + ["loss_mask"]
        loss = fluid.layers.sum(losses)
        rloss = [loss] + losses
        return rloss, rkeys, self.rpn_rois


    def eval_bbox_out(self):
        return self.pred_result

    def build_input(self, image_shape):
        if self.use_pyreader:
            in_shapes = [[-1] + image_shape, [-1, 5], [-1, 1], [-1, 1],
                         [-1, 3], [-1, 1]]
            lod_levels = [0, 1, 1, 1, 0, 0]
            dtypes = [
                'float32', 'float32', 'int32', 'int32', 'float32', 'int64'
            ]
            if cfg.MASK_ON:
                in_shapes.append([-1, 2])
                lod_levels.append(3)
                dtypes.append('float32')
            self.py_reader = fluid.layers.py_reader(
                capacity=64,
                shapes=in_shapes,
                lod_levels=lod_levels,
                dtypes=dtypes,
                use_double_buffer=True)
            ins = fluid.layers.read_file(self.py_reader)
            self.image = ins[0]
            self.gt_box = ins[1]
            self.gt_label = ins[2]
            self.is_crowd = ins[3]
            self.im_info = ins[4]
            self.im_id = ins[5]
            if cfg.MASK_ON:
                self.gt_masks = ins[6]
        else:
            print("######################")
            self.image = fluid.layers.data(
                name='image', shape=image_shape, dtype='float32')
            self.gt_box = fluid.layers.data(
                name='gt_box', shape=[4], dtype='float32', lod_level=1)
            self.gt_label = fluid.layers.data(
                name='gt_label', shape=[1], dtype='int32', lod_level=1)
            self.is_crowd = fluid.layers.data(
                name='is_crowd', shape=[1], dtype='int32', lod_level=1)
            self.im_info = fluid.layers.data(
                name='im_info', shape=[3], dtype='float32')
            self.im_id = fluid.layers.data(
                name='im_id', shape=[1], dtype='int64')

            self.difficult = fluid.layers.data(
                name='difficult', shape=[1], dtype='float32', lod_level=1)


    def feeds(self):
        if self.mode == 'infer':
            return [self.image, self.im_info]
        if self.mode == 'val':
            return [self.image, self.gt_box, self.gt_label, self.is_crowd,
                self.im_info, self.im_id, self.difficult]
        if not cfg.MASK_ON:
            return [
                self.image, self.gt_box, self.gt_label, self.is_crowd,
                self.im_info, self.im_id
            ]
        return [
            self.image, self.gt_box, self.gt_label, self.is_crowd, self.im_info,
            self.im_id, self.gt_masks
        ]

    def eval_bbox(self):
        self.im_scale = fluid.layers.slice(
        self.im_info, [1], starts=[2], ends=[3])
        im_scale_lod = fluid.layers.sequence_expand(self.im_scale,
                                                    self.rpn_rois)
        results = []
        boxes = self.rpn_rois
        cls_prob = fluid.layers.softmax(self.cls_score, use_cudnn=False)
        bbox_pred = fluid.layers.reshape(self.bbox_pred,
                                                 (-1, cfg.class_num, 5))
        for i in range(cfg.class_num-1):
            bbox_pred_slice = fluid.layers.slice(
                bbox_pred, axes=[1], starts=[i + 1], ends=[i + 2])
            bbox_pred_reshape = fluid.layers.reshape(bbox_pred_slice, (-1, 5))
            decoded_box = rrpn_box_coder(prior_box=boxes, \
                                         target_box=bbox_pred_reshape, \
                                         prior_box_var=[10.0, 10.0, 5.0, 5.0, 1.0])
            score_slice = fluid.layers.slice(
                cls_prob, axes=[1], starts=[i + 1], ends=[i + 2])
            score_slice = fluid.layers.reshape(score_slice, shape=[-1, 1])
            box_positive = fluid.layers.reshape(decoded_box, shape=[-1, 8])
            box_reshape = fluid.layers.reshape(x=box_positive, shape=[1, -1, 8])
            score_reshape = fluid.layers.reshape(
                x=score_slice, shape=[1, 1, -1])
            #pred_result = fluid.layers.multiclass_nms(bboxes=box_reshape, scores=score_reshape)
            pred_result = fluid.layers.multiclass_nms(
                bboxes=box_reshape,
                scores=score_reshape,
                score_threshold=0.01,
                nms_top_k=-1,
                nms_threshold=0.3,
                keep_top_k=300,
                normalized=False,
                background_label=-1)
            result_shape = fluid.layers.shape(pred_result)
            res_dimension = fluid.layers.slice(
                result_shape, axes=[0], starts=[1], ends=[2])
            res_dimension = fluid.layers.reshape(res_dimension, shape=[1, 1])
            dimension = fluid.layers.fill_constant(
                shape=[1, 1], value=2, dtype='int32')
            cond = fluid.layers.less_than(dimension, res_dimension)
            res = fluid.layers.create_global_var(
                shape=[1, 10], value=0.0, dtype='float32', persistable=False)
            with fluid.layers.control_flow.Switch() as switch:
                with switch.case(cond):
                    coordinate = fluid.layers.fill_constant(
                        shape=[9], value=0.0, dtype='float32')
                    pred_class = fluid.layers.fill_constant(
                        shape=[1], value=i + 1, dtype='float32')
                    add_class = fluid.layers.concat(
                        [pred_class, coordinate], axis=0)
                    normal_result = fluid.layers.elementwise_add(pred_result,
                                                                 add_class)
                    fluid.layers.assign(normal_result, res)
                with switch.default():
                    normal_result = fluid.layers.fill_constant(
                        shape=[1, 10], value=-1.0, dtype='float32')
                    fluid.layers.assign(normal_result, res)
            results.append(res)
        if len(results) == 1:
            self.pred_result=results[0]
            return
        outs = []
        out = fluid.layers.concat(results)
        zero = fluid.layers.fill_constant(
            shape=[1, 1], value=0.0, dtype='float32')
        out_split, _ = fluid.layers.split(out, dim=1, num_or_sections=[1, 9])
        out_bool = fluid.layers.greater_than(out_split, zero)
        idx = fluid.layers.where(out_bool)
        idx_split, _ = fluid.layers.split(idx, dim=1, num_or_sections=[1, 1])
        idx = fluid.layers.reshape(idx_split, [-1, 1])
        self.pred_result = fluid.layers.gather(input=out, index=idx)


    def rpn_heads(self, rpn_input):
        #fluid.layers.Print(rpn_input)
        # RPN hidden representation
        dim_out = rpn_input.shape[1]
        rpn_conv = fluid.layers.conv2d(
            input=rpn_input,
            num_filters=dim_out,
            filter_size=3,
            stride=1,
            padding=1,
            act='relu',
            name='conv_rpn',
            param_attr=ParamAttr(
                name="conv_rpn_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name="conv_rpn_b", learning_rate=2., regularizer=L2Decay(0.)))
        #fluid.layers.Print(rpn_conv)
        self.anchor, self.var = rotated_anchor_generator(
            input=rpn_conv,
            anchor_sizes=[128, 256, 512],
            aspect_ratios=[0.2, 0.5, 1.0],
            angles=(-30.0, 0.0, 30.0, 60.0, 90.0, 120.0),
            variance=[1.0, 1.0, 1.0, 1.0, 1.0],
            stride=[16.0, 16.0],
            offset=0.5)
        num_anchor = self.anchor.shape[2]
        # Proposal classification scores
        self.rpn_cls_score = fluid.layers.conv2d(
            rpn_conv,
            num_filters=num_anchor,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            name='rpn_cls_score',
            param_attr=ParamAttr(
                name="rpn_cls_logits_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name="rpn_cls_logits_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        # Proposal bbox regression deltas
        self.rpn_bbox_pred = fluid.layers.conv2d(
            rpn_conv,
            num_filters=5 * num_anchor,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            name='rpn_bbox_pred',
            param_attr=ParamAttr(
                name="rpn_bbox_pred_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name="rpn_bbox_pred_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        #fluid.layers.Print(self.rpn_cls_score)
        #fluid.layers.Print(self.rpn_bbox_pred)
        rpn_cls_score_prob = fluid.layers.sigmoid(
            self.rpn_cls_score, name='rpn_cls_score_prob')

        param_obj = cfg.TRAIN if self.mode == 'train' else cfg.TEST
        pre_nms_top_n = param_obj.rpn_pre_nms_top_n
        post_nms_top_n = param_obj.rpn_post_nms_top_n
        nms_thresh = param_obj.rpn_nms_thresh
        min_size = param_obj.rpn_min_size
        eta = param_obj.rpn_eta
        self.rpn_rois, self.rpn_roi_probs = rotated_generate_proposals(
            scores=rpn_cls_score_prob,
            bbox_deltas=self.rpn_bbox_pred,
            im_info=self.im_info,
            anchors=self.anchor,
            variances=self.var,
            pre_nms_top_n=6000,
            post_nms_top_n=1000,
            nms_thresh=0.7,
            min_size=0)
        
        #print("**************************")
        #fluid.layers.Print(self.rpn_rois)
        #fluid.layers.Print(self.gt_label)
        #fluid.layers.Print(self.gt_box)
        #fluid.layers.Print(self.rpn_rois, summarize=-1)
        if self.mode == 'train':
            outs = rotated_generate_proposal_labels(
                rpn_rois=self.rpn_rois,
                gt_classes=self.gt_label,
                is_crowd=self.is_crowd,
                gt_boxes=self.gt_box,
                im_info=self.im_info,
                batch_size_per_im=256,
                fg_fraction=0.25,
                fg_thresh=0.5,
                bg_thresh_hi=0.5,
                bg_thresh_lo=0.0,
                bbox_reg_weights=cfg.bbox_reg_weights,
                class_nums=2,
                use_random=True)

            self.rois = outs[0]
            self.labels_int32 = outs[1]
            self.bbox_targets = outs[2]
            self.bbox_inside_weights = outs[3]
            self.bbox_outside_weights = outs[4]
            #fluid.layers.Print(self.rois, summarize=-1)

    def fast_rcnn_heads(self, roi_input):
        if self.mode == 'train':
            pool_rois = self.rois
        else:
            pool_rois = self.rpn_rois
        pool = rotated_roi_align(
            input=roi_input,
            rois=pool_rois,
            pooled_height=cfg.roi_resolution,
            pooled_width=cfg.roi_resolution,
            spatial_scale=cfg.spatial_scale)
        self.res5_2_sum = self.add_roi_box_head_func(pool)
        rcnn_out = fluid.layers.pool2d(
            self.res5_2_sum, pool_type='avg', pool_size=7, name='res5_pool')
        #fluid.layers.Print(rcnn_out)
        self.cls_score = fluid.layers.fc(input=rcnn_out,
                                         size=cfg.class_num,
                                         act=None,
                                         name='cls_score',
                                         param_attr=ParamAttr(
                                             name='cls_score_w',
                                             initializer=Normal(
                                                 loc=0.0, scale=0.001)),
                                         bias_attr=ParamAttr(
                                             name='cls_score_b',
                                             learning_rate=2.,
                                             regularizer=L2Decay(0.)))
        self.bbox_pred = fluid.layers.fc(input=rcnn_out,
                                         size=5 * cfg.class_num,
                                         act=None,
                                         name='bbox_pred',
                                         param_attr=ParamAttr(
                                             name='bbox_pred_w',
                                             initializer=Normal(
                                                 loc=0.0, scale=0.01)),
                                         bias_attr=ParamAttr(
                                             name='bbox_pred_b',
                                             learning_rate=2.,
                                             regularizer=L2Decay(0.)))
        #fluid.layers.Print(self.cls_score)
        #fluid.layers.Print(self.bbox_pred)

    def fast_rcnn_loss(self):
        labels_int64 = fluid.layers.cast(x=self.labels_int32, dtype='int64')
        labels_int64.stop_gradient = True
        loss_cls = fluid.layers.softmax_with_cross_entropy(
            logits=self.cls_score,
            label=labels_int64,
            numeric_stable_mode=True, )
        loss_cls = fluid.layers.reduce_mean(loss_cls)
        loss_bbox = fluid.layers.smooth_l1(
            x=self.bbox_pred,
            y=self.bbox_targets,
            inside_weight=self.bbox_inside_weights,
            outside_weight=self.bbox_outside_weights,
            sigma=1.0)
        loss_bbox = fluid.layers.reduce_mean(loss_bbox)
        return loss_cls, loss_bbox

    def rpn_loss(self):
        rpn_cls_score_reshape = fluid.layers.transpose(
            self.rpn_cls_score, perm=[0, 2, 3, 1])
        rpn_bbox_pred_reshape = fluid.layers.transpose(
            self.rpn_bbox_pred, perm=[0, 2, 3, 1])

        anchor_reshape = fluid.layers.reshape(self.anchor, shape=(-1, 5))
        var_reshape = fluid.layers.reshape(self.var, shape=(-1, 5))

        rpn_cls_score_reshape = fluid.layers.reshape(
            x=rpn_cls_score_reshape, shape=(0, -1, 1))
        rpn_bbox_pred_reshape = fluid.layers.reshape(
            x=rpn_bbox_pred_reshape, shape=(0, -1, 5))
        score_pred, loc_pred, score_tgt, loc_tgt = \
            rrpn_target_assign(
                bbox_pred=rpn_bbox_pred_reshape,
                cls_logits=rpn_cls_score_reshape,
                anchor_box=anchor_reshape,
                #anchor_var=var_reshape,
                gt_boxes=self.gt_box,
                #is_crowd=self.is_crowd,
                im_info=self.im_info,
                rpn_batch_size_per_im=256,
                rpn_straddle_thresh=-1,
                rpn_fg_fraction=cfg.TRAIN.rpn_fg_fraction,
                rpn_positive_overlap=cfg.TRAIN.rpn_positive_overlap,
                rpn_negative_overlap=cfg.TRAIN.rpn_negative_overlap,
                use_random=True)
        score_tgt = fluid.layers.cast(x=score_tgt, dtype='float32')
        rpn_cls_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=score_pred, label=score_tgt)
        rpn_cls_loss = fluid.layers.reduce_mean(
            rpn_cls_loss, name='loss_rpn_cls')

        rpn_reg_loss = fluid.layers.smooth_l1(
            x=loc_pred,
            y=loc_tgt,
            sigma=3.0)#,
            #inside_weight=bbox_weight,
            #outside_weight=bbox_weight)
        rpn_reg_loss = fluid.layers.reduce_sum(
            rpn_reg_loss, name='loss_rpn_bbox')
        score_shape = fluid.layers.shape(score_tgt)
        score_shape = fluid.layers.cast(x=score_shape, dtype='float32')
        norm = fluid.layers.reduce_prod(score_shape)
        norm.stop_gradient = True
        rpn_reg_loss = rpn_reg_loss / norm
        return rpn_cls_loss, rpn_reg_loss
