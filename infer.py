#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import cv2
import time
import numpy as np
import pickle
from eval_helper import *
import paddle
import paddle.fluid as fluid
import reader
from utility import print_arguments, parse_args, check_gpu
import models.model_builder as model_builder
import models.resnet_pp as resnet_pp
#from models.resnet_pp import ResNetC5 as resnetc5
from config import cfg
from data_utils import DatasetPath
import checkpoint as checkpoint
#from train import *
from eval_helper import clip_box


def infer():


#    image_shape = [3, cfg.TEST.max_size, cfg.TEST.max_size]
#    class_nums = cfg.class_num

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    image_shape = [3, cfg.TEST.max_size, cfg.TEST.max_size]
    class_nums = cfg.class_num
#    model = model_builder.RCNN(
#        add_conv_body_func=resnet.add_ResNet50_conv4_body,
#        add_roi_box_head_func=resnet.add_ResNet_roi_conv5_head,
#        use_pyreader=False,
#        mode='infer')
    model = model_builder.RCNN(
        add_conv_body_func=resnet_pp.ResNet(),
        add_roi_box_head_func=resnet_pp.ResNetC5(),
        use_pyreader=False,
        mode='infer')

    startup_prog = fluid.Program()
    infer_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            model.build_model(image_shape)
            pred_boxes = model.eval_bbox_out()
    infer_prog = infer_prog.clone(True)
    exe.run(startup_prog)
    # yapf: disable
    def if_exist(var):
        return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
#    fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)
    if cfg.pretrained_model:
        #fluid.io.load_vars(exe, cfg.pretrained_model, infer_prog, predicate=if_exist)
        checkpoint.load_params(exe, infer_prog, cfg.pretrained_model)
#    cnt=0
#    if not os.path.exists(cfg.pretrained_model):
#        raise ValueError("Model path [%s] does not exist." % (cfg.pretrained_model))
#
#    all_vars = infer_prog.global_block().vars
#    for k, v in all_vars.items():
#        if v.persistable and ('weights' in k or 'biases' in k or 'scale' in k or 'offset' in k or '_w' in k or '_b' in k) and ('tmp' not in k) and ('velocity' not in k):
#                #bn_vars.append(k)
#                #paddle_vars[k] = ""
#            print(k)
#            print(np.array(fluid.global_scope().find_var(k).get_tensor()))
#            cnt += 1
#    print('weight sum ', cnt)

#    def if_exist(var):
#        return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
#    fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)
#    if cfg.pretrained_model:
        #fluid.io.load_vars(exe, cfg.pretrained_model, infer_prog, predicate=if_exist)
#        checkpoint.load_params(exe, infer_prog, cfg.pretrained_model)
        #print(infer_prog)
        #fluid.io.load_persistables(exe, cfg.pretrained_model, infer_prog)
    # yapf: enable
    infer_reader = reader.infer(cfg.image_path)
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    fetch_list = [pred_boxes]
    imgs = os.listdir(cfg.image_path)
    imgs.sort()

    for i, data in enumerate(infer_reader()):
        im_info = [data[0][1]]
        result = exe.run(infer_prog, fetch_list=[v.name for v in fetch_list],
                         feed=feeder.feed(data),
                         return_numpy=False)
        pred_boxes_v = result[0]
        if cfg.MASK_ON:
            masks_v = result[1]
        new_lod = pred_boxes_v.lod()
        nmsed_out = pred_boxes_v
        image = None
        im_scale = im_info[0][2]
        outs = np.array(nmsed_out)
        im_ = cv2.imread(os.path.join(cfg.image_path, imgs[i]))
        with open("./result_final/res_" + str(imgs[i].split(".")[0]) + ".txt", 'w') as f:
            for out in outs:
                if out[1] <= 0.8:
                    continue
                pts = clip_box(out[2:], im_info)
                pts = pts.reshape(4, 2).astype('int32')
               # out = out / im_scale
               # pts = out[2:].reshape(4, 2).astype('int32')
               # pts[np.where(pts < 0)] = 1

               # pts[np.where(pts[:,0] > w), 0] = w - 1
               # pts[np.where(pts[:,1] > h), 1] = h - 1
                #cv2.polylines(im_, [pts], True, (0, 255, 255), 1)
                rect = cv2.minAreaRect(pts)
                box = cv2.boxPoints(rect)
                cv2.polylines(im_, np.int32([box]), True, (255, 0, 0), 1)
                #f.write(str(int(out[2])) + "," + str(int(out[3])) + "," + str(int(out[4])) + "," + str(int(out[5])) + "," + \
                #        str(int(out[6])) + "," + str(int(out[7])) + "," + str(int(out[8])) + "," + str(int(out[9])) + "\n")
                f.write(str(int(pts[0][0])) + "," + str(int(pts[0][1])) + "," + str(int(pts[1][0])) + "," + str(int(pts[1][1])) + "," + \
                        str(int(pts[2][0])) + "," + str(int(pts[2][1])) + "," + str(int(pts[3][0])) + "," + str(int(pts[3][1])) + "\n")
            cv2.imwrite("./result/" + str(imgs[i].split(".")[0]) + ".jpg", im_)
#        exit() 
        print("finish " + str(imgs[i]) + "image")
        #draw_bounding_box_on_image(cfg.image_path, nmsed_out, cfg.draw_threshold,
        #                           labels_map, image)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    check_gpu(args.use_gpu)
    infer()
