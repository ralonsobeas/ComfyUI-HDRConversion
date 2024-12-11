import logging

import sys
sys.path.append("/project/aksoy-lab/Sebastian/intrinsic-inverse-tonemapping/baselines/SingleHDR")

logging.basicConfig(level=logging.INFO)
import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import tensorflow as tf
from dequantization_net import Dequantization_net
from linearization_net import Linearization_net
import hallucination_net
from refinement_net import Refinement_net
from util import apply_rf
import numpy as np
import cv2
import glob
import time

FLAGS = tf.app.flags.FLAGS
epsilon = 0.001
# ---

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--test_imgs', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
ARGS = parser.parse_args()

# ---

_clip = lambda x: tf.clip_by_value(x, 0, 1)

global rt_total
global rt_recon
global rt_ref

rt_total = []
rt_recon = []
rt_ref = []

def build_graph(
        ldr,  # [b, h, w, c]
        is_training,
):
    with tf.variable_scope("Dequantization_Net"):
        dequantization_model = Dequantization_net(is_train=is_training)
        C_pred = _clip(dequantization_model.inference(ldr))

    lin_net = Linearization_net()
    pred_invcrf = lin_net.get_output(C_pred, is_training)
    B_pred = apply_rf(C_pred, pred_invcrf)

    start_t = time.time()

    thr = 0.12
    alpha = tf.reduce_max(B_pred, reduction_indices=[3])
    alpha = tf.minimum(1.0, tf.maximum(0.0, alpha - 1.0 + thr) / thr)
    alpha = tf.reshape(alpha, [-1, tf.shape(B_pred)[1], tf.shape(B_pred)[2], 1])
    alpha = tf.tile(alpha, [1, 1, 1, 3])
    with tf.variable_scope("Hallucination_Net"):
        net_test, vgg16_conv_layers_test = hallucination_net.model(B_pred, ARGS.batch_size, False)
        y_predict_test = net_test.outputs
        y_predict_test = tf.nn.relu(y_predict_test)
        A_pred = (B_pred) + alpha * y_predict_test
    rec_t = time.time()
    rt_recon.append(rec_t-start_t)

    # Refinement-Net
    with tf.variable_scope("Refinement_Net"):
        refinement_model = Refinement_net(is_train=is_training)
        refinement_output = tf.nn.relu(refinement_model.inference(tf.concat([A_pred, B_pred, C_pred], -1)))

    ref_t = time.time()
    rt_ref.append(ref_t-rec_t)
    rt_total.append(ref_t-start_t)

    return refinement_output

ldr = tf.placeholder(tf.float32, [None, None, None, 3])
is_training = tf.placeholder(tf.bool)

HDR_out = build_graph(ldr, is_training)


class Tester:

    def __init__(self):
        return

    def test_it(self, path):
        ldr_imgs = glob.glob(os.path.join(path,'**','*.jpg'),recursive=True)
        #ldr_imgs = glob.glob(os.path.join(path,'*.png'))
        #ldr_imgs.extend(glob.glob(os.path.join(path, '*.jpg')))
        ldr_imgs = sorted(ldr_imgs)

        

        for ldr_img_path in ldr_imgs:
            print(ldr_img_path)
            ldr_img = cv2.imread(ldr_img_path)

            ldr_val = np.flip(ldr_img, -1).astype(np.float32) / 255.0

            ORIGINAL_H = ldr_val.shape[0]
            ORIGINAL_W = ldr_val.shape[1]

            """resize to 64x"""
            if ORIGINAL_H % 64 != 0 or ORIGINAL_W % 64 != 0:
                RESIZED_H = int(np.ceil(float(ORIGINAL_H) / 64.0)) * 64
                RESIZED_W = int(np.ceil(float(ORIGINAL_W) / 64.0)) * 64
                ldr_val = cv2.resize(ldr_val, dsize=(RESIZED_W, RESIZED_H), interpolation=cv2.INTER_CUBIC)
            
            padding = 32
            ldr_val = np.pad(ldr_val, ((padding, padding), (padding, padding), (0, 0)), 'symmetric')

            HDR_out_val = sess.run(HDR_out, {
                ldr: [ldr_val],
                is_training: False,
            })



            HDR_out_val = np.flip(HDR_out_val[0], -1)
            HDR_out_val = HDR_out_val[padding:-padding, padding:-padding]
            if ORIGINAL_H % 64 != 0 or ORIGINAL_W % 64 != 0:
                HDR_out_val = cv2.resize(HDR_out_val, dsize=(ORIGINAL_W, ORIGINAL_H), interpolation=cv2.INTER_CUBIC)
            #cv2.imwrite(os.path.join(ARGS.output_path, os.path.split(ldr_img_path)[-1][:-3]+'exr'), HDR_out_val)


        print(f'Average time for reconstruction: {np.mean(rt_recon)}')
        print(f'Average time for refinement: {np.mean(rt_ref)}')
        print(f'Average total runtime: {np.mean(rt_total)}')
        return

# ---

sess = tf.Session()
restorer = tf.train.Saver()
restorer.restore(sess, ARGS.ckpt_path)

tester = Tester()

if not os.path.exists(ARGS.output_path):
    os.makedirs(ARGS.output_path)

tester.test_it(ARGS.test_imgs)