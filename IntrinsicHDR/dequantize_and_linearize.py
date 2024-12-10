import logging

logging.basicConfig(level=logging.INFO)
import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from .baselines.SingleHDR.dequantization_net import Dequantization_net
from .baselines.SingleHDR.linearization_net import Linearization_net
from .baselines.SingleHDR.util import apply_rf
import numpy as np
import cv2
import glob
from PIL import Image


FLAGS = tf.app.flags.FLAGS
epsilon = 0.001

_clip = lambda x: tf.clip_by_value(x, 0, 1)

def build_graph(
        ldr,  # [b, h, w, c]
        is_training,
):
    """Build the graph for the single HDR model.
    Args:
        ldr: [b, h, w, c], float32
        is_training: bool
    Returns:
        B_pred: [b, h, w, c], float32
    """

    # dequantization
    print('dequantize ...')
    with tf.variable_scope("Dequantization_Net", reuse=tf.AUTO_REUSE):
        dequantization_model = Dequantization_net(is_train=is_training)
        C_pred = _clip(dequantization_model.inference(ldr))

    # linearization
    print('linearize ...')
    #with tf.variable_scope("Linearization_Net", reuse=tf.AUTO_REUSE):
    lin_net = Linearization_net()
    pred_invcrf = lin_net.get_output(C_pred, is_training)
    B_pred = apply_rf(C_pred, pred_invcrf)

    return B_pred

def build_session(root):
    """Build TF session and load models.
    Args:
        root: root path
    Returns:
        sess: TF session
    
    global sess  # Define sess as global
    if 'sess' in globals() and sess is not None:
        print("Using existing session.")
        return sess
    """
    global sess  # Define sess as global
    global graph  # Define graph as global

    if 'sess' in globals() and sess is not None:
        print("Using existing session.")
        return sess

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # Load models
    print("Loading session...")
    
    if 'graph' not in globals() or graph is None:
        graph = tf.compat.v1.Graph()
    else:
        print("Using existing graph.")
    
    # Reset the default graph to avoid conflicts
    tf.compat.v1.reset_default_graph()
    
    with graph.as_default():
        sess = tf.compat.v1.Session(config=config, graph=graph)
    
    return sess







    #sess.run(tf.compat.v1.global_variables_initializer())

    print('load models ...')



    return sess



def dequantize_and_linearize(ldr_img, sess, graph, ldr, is_training):
    """Dequantize and linearize LDR image.
    Args:
        ldr_img: [H, W, 3], uint8
        sess: TF session
        graph: TF graph
    Returns:
        linear_img: [H, W, 3], float32
    """
    print('preprocess ...')

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

    print('inference ...')

    """run inference"""
    lin_img = sess.run(graph, {
        ldr: [ldr_val],
        is_training: False,
    })

    """output transforms"""
    lin_img = np.flip(lin_img[0], -1)
    lin_img = lin_img[padding:-padding, padding:-padding]
    if ORIGINAL_H % 64 != 0 or ORIGINAL_W % 64 != 0:
        lin_img = cv2.resize(lin_img, dsize=(ORIGINAL_W, ORIGINAL_H), interpolation=cv2.INTER_CUBIC)

    return lin_img


def save_image(image, root):
    # Image tensor to normal numpy array
    image_np = image.cpu().numpy()

    print("IMAGE DIMENSIONS: ", image_np.shape)

    # Ensure the image has 4 dimensions (batch, height, width, channels)
    if image_np.ndim == 4:
        batch_size, height, width, channels = image_np.shape
        if channels not in [3, 4]:
            raise ValueError("Image must have 3 (RGB) or 4 (RGBA) channels")
        
        # Create the tmp directory if it doesn't exist
        tmp_dir = os.path.join(root, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        # Save each image in the batch
        for i in range(batch_size):
            img = image_np[i]
            # Normalize the image to the range [0, 255]
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)
            # Convert image to cv2 format (BGR)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # Save image
            cv2.imwrite(os.path.join(tmp_dir, f"tmp_{i}.png"), img_bgr)
    else:
        raise ValueError("Image must be 4D (batch, height, width, channels)")


def dequantize_and_linearize_run(test_imgs, root, start_id=0, end_id=None):

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_imgs', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--start_id',type=int, default=0)
    parser.add_argument('--end_id',type=int, default=None)
    args = parser.parse_args()
    """
    # Reset the default graph
    tf.compat.v1.reset_default_graph()
    # Get session
    sess = build_session(root)

    with sess.graph.as_default():

        # Define placeholders within the graph context
        ldr = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3])
        is_training = tf.compat.v1.placeholder(tf.bool)

        # Build the graph
        lin_graph = build_graph(ldr, is_training)



        #with build_session(root,graph) as sess:
        #sess.run(tf.compat.v1.global_variables_initializer())

        # If the model restored before, no need to restore again
        global restored
        #if 'restored' not in globals() or not restored:
        print("Restoring model...")
        restored = True


        restorer0 = tf.compat.v1.train.Saver(var_list=[var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'Dequantization_Net' in var.name])
        restorer0.restore(sess, root + '/baselines/SingleHDR/checkpoints/model.ckpt')

        

        # Define the variables to restore
        var_list = [var for var in tf.compat.v1.global_variables() if 'crf_feature_net' in var.name or 'ae_invcrf_' in var.name]

        # Create a Saver object for the variables
        restorer = tf.compat.v1.train.Saver(var_list=var_list)

        # Restore the variables from the checkpoint
        try:
            restorer.restore(sess, root + '/baselines/SingleHDR/checkpoints/model.ckpt')
            print("Model restored successfully from checkpoint.")
        except tf.errors.NotFoundError as e:
            #missing_vars = [var for var in var_list if not sess.run(tf.compat.v1.is_variable_initialized(var))]

            # Reinitialize missing variables
            sess.run(tf.compat.v1.variables_initializer(var_list))

        # Get images
        ldr_imgs = glob.glob(os.path.join(test_imgs, '*.png'))
        ldr_imgs.extend(glob.glob(os.path.join(test_imgs, '*.jpg')))
        ldr_imgs = sorted(ldr_imgs)[start_id:end_id]

        output_path = os.path.join(root, 'tmp')
        os.makedirs(output_path, exist_ok=True)

        for d, ldr_img_path in tqdm(enumerate(ldr_imgs), initial=start_id):
            print("Processing image " + ldr_img_path)

            # Load img and preprocess
            ldr_img = cv2.imread(ldr_img_path)
            ldr_img = cv2.cvtColor(ldr_img, cv2.COLOR_BGR2RGB)

            # Dequantize and linearize
            linear_img = dequantize_and_linearize(ldr_img, sess, lin_graph, ldr, is_training)

            # Save linear image
            cv2.imwrite(os.path.join(output_path, os.path.split(ldr_img_path)[-1][:-3] + 'exr'),
                        cv2.cvtColor(linear_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_COMPRESSION, 1])
            
    # Ensure the session is closed after use
    #sess.close()
        #sess.run(tf.global_variables_initializer())
            


    


    print('Finished!')
