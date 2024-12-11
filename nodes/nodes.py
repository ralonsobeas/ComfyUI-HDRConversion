from ..IntrinsicHDR.dequantize_and_linearize import dequantize_and_linearize_run

from ..IntrinsicHDR.inference import inference

import imageio
import Imath
import numpy as np
import cv2
#import tensorflow as tf
import os




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



class HDRConversion:     

    @classmethod
    def INPUT_TYPES(cls):
               
        return {"required": {       
                    "image": ("IMAGE",),
                    "output_path": ("STRING", {"multiline": False, "default": "path/to/image"}),
                    "image_name": ("STRING", {"multiline": False, "default": "image.exr"}),
                    }
                }

    RETURN_TYPES = ()
    FUNCTION = "to_hdr"
    OUTPUT_NODE = True
    CATEGORY = "ElRanchito/HDR"

    def to_hdr(self, image,output_path,image_name):

        #root = "custom_nodes/ComfyUI-HDRConversion/IntrinsicHDR"
        #root = os.path.join("custom_nodes", "ComfyUI-HDRConversion", "IntrinsicHDR")
        root = os.path.normpath(os.path.join("custom_nodes", "ComfyUI-HDRConversion", "IntrinsicHDR"))
        root = root.replace("\\\\", "\\")

        print("IMAGE TYPE: ", type(image))

        save_image(image, root)


        # Run linearization
        
        #with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
        #dequantize_and_linearize_run("custom_nodes/ComfyUI-HDRConversion/IntrinsicHDR/tmp", root)
        dequantize_and_linearize_run(os.path.normpath(os.path.join("custom_nodes", "ComfyUI-HDRConversion", "IntrinsicHDR", "tmp")).replace("\\\\", "\\"), root)
        # Save the image in downloads folder
        #image.save(output + "/" + image_name)

        
        # Run inference
        #inference("custom_nodes/ComfyUI-HDRConversion/IntrinsicHDR/tmp",output_path,image_name)
        inference(os.path.normpath(os.path.join("custom_nodes", "ComfyUI-HDRConversion", "IntrinsicHDR", "tmp")).replace("\\\\", "\\"), output_path, image_name)
        #save_exr(image, output + "/" + image_name)

        # Remove tf stream
        #tf.compat.v1.reset_default_graph()

        
        return {}

