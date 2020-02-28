import numpy as np

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
import tensorflow as tf

def preprocess_image(image_path,img_nrows,img_ncols):
    '''
    Preprocess the image so that it can be used by Keras.
    Args:
        image_path: path to the image
        img_width: image width after resizing. Optional: defaults to 256
        img_height: image height after resizing. Optional: defaults to 256
        load_dims: decides if original dimensions of image should be saved,
                   Optional: defaults to False
        vgg_normalize: decides if vgg normalization should be applied to image.
                       Optional: defaults to False
        resize: whether the image should be resided to new size. Optional: defaults to True
        size_multiple: Deconvolution network needs precise input size so as to
                       divide by 4 ("shallow" model) or 8 ("deep" model).
    Returns: an image of shape (3, img_width, img_height) for dim_ordering = "th",
             else an image of shape (img_width, img_height, 3) for dim ordering = "tf"
    '''
    img = image.load_img(image_path, target_size=(img_nrows, img_ncols), color_mode="rgb")
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
    

def preprocess_reflect_image(image_path, size_multiple=4):
    img = image.load_img(image_path, color_mode="rgb")  # Prevents crashes due to PNG images (ARGB)
    org_w = img.shape[0]
    org_h = img.shape[1]
    aspect_ratio = org_h/org_w
    sw = (org_w // size_multiple) * size_multiple # Make sure width is a multiple of 4
    sh = (org_h // size_multiple) * size_multiple # Make sure width is a multiple of 4
    size  = sw if sw > sh else sh
    pad_w = (size - sw) // 2
    pad_h = (size - sh) // 2
    kvar = K.variable(value=img)
    paddings = [[pad_w,pad_w],[pad_h,pad_h],[0,0]]
    squared_img = tf.pad(kvar,paddings, mode='REFLECT', name=None)
    img = K.eval(squared_img)
    img = tf.image.resize(img, (size, size), method='nearest')
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return (aspect_ratio, img)