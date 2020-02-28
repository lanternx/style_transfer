import argparse
import time
import numpy as np
import h5py
from skimage import color, exposure, transform
from scipy import ndimage
from scipy.ndimage.filters import median_filter

import nets #python file
import loss #python file
import img_util

from keras.preprocessing.image import ImageDataGenerator
from matplotlib.pyplot import imsave
import tensorflow as tf

def display_img(i, x, output_path, style, is_val=False):
    # save current generated image
    img = x #deprocess_image(x)
    if is_val:
        if output_path:
            fname = output_path
        else:
            fname = 'images/output/%s_%d_val.png' % (style, i)
    else:
        if output_path:
            fname = 'images/output/%s_%d.png' % (style, i)
    imsave(fname, img)
    print('Image saved as', fname)

def batch_generator(x_generator, y):
    for x in x_generator:
        yield (x, y)

def train(args):
    style_weight = args.style_weight
    content_weight = args.content_weight
    tv_weight = args.tv_weight
    output_path = args.output
    img_width, img_height = [int(x) for x in args.image_size.split("*")]
    style = args.style
    style_image_path = "images/style/"+style+".jpg"
    net = nets.image_transform_net(img_width, img_height, tv_weight)
    model = nets.loss_net(net, img_width, img_height, style_image_path, content_weight, style_weight)
    model.summary()
    nb_epoch = 2000
    train_batchsize = 1
    train_image_path = "images/train/"
    model.compile(optimizer="adam", loss=loss.dummy_loss)  # Dummy loss since we are learning from regularizes
    datagen = ImageDataGenerator()
    dummy_y = np.zeros((train_batchsize, img_width, img_height, 3)) # Dummy output, not used since we use regularizers to train
    skip_to = 0
    i=0
    t1 = time.time()
    """
    for x in datagen.flow_from_directory(train_image_path, class_mode=None, batch_size=train_batchsize, target_size=(img_width, img_height), shuffle=True):
        if i > nb_epoch:
            break
        if i < skip_to:
            i+=train_batchsize
            if i % 1000 ==0:
                print("skip to: %d" % i)
            continue
        hist = model.train_on_batch(x, dummy_y)
        if i % 50 == 0:
            print(hist, (time.time() -t1))
            t1 = time.time()
        if i % 500 == 0:
            print("epoc: ", i)
            val_x = net.predict(x)
            display_img(i, x[0], output_path, style)
            display_img(i, val_x[0], output_path, style, True)
            model.save_weights("pretrained/"+style+'_weights.h5')
        i += train_batchsize
    """
    x_generator = datagen.flow_from_directory(train_image_path, class_mode=None, batch_size=train_batchsize, target_size=(img_width, img_height), shuffle=True)
    dummy_y = np.zeros((train_batchsize, img_width, img_height, 3))
    model.fit_generator(batch_generator(x_generator, dummy_y), steps_per_epoch=2000, epochs=2000)
    model.save_weights("pretrained/"+style+'_weights.h5')


# from 6o6o's fork. https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py
def original_colors(original, stylized,original_color):
    # Histogram normalization in v channel
    ratio=1. - original_color 
    hsv = color.rgb2hsv(original/255)
    hsv_s = color.rgb2hsv(stylized/255)
    hsv_s[:,:,2] = (ratio* hsv_s[:,:,2]) + (1-ratio)*hsv [:,:,2]
    img = color.hsv2rgb(hsv_s)    
    return img


def blend(original, stylized, alpha):
    return alpha * original + (1 - alpha) * stylized


def median_filter_all_colours(im_small, window_size):
    """
    Applies a median filer to all colour channels
    """
    ims = []
    for d in range(3):
        im_conv_d = median_filter(im_small[:,:,d], size=(window_size,window_size))
        ims.append(im_conv_d)
    im_conv = np.stack(ims, axis=2).astype("uint8")
    return im_conv


def load_weights(model,file_path):
    f = h5py.File(file_path)
    layer_names = [name for name in f.attrs['layer_names']]
    for i, layer in enumerate(model.layers[:31]):
        g = f[layer_names[i]]
        weights = [g[name] for name in g.attrs['weight_names']]
        layer.set_weigh
        ts(weights)
    f.close()
    print('Pretrained Model weights loaded.')


def transform(args):
    style= args.style
    #img_width = img_height =  args.image_size
    output_file =args.output
    input_file = args.input
    original_color = args.original_color
    blend_alpha = args.blend
    media_filter = args.media_filter
    aspect_ratio, x = img_util.preprocess_reflect_image(input_file, size_multiple=4)
    img_width = img_height = x.shape[1]
    net = nets.image_transform_net(img_width, img_height)
    model = nets.loss_net(net, img_width, img_height, "", 0, 0)
    model.compile("adam", loss.dummy_loss)  # Dummy loss since we are learning from regularizes
    model.load_weights("pretrained/"+style+'_weights.h5', by_name=False)
    t1 = time.time()
    y = net.predict(x)[0] 
    y = crop_image(y, aspect_ratio)
    print("process: %s" % (time.time() -t1))
    ox = crop_image(x[0], aspect_ratio)
    y =  median_filter_all_colours(y, media_filter)
    if blend_alpha > 0:
        y = blend(ox,y,blend_alpha)
    if original_color > 0:
        y = original_colors(ox,y,original_color )
    imsave('%s_output.png' % output_file, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time style transfer')
    parser.add_argument('--mode', '-m', type=str, required=True, 
                        help="choose to train or to generate new pic. Modes range from train to tranform. ")
    parser.add_argument('--style', '-s', type=str, required=True,
                        help='style image file name. Require')
    parser.add_argument('--input', '-i', default=None, type=str,
                        help='input file name. Require')
    parser.add_argument('--output', '-o', default=None, type=str,
                        help='output model file name. Not require')
    parser.add_argument('--tv_weight', default=1e-6, type=float,
                        help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6. Not require')
    parser.add_argument('--content_weight', default=1.0, type=float)
    parser.add_argument('--style_weight', default=4.0, type=float)
    parser.add_argument('--image_size', default="244*244", type=str, 
                        help='input image size, sample format is "244*244". Not require')
    parser.add_argument('--original_color', '-c', default=0, type=float,
                        help='0~1 for original color. Not require')
    parser.add_argument('--blend', '-b', default=0, type=float,
                        help='0~1 for blend with original image. Not require')
    parser.add_argument('--media_filter', '-f', default=3, type=int,
                        help='media_filter size. Not require')
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "transform":
        transform(args)
    else:
        print("Mode choose error!")


# deprecated util
# init = tf.compat.v1.initialize_all_variables()
# sess = tf.compat.v1.Session()
# graph = tf.compat.v1.get_default_graph()
# sess.run(init)