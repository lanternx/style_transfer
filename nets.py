import vgg16
import layers
import img_util
import loss

from keras.layers.merge import concatenate
from keras.models import Input, Model
import keras.backend as K

def add_style_loss(vgg,style_image_path,vgg_layers,vgg_output_dict,img_width, img_height,weight):
    style_img = img_util.preprocess_image(style_image_path, img_width, img_height)
    print('Getting style features from VGG network.')
    style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
    style_layer_outputs = []
    for layer in style_layers:
        style_layer_outputs.append(vgg_output_dict[layer])
    vgg_style_func = K.function([vgg.layers[-19].input], style_layer_outputs)
    style_features = vgg_style_func([style_img])
    for i, layer_name in enumerate(style_layers):
        layer = vgg_layers[layer_name]
        feature_var = K.variable(style_features[i][0])
        style_loss = loss.StyleReconstructionRegularizer(
                            style_feature_target=feature_var, 
                            weight=weight)(layer)
        layer.add_loss(style_loss)


def add_content_loss(vgg_layers,vgg_output_dict,weight):
    content_layer = 'block4_conv2'
    content_layer_output = vgg_output_dict[content_layer]
    layer = vgg_layers[content_layer]
    content_regularizer = loss.FeatureReconstructionRegularizer(weight)(layer)
    layer.add_loss(content_regularizer)


def loss_net(res_net, width, height, style_image_path, content_weight, style_weight):
    x = concatenate([res_net.output, res_net.input], axis=0)
    x = layers.VGGNormalize(name="vgg_normalize")(x)
    vgg = vgg16.build(include_top=False, input_tensor=x)
    vgg_output_dict = dict([(layer.name, layer.output) for layer in vgg.layers[-18:]])
    vgg_layers = dict([(layer.name, layer) for layer in vgg.layers[-18:]])
    if style_weight > 0:
        add_style_loss(vgg, style_image_path, vgg_layers, vgg_output_dict, width, height, style_weight)
    if content_weight > 0:
        add_content_loss(vgg_layers, vgg_output_dict, content_weight)
    vgg.trainable = False
    for l in vgg.layers:
        l.trainable = False

    st_input = res_net.input
    x = res_net(st_input)
    for i in range(1, len(vgg.layers)):
        x = vgg.layers[i](x)
    model = Model(st_input, x)
    return model


def add_total_variation_loss(transform_output_layer, weight):
    # Total Variation Regularization
    layer = transform_output_layer  # Output layer
    tv_regularizer = loss.TVRegularizer(weight)(layer)
    layer.add_loss(tv_regularizer)


def image_transform_net(img_width, img_height, tv_weight=1):
    x = Input(shape=(img_width,img_height,3), name="input")
    a = layers.InputNormalize()(x)
    a = layers.ReflectionPadding2D(padding=(40,40),input_shape=(img_width, img_height,3))(a)
    a = layers.conv_bn_relu(32, 9, 9, stride=(1,1))(a)
    a = layers.conv_bn_relu(64, 9, 9, stride=(2,2))(a)
    a = layers.conv_bn_relu(128, 3, 3, stride=(2,2))(a)
    for _ in range(5):
        a = layers.res_conv(128,3,3)(a)
    a = layers.dconv_bn_nolinear(64, 3, 3)(a)
    a = layers.dconv_bn_nolinear(64, 3, 3)(a)
    a = layers.conv_bn_relu(3, 9, 9, stride=(1,1), relu=False)(a)
    y = layers.Denormalize(name='transform_output')(a)
    model = Model(inputs=x, outputs=y)
    if tv_weight > 0:
        add_total_variation_loss(model.layers[-1], tv_weight)
    return model 