import numpy as np

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Sequential, Model
from keras.models import Input
from keras_applications.imagenet_utils import _obtain_input_shape

MEAN_PIXEL = [123.68, 116.779, 103.93]

def build(input_tensor=None, include_top=True, pooling=None):
    if input_tensor is None:
        input_shape = _obtain_input_shape(None, default_size=224, min_size=48, data_format='channels_last', require_flatten=include_top)
        #input_shape = (224, 224, 3)
        image_input = Input(shape=input_shape)
    else:
        image_input = Input(tensor= input_tensor)

    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="SAME", activation="relu", name="block1_conv1")(image_input)
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="SAME", activation="relu", name="block1_conv2")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="SAME", name="block1_pool")(x)
    x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="SAME", activation="relu", name="block2_conv1")(x)
    x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="SAME", activation="relu", name="block2_conv2")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="SAME", name="block2_pool")(x)
    x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="SAME", activation="relu", name="block3_conv1")(x)
    x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="SAME", activation="relu", name="block3_conv2")(x)
    x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="SAME", activation="relu", name="block3_conv3")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="SAME", name="block3_pool")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="SAME", activation="relu", name="block4_conv1")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="SAME", activation="relu", name="block4_conv2")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="SAME", activation="relu", name="block4_conv3")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="SAME", name="block4_pool")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="SAME", activation="relu", name="block5_conv1")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="SAME", activation="relu", name="block5_conv2")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="SAME", activation="relu", name="block5_conv3")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="SAME", name="block5_pool")(x)
    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    vgg16_model = Model(inputs=image_input, outputs=x, name="vgg16")
    if include_top:
        vgg16_model.load_weights("C:\\Users\\13630\\Desktop\\style transfer\\vgg16_weights_tf_dim_ordering_tf_kernels.h5", by_name=True)
    else:
        vgg16_model.load_weights("C:\\Users\\13630\\Desktop\\style transfer\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)
    return vgg16_model


if __name__ == "__main__":
    from keras.preprocessing import image
    from keras.applications.vgg16 import preprocess_input, decode_predictions
    import matplotlib.pyplot as plt
    model = build(include_top=True)
    img = image.load_img("D:\\Picture\\timg.jpg", target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    fe = model.predict(x)
    pred = decode_predictions(fe, top=5)[0]
    for e in pred:
        print(e[1], e[2])
    # values, labels = [], []
    # for e in pred:
    #     values.append(e[2])
    #     labels.append(e[1])
    # fig = plt.figure(u"TOP5 预测结果")
    # ax = fig.add_subplot(111)
    # ax.bar(range(len(values)), values, tick_label=labels, width=0.5, fc='g')
    # ax.set_ylabel(u'probability') 
    # ax.set_title(u'Top-5')
    # for a,b in zip(range(len(values)), values):
    #     ax.text(a, b+0.0005, '%.2f%%' % (b * 100), ha='center', va = 'bottom', fontsize=7)
    # fig = plt.gcf()
    # plt.show()

