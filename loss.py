
import keras.backend as K
import keras.backend as Ke
from keras.regularizers import Regularizer
from keras.objectives import mean_squared_error


def dummy_loss(y_true, y_pred):
    return K.variable(0.0)


def gram_matrix(x):
    assert K.ndim(x) == 3
    features = Ke.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    shape = K.shape(x)
    C, W, H = (shape[0],shape[1], shape[2])
    cf = K.reshape(features ,(C,-1))
    gram = K.dot(cf, K.transpose(cf)) /  K.cast(C*W*H,dtype='float32')
    return gram


class StyleReconstructionRegularizer(Regularizer):
    """ Johnson et al 2015 https://arxiv.org/abs/1603.08155 """
    def __init__(self, style_feature_target, weight=1.0):
        self.style_feature_target = style_feature_target
        self.weight = weight
        self.uses_learning_phase = False
        super(StyleReconstructionRegularizer, self).__init__()
        self.style_gram = gram_matrix(style_feature_target)

    def __call__(self, x):
        output = x.output[0] # Generated by network
        loss = self.weight * K.sum(K.mean(K.square((self.style_gram-gram_matrix(output))))) 
        return loss


class FeatureReconstructionRegularizer(Regularizer):
    """ Johnson et al 2015 https://arxiv.org/abs/1603.08155 """
    def __init__(self, weight=1.0):
        self.weight = weight
        self.uses_learning_phase = False
        super(FeatureReconstructionRegularizer, self).__init__()

    def __call__(self, x):
        generated = x.output[0] # Generated by network features
        content = x.output[1] # True X input features
        loss = self.weight *  K.sum(K.mean(K.square(content-generated)))
        return loss


class TVRegularizer(Regularizer):
    """ Enforces smoothness in image output. """
    def __init__(self, weight=1.0):
        self.weight = weight
        self.uses_learning_phase = False
        super(TVRegularizer, self).__init__()

    def __call__(self, x):
        assert K.ndim(x.output) == 4
        x_out = x.output
        shape = K.shape(x_out)
        img_width, img_height,channel = (shape[1],shape[2], shape[3])
        size = img_width * img_height * channel
        a = K.square(x_out[:, :img_width - 1, :img_height - 1, :] - x_out[:, 1:, :img_height - 1, :])
        b = K.square(x_out[:, :img_width - 1, :img_height - 1, :] - x_out[:, :img_width - 1, 1:, :])
        loss = self.weight * K.sum(K.pow(a + b, 1.25)) 
        return loss
