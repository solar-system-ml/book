"""Import the basics: numpy, pandas, matplotlib, etc."""
import numpy as np
import pickle, os
"""Import keras and other ML tools"""
import tensorflow as tf
import keras
import keras.backend as K

from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPool2D, Concatenate
from keras.layers.core import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

#for custom keras loss function
from keras.layers import Lambda, Multiply,Add
from tensorflow.python.keras.losses import Loss, LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.util.tf_export import keras_export


#the custom loss below comes from Bialek, Fabbro, et al. (2020)


#elu modified
def elu_plus(x):
    return K.elu(x) + 1.0e-16 + 1.0



""" Custom Loss Function. Can be used without weights.
Alternatively, the wrapper set below can be used WITH weights.

Not utilized in the provided notebook, but kept in case the user needs it.
"""
def like_loss(y_true, y_pred):
    m, s = tf.split(y_pred, 2, axis=-1)

    s = K.maximum(s, 1.e-6)
    
    square_top = K.square(m - y_true)
    square_bottom = K.square(s)
    div_result = Lambda(lambda x: x[0]/x[1])([square_top, square_bottom])

    loss = tf.add(K.log(s), tf.multiply(0.5, div_result) )

    return tf.reduce_mean(loss)



@keras_export('keras.losses.GaussLoss')
class GaussLoss(LossFunctionWrapper):
    """
    """
    def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='gauss_loss', **kwargs):
        super(GaussLoss, self).__init__(
            gauss_loss, name=name, reduction=reduction)


@keras_export('keras.losses.gauss_loss')
def gauss_loss(y_true, y_pred):
    """
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    m, s = tf.split(y_pred, 2, axis=-1)

    s = K.maximum(s, 1.e-6) # 1.e-6 is fine to prevent underflow. Smaller possible, but untested. 

    square_top = K.square(m - y_true)
    square_bottom = K.square(s)
    div_result = Lambda(lambda x: x[0]/x[1])([square_top, square_bottom])

    loss = 0.5*K.log(square_bottom) + 0.5*div_result


    return loss


class convnet_model():

    def __init__(self, input_shape, learning_rate = 0.0007,
                 num_dense_nodes = 128, num_dense_layers = 2,
                 num_models = 1, num_filters = 16):


        super(convnet_model, self).__init__()

        self.input_shape = input_shape
        self._model_type = 'cnn'
        self.lr = learning_rate
        self.initializer = 'he_normal'
        #self.initializer = 'random_uniform'
        self.activation = 'relu'
        self.num_dense_layers = num_dense_layers
        self.num_dense_nodes = num_dense_nodes
        self.num_models = num_models
        self.num_filters = num_filters
        
        self.beta_1 = 0.9  # exponential decay rate for the 1st moment estimates for optimization algorithm
        self.beta_2 = 0.999  # exponential decay rate for the 2nd moment estimates for optimization algorithm
        self.optimizer_epsilon = 1e-08  # a small constant for numerical stability for optimization algorithm
        self.clipnorm = 1.0
        self.optimizer = Adam(learning_rate=self.lr, beta_1=self.beta_1, beta_2=self.beta_2,
                              epsilon=self.optimizer_epsilon,
                              clipnorm=self.clipnorm)
        self.last_layer_activation = 'linear'

        self.loss_func = GaussLoss()


    def compile(self):

        input_tensor = Input(shape=self.input_shape, name='input')

        #checkpointer = ModelCheckpoint('keras_cnn_mag_model.h5', verbose=1)
        #self.callbacks = [checkpointer]
        self.callbacks = []

        self.models = {}
        for ii in range(self.num_models):
            model_ = Model(input_tensor, self.mag_model(input_tensor))

            model_.compile(optimizer=self.optimizer,
                            loss = self.loss_func,
                            metrics=['mean_absolute_error'])
            self.models[ii] = model_
        self.model_ = self.models[ii]

    def mag_model(self, x):

        cnn_layer_1 = Conv2D(filters=self.num_filters, kernel_size=(3, 3),
                             input_shape=self.input_shape, activation='relu', padding='valid')(x)
        #dropout_layer_1 = Dropout(dropout_rate)(cnn_layer_1)
        pool_layer_1 = MaxPool2D(pool_size=(2, 2), padding='same')(cnn_layer_1)

        #bnorm_1 = BatchNormalization()(pool_layer_1) # including this works relatively well

        cnn_layer_2 = Conv2D(filters=self.num_filters, kernel_size=(3, 3), activation='relu', padding='valid')(pool_layer_1)
        #dropout_layer_2 = Dropout(dropout_rate)(cnn_layer_2)
        pool_layer_2 = MaxPool2D(pool_size=(2, 2), padding='valid')(cnn_layer_2)

        #bnorm_2 = BatchNormalization()(pool_layer_2)

        cnn_layer_3 = Conv2D(filters=int(self.num_filters/2), kernel_size=(3, 3), activation='relu', padding='valid')(pool_layer_2)
        #dropout_layer_3 = Dropout(dropout_rate)(cnn_layer_3)
        pool_layer_3 = MaxPool2D(pool_size=(2, 2), padding='valid')(cnn_layer_3)

        #bnorm_3 = BatchNormalization()(pool_layer_3)

        flattener = Flatten()(pool_layer_3)


        #non-forked FC layers
        dense_layers = []
        fc_dropout_layers = []
        for ii in range(self.num_dense_layers):
            if ii==0:
                dense_layers.append(Dense(self.num_dense_nodes, activation=self.activation)(flattener))
            else:
                dense_layers.append(Dense(self.num_dense_nodes, activation=self.activation)(dense_layers[-1]))


        mu = Dense(1, name = 'mu')(dense_layers[-1])
        # I don't think this line is necessary sigma = Dense(1, activation = 'softplus', name='sigma')(dense_layers[-1])
        sigma = Dense(1, activation = elu_plus, name='sigma')(dense_layers[-1])

        output = Concatenate(name = 'outputs')([mu, sigma])

        return output


    def train_models(self, X_train, Y_train,
                     sample_weights = None,
                     train_epochs = 40,
                     batch_size=512,
                     useSampleWeights = True):
        self.classifiers = {}
        for ii in range(self.num_models):
            print(f'\nTraining model {ii+1} of {self.num_models}.')
            if useSampleWeights:
                self.classifiers[ii] = self.models[ii].fit(X_train, Y_train, sample_weight = sample_weights,
                                                           epochs=train_epochs,
                                                           batch_size=batch_size,
                                                           callbacks=self.callbacks)
            else:
                self.classifiers[ii] = self.models[ii].fit(X_train, Y_train,
                                                           epochs=train_epochs,
                                                           batch_size=batch_size,
                                                           callbacks=self.callbacks)

    def predict(self, X, merge_type='mean'):
        mags = []
        errs = []
        for ii in range(self.num_models):
            p = self.models[ii].predict(X, verbose=1)
            mags.append(p[:, 0])
            errs.append(p[:, 1])
        mags = np.array(mags)
        errs = np.array(errs)
        if merge_type =='mean':
            out = np.zeros((len(p),2), dtype=mags.dtype)
            out[:, 0] = np.mean(mags, axis=0)

            dm = mags - out[:, 0]

            dm2 = mags**2 - out[:, 0]**2

            out[:, 1] = np.mean(errs*errs + dm2, axis=0)
            return(out)
        else:
            return (mags, errs)

    def saveModel(self, Flux_std, mean, med, std, save_dir = 'ML_MAG_modelSaves'):
        for ii in range(self.num_models):
            dir_name = f'{save_dir}/model_{ii}'
            if os.path.isdir(dir_name):
                os.system(f'rm -r {dir_name}')
            os.system(f'mkdir -p {dir_name}')
            self.models[ii].save_weights(f'{dir_name}/ensemble_weights.h5')
            print(f'   saved to {dir_name}\n')
            with open(f'{save_dir}/median.properties','w+') as han:
                print(Flux_std, mean, med, std, file=han)

    def loadModel(self, save_dir = 'ML_MAG_modelSaves'):


        for ii in range(self.num_models):
            dir_name = f'{save_dir}/model_{ii}'
            print(f'\n   Loading model {dir_name}\n')
            self.models[ii].load_weights(f'{dir_name}/ensemble_weights.h5')

        with open(f'{save_dir}/median.properties') as han:
            data = han.readlines()
        s = data[0].split()
        Flux_std = float(s[0])
        mean = float(s[1])
        med = float(s[2])
        std = float(s[3])

        return (Flux_std, mean, med, std)
