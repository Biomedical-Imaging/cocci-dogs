import keras
import tensorflow as tf
import keras.backend as K

from keras.models import Model, load_model, Sequential
from keras import backend as K
from keras.utils import plot_model
from keras.utils import multi_gpu_model
from keras.layers import *
from keras.optimizers import *
from keras.backend import eval
from keras.applications.vgg16 import VGG16
from keras.applications import InceptionV3, ResNet50, MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *

models = {
    'vgg' : VGG16,
    'inception' : InceptionV3,
    'resnet' : ResNet50,
    'mobilenet' : MobileNetV2,
}
optimizers = {
    'rmsprop': RMSprop,
    'adam': Adam,
    'sgd': SGD
}

shallow_model = {
    'num_conv_layers' : 4,
    'num_conv_filters' : 64,
}
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


class Network:
    def __init__(self, args):
        self.args = args

        if args['model_type'] in ['vgg', 'inception', 'resnet', 'mobilenet']:
            self.load_pretrained()
        else:
            self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(GaussianNoise(self.args['noise'], input_shape=(400, 500, 3)))

        for i in range(shallow_model['num_conv_layers']):
            model.add(Conv2D(shallow_model['num_conv_filters'], (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(GlobalAveragePooling2D())
        model.add(Dropout(self.args['dropout']))

        for i in range(self.args['num_dense_layers']):
            model.add(Dense(self.args['num_dense_nodes'], activation='relu'))
            model.add(Dropout(self.args['dropout']))

        model.add(Dense(2, activation='softmax'))

        self.model = self._compile(model)

    def train(self, data, trial=None, client=None, batch_size=16, fold=1):
        x_train, x_test, y_train, y_test = data

        def schedule(epoch, lr):
            return lr * self.args['lr_decay']

        # KERAS CALLBACKS
        es = EarlyStopping(monitor='val_acc',patience=self.args['patience'])
        callbacks = [es, LearningRateScheduler(schedule)]

        if self.args['notsherpa']:
            ch = ModelCheckpoint('Models/%d/%05d.h5' % (self.args['sherpa_trial'], fold+1), monitor='val_acc', save_best_only=True)
            callbacks.append(ch)

        # if client is not None:
        #     callbacks.append(
        #         client.keras_send_metrics(
        #             trial,
        #             'val_acc',
        #             context_names=['loss', 'val_loss', 'acc', 'val_acc', 'auc', 'val_auc']
        #         )
        #     )

        if self.args['class_weight'] == 1:
            penalty = int(1 / (y_train.sum(axis=0)[1] / y_train.sum()))
            class_weight = {0:1, 1:penalty}
            print(y_train.sum(axis=0), y_test.sum(axis=0), class_weight)
        else:
            class_weight = {0:1, 1:1}

        if self.args['data_aug']:
            # TRAINING WITH DATA AUGMENTATION - GENERATOR
            train_gen = ImageDataGenerator(
                featurewise_center=self.args['normalize'],
                featurewise_std_normalization=self.args['normalize'],
                rotation_range=10,
                horizontal_flip=True,
                vertical_flip=False,
                zoom_range=0.1,
                width_shift_range=0.1,
                height_shift_range=0.1
            )
            if self.args['normalize']: train_gen.fit(x_train)
            generator = train_gen.flow(
                x_train,
                y_train,
                batch_size=8
            )
            test_generator = train_gen.flow(
                x_test,
                y_test,
                batch_size=8,
                shuffle=False
            )

            history = self.model.fit_generator(
                generator,
                steps_per_epoch=len(y_train) // batch_size,
                epochs=self.args['epochs'],
                validation_data=test_generator,
                validation_steps=len(y_test) // batch_size,
                verbose=2 if self.args['verbose'] else 0,
                callbacks=callbacks,
                class_weight=class_weight,
            )
        else:
            # TRAINING WITHOUT DATA AUGMENTATION
            history = self.model.fit(
                x_train, y_train,
                epochs=self.args['epochs'],
                validation_data=(x_test, y_test),
                batch_size=batch_size,
                verbose=2 if self.args['verbose'] else 0,
                callbacks=callbacks,
                class_weight=class_weight,
            )

        return history.history

    def load_pretrained(self):

        input_tensor = GaussianNoise(self.args['noise'])(Input(shape=(400,500,3)))
        weights = None if self.args['weights'] == 'none' else self.args['weights']
        base_model = models[self.args['model_type']](input_tensor=input_tensor, include_top=False, weights=weights)

        # i.e. freeze all convolutional layers
        for layer in base_model.layers:
            layer.trainable = self.args['train_base']

        x = base_model.output

        x = GlobalAveragePooling2D()(x)
        # x = Flatten()(x)
        x = Dropout(self.args['dropout'])(x)

        for i in range(self.args['num_dense_layers']):
            x = Dense(self.args['num_dense_nodes'], activation='relu')(x)
            x = Dropout(self.args['dropout'])(x)

        predictions = Dense(2, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        self.model = self._compile(model)

    def _compile(self, model):
        opt = optimizers[self.args['optimizer']](lr=self.args['lr'])

        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['acc', auc]
        )
        return model

    def load(self, file_name='model'):
        self.model = load_model(file_name + '.h5')

    def save(self, file_name='model'):
        self.model.save(file_name+'.h5')
