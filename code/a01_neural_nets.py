# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *
import warnings
from keras.callbacks import Callback
from keras.metrics import top_k_categorical_accuracy


class ModelCheckpoint_MAP5(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='max', period=1, patience=None, validation_data=()):
        super(ModelCheckpoint_MAP5, self).__init__()
        self.interval = period
        self.valid, self.target = validation_data
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        if mode is 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            self.monitor_op = np.less
            self.best = np.Inf

        # part for early stopping
        self.epochs_from_best_model = 0
        self.patience = patience

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            # Part with score calculation
            start_time = time.time()
            probs = self.model.predict(self.valid, verbose=0)
            preds = probs.argsort(axis=1)[:, -5:][:, ::-1]
            score = mapk(self.target, preds, k=5)
            print("MAP5 score: {:.6f} Epoch: {} Time: {:.2f} sec".format(score, epoch + 1, time.time() - start_time))

            filepath = self.filepath.format(epoch=epoch + 1, score=score, **logs)
            if self.monitor_op(score, self.best):
                self.epochs_from_best_model = 0
            else:
                self.epochs_from_best_model += 1

            if self.save_best_only:
                current = score
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

            if self.patience is not None:
                if self.epochs_from_best_model > self.patience:
                    print('Early stopping: {}'.format(self.epochs_from_best_model))
                    self.model.stop_training = True


def get_model_densenet121(in_shape=(None, None, 3)):
    from keras.models import Model
    from keras.applications.densenet import DenseNet121
    from keras.layers import Dense, Input, concatenate, Dropout

    # base model
    model = DenseNet121(input_shape=in_shape, weights='imagenet', include_top=False, pooling='avg')
    x = model.layers[-1].output
    # x = Dropout(0.1)(x)
    x = Dense(UNIQUE_WHALES, activation='softmax', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model


def get_model_inception_resnet_v2_binary(in_shape=(None, None, 3)):
    from keras.models import Model
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, concatenate, Dropout

    # base model
    model = InceptionResNetV2(input_shape=in_shape, weights='imagenet', include_top=False, pooling='avg')
    x = model.layers[-2].output
    y1 = GlobalAveragePooling2D()(x)
    y2 = GlobalMaxPooling2D()(x)
    x = concatenate([y1, y2])
    # x = Dropout(0.1)(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model


def freeze_unfreeze_model(model, freeze=True):
    for layer in model.layers:
        if layer.name == 'predictions':
            layer.trainable = True
            continue
        if freeze is True:
            layer.trainable = False
        else:
            layer.trainable = True
    return model


def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    print(K.floatx())
    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes