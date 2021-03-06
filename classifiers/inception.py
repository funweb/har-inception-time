# resnet model
import os
import shutil
from glob import glob
import re

import keras
import numpy as np
import time

from tools.general import ModelCheckpoint_cus
from utils.utils import save_logs
from utils.utils import calculate_metrics
from utils.utils import save_test_duration


class Classifier_INCEPTION:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, batch_size=64,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500):

        self.output_directory = output_directory  # 结果输出的目录

        self.nb_filters = nb_filters  # ??
        self.use_residual = use_residual  # 是否使用残差
        self.use_bottleneck = use_bottleneck  #  是否使用 bottleneck
        self.depth = depth  # 网络深度
        self.kernel_size = kernel_size - 1  # ??
        self.callbacks = None  # ??
        self.batch_size = batch_size
        self.bottleneck_size = 32  # bottleneck 大小
        self.nb_epochs = nb_epochs  # epoch

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)  # 构建模型, 包括 inception 和 残差 层
            if (verbose == True):
                pass
                # self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:  # [-1] 是最后一个维度的 size
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]  # 核大小 [40, 20, 10]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):  # 遍历深度

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        # file_path = os.path.join(self.output_directory, 'best_model.hdf5')

        file_path = os.path.join(self.output_directory, "saved-model-{epoch:06d}-{loss:.6f}-{val_acc:.6f}.hdf5")

        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')

        # model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path,
        #                                                    monitor='loss', verbose=1,
        #                                                    save_best_only=False,
        #                                                    mode='min', period=int(self.nb_epochs/5),
        #                                                    )

        model_checkpoint = ModelCheckpoint_cus(filepath=file_path,
                                               monitor='loss', verbose=1,
                                               save_best_only=False,
                                               save_best_only_period=True,
                                               mode='min', period=100,  # int(self.nb_epochs/5)
                                               )


        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true, plot_test_acc=True):
        if len(keras.backend.tensorflow_backend._get_available_gpus()) == 0:
            print('notice: These is no gpu')
            time.sleep(30)
            # exit()

        # x_val and y_val are only used to monitor the test loss and NOT for training
        # x_val 和 y_val 仅用于监测测试损失，不用于训练

        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size

        start_time = time.time()

        if plot_test_acc:

            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        else:

            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=self.verbose, callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        ### -----------  认为 最小loss 是最好的
        min_loss = 128
        best_model_name = ""
        pattern = r'[Pbest_]?saved-model-(\d+)-(\d*\.\d*)-(\d*\.\d*).hdf5'

        prog = re.compile(pattern)
        for hdf5_name in glob("{}/saved*hdf5".format(self.output_directory)):
            matchObj = prog.match(os.path.basename(hdf5_name))
            c_epoch = matchObj.group(1)
            c_loss = matchObj.group(2)
            c_acc = matchObj.group(3)
            if float(c_loss) < min_loss:  # 这里出现了BUG,,, 比较之后忘记将最小值赋值给当前值...
                min_loss = float(c_loss)
                best_model_name = hdf5_name
        if best_model_name == "":
            best_model_name = "last_model.hdf5"
        shutil.copy(os.path.join(self.output_directory, os.path.basename(best_model_name)), os.path.join(self.output_directory, "best_model.hdf5"))
        print("{} --> {}".format(os.path.join(self.output_directory, os.path.basename(best_model_name)), os.path.join(self.output_directory, "best_model.hdf5")))
        ### -----------  认为 最小loss 是最好的

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val, return_df_metrics=False)

        # save predictions  保存预测
        np.save(self.output_directory + 'y_pred.npy', y_pred)
        np.savetxt(self.output_directory + 'y_pred.txt', y_pred)

        # convert the predicted from binary to integer  将预测值从二进制转换为整数
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration, plot_test_acc=plot_test_acc)


        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        model_path = os.path.join(self.output_directory, 'best_model.hdf5')
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred
