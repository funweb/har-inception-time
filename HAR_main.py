import json
import os

import emoji

from tools import general
from tools.configure.constants import DATASETS_CONSTANT, INCEPTION_CONSTANT, METHOD_PARAMETER_TEMPLATE
# from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES
import pandas as pd
# from utils.utils import read_all_datasets
from utils.utils import transform_labels
from utils.utils import create_directory
# from utils.utils import run_length_xps
# from utils.utils import generate_results_csv  # 这个是生成结果文件, 看来后期需要探索一下.

import utils
import numpy as np
import sys
import sklearn
import tensorflow as tf


# Load data in the following format:
# cutdatadir + '\\' + data_name + '-test-y-' + str(k) + '.npy', datas_y[test]
def load_data(data_name, cutdatadir, data_lenght=2000, k=2):
    """
    功能: 载入数据集, 可以按照约定的格式返回, 包括: 训练数据集, 测试数据集, 类别, 编码等;  # TODO: 类别编码问题估计还需要进一步调整
    可调整的包括: 控制数据的长度
    单变量: 是否将单变量转换为适用于 卷积 的格式
    """

    assert 2001 > data_lenght > 0, "Please check data_lenght: {} ".format(data_lenght)  # 长度要适合

    data_type = 'train'
    data_x_path = os.path.join(cutdatadir, data_name + '-' + data_type + '-x-' + str(k) + '.npy')

    print(emoji.emojize("\n:trade_mark: data_x_path: {}\n\n".format(data_x_path)))

    x_train = np.load(data_x_path, allow_pickle=True)

    data_y_path = os.path.join(cutdatadir, data_name + '-' + data_type + '-y-' + str(k) + '.npy')
    y_train = np.load(data_y_path, allow_pickle=True)

    data_labels_path = os.path.join(cutdatadir, data_name + '-labels.npy')
    dictActivities_x = np.load(data_labels_path, allow_pickle=True).item()

    data_type = 'test'
    data_x_path = os.path.join(cutdatadir, data_name + '-' + data_type + '-x-' + str(k) + '.npy')
    x_test = np.load(data_x_path, allow_pickle=True)

    data_y_path = os.path.join(cutdatadir, data_name + '-' + data_type + '-y-' + str(k) + '.npy')
    y_test = np.load(data_y_path, allow_pickle=True)

    data_labels_path = os.path.join(cutdatadir, data_name + '-labels.npy')
    dictActivities_y = np.load(data_labels_path, allow_pickle=True).item()

    # x_range = len(np.unique(np.concatenate((x_train, x_test), axis=0)))  # 一共有多少种状态
    # x_train = x_train / (x_range + 1)  # 归一化处理  # TODO: 感觉没必要
    x_train = x_train[:, -data_lenght:]  # 控制数据长度
    # x_test = x_test / (x_range + 1)
    x_test = x_test[:, -data_lenght:]

    # ---
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))  # 类别数据, 也就是 len(dictActivities_y)

    # make the min to zero of labels  将标签的最小值设为零
    y_train, y_test = transform_labels(y_train, y_test)

    # save orignal y because later we will use binary  保存原始y，因为稍后我们将使用二进制
    y_true = y_test.astype(np.int64)
    y_true_train = y_train.astype(np.int64)
    # transform the labels from integers to one hot vectors  将标签从整数转换为 one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    if len(x_train.shape) == 2:  # if univariate, 如果是单变量
        # add a dimension to make it multivariate with one dimension  添加一个维度，使其具有一个维度的多变量
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # return x_train, y_train, x_train, y_train, y_true_train, nb_classes, y_true_train, enc
    return x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc
    # ---


def fit_classifier(x_train, y_train, x_test, y_test, y_true, classifier_name, nb_classes, output_directory):
    input_shape = x_train.shape[1:]

    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True)  # 这个 verbose 是总的

    classifier.fit(x_train, y_train, x_test, y_test, y_true)  # plot_test_acc 决定了是否在训练的时候查看验证效果

    return classifier


def create_classifier(classifier_name, input_shape, nb_classes, output_directory,
                      verbose=False, build=True):
    if classifier_name == 'nne':
        from classifiers import nne
        return nne.Classifier_NNE(output_directory, input_shape, nb_classes, verbose)

    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose, build=build,
                                              batch_size=METHOD_PARAMETER_TEMPLATE["batch_size"],
                                              nb_filters=METHOD_PARAMETER_TEMPLATE["nb_filters"],
                                              use_residual=METHOD_PARAMETER_TEMPLATE["use_residual"],
                                              use_bottleneck=METHOD_PARAMETER_TEMPLATE["use_bottleneck"],
                                              depth=METHOD_PARAMETER_TEMPLATE["depth"],
                                              kernel_size=METHOD_PARAMETER_TEMPLATE["kernel_size"],
                                              nb_epochs=METHOD_PARAMETER_TEMPLATE["nb_epochs"])


# def train_val(dataset_name, distance_int, archive_name="casas", nb_iter_=5, length_limit=2000):
def train_val(dict_config_cus):

    general.Merge(METHOD_PARAMETER_TEMPLATE, dict_config_cus)

    if METHOD_PARAMETER_TEMPLATE['calculation_unit'] is '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        if METHOD_PARAMETER_TEMPLATE['calculation_unit'] is '1':
            # 使用第0块GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 显示信息的等级  3：只显示error


    datadir = os.path.join(METHOD_PARAMETER_TEMPLATE["datasets_dir"], 'ende',
                           METHOD_PARAMETER_TEMPLATE["dataset_name"],
                           str(METHOD_PARAMETER_TEMPLATE["distance_int"]),
                           'npy')

    df_metrics = pd.DataFrame()

    for k in range(METHOD_PARAMETER_TEMPLATE['ksplit']):
        # run nb_iter_ iterations of Inception on the whole TSC archive
        # 在整个 TSC 存档上运行 Inception 的 nb_iter_ 迭代

        for iter in range(METHOD_PARAMETER_TEMPLATE["nb_iter_"]):
            print(emoji.emojize(":repeat_single_button: iter: {}".format(iter)))

            trr = '_itr_' + str(iter)

            classifier_name = 'inception'  # METHOD_PARAMETER_TEMPLATE["model_name"]

            tmp_output_directory = os.path.join(METHOD_PARAMETER_TEMPLATE["datasets_dir"], "results",
                                                classifier_name,
                                                METHOD_PARAMETER_TEMPLATE["archive_name"],
                                                METHOD_PARAMETER_TEMPLATE["dataset_name"],
                                                str(METHOD_PARAMETER_TEMPLATE["distance_int"]),
                                                str(k),
                                                str(trr))  # 临时结果文件夹

            print(emoji.emojize(":white_exclamation_mark: dataset_name: {}").format(METHOD_PARAMETER_TEMPLATE["dataset_name"]))

            # ---------------------------------
            # 这里的超参数还需要调整
            cutdatadir = os.path.join(datadir, str(METHOD_PARAMETER_TEMPLATE["ksplit"]))

            x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = load_data(METHOD_PARAMETER_TEMPLATE["dataset_name"],
                                                                                                cutdatadir,
                                                                                                METHOD_PARAMETER_TEMPLATE["data_lenght"],
                                                                                                k)
            # ---------------------------------

            output_directory = os.path.join(tmp_output_directory)
            output_directory += "/"

            if METHOD_PARAMETER_TEMPLATE["reTrain"]:
                general.reTrain(output_directory)

            create_directory(output_directory)

            complete_flag_file = os.path.join(output_directory, METHOD_PARAMETER_TEMPLATE["complete_flag"])
            if os.path.exists(complete_flag_file):  # 说明存在文件夹  #TODO: 能够说明已经运行完毕了, 不错.
                print('Already_done', tmp_output_directory, METHOD_PARAMETER_TEMPLATE["dataset_name"])
                continue

            fit_classifier(x_train, y_train, x_test, y_test, y_true, classifier_name, nb_classes, output_directory)

            # the creation of this directory means
            # create_directory(output_directory + '/DONE')
            with open(complete_flag_file, "w", encoding="utf-8") as fw:  # 这个文件的内容应该是运行程序所有参数的设置...因此, 是否超参数都应该设置成为字典呢?
                json.dump(METHOD_PARAMETER_TEMPLATE, fw)

        # run the ensembling of these iterations of Inception  运行这些 Inception 迭代的集成
        classifier_name = 'nne'
        print("nne_________________")

        # datasets_dict = read_all_datasets(root_dir, archive_name)

        # tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'

        tmp_output_directory = os.path.join(METHOD_PARAMETER_TEMPLATE["datasets_dir"],
                                            "results",
                                            classifier_name,
                                            METHOD_PARAMETER_TEMPLATE["archive_name"])  # 临时结果文件夹

        print('\t\t\tdataset_name: ', METHOD_PARAMETER_TEMPLATE["dataset_name"])

        # x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data()
        x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = load_data(
            METHOD_PARAMETER_TEMPLATE["dataset_name"],
            cutdatadir,
            METHOD_PARAMETER_TEMPLATE["data_lenght"],
            k)

        output_directory = os.path.join(tmp_output_directory,
                                        METHOD_PARAMETER_TEMPLATE["dataset_name"],
                                        str(METHOD_PARAMETER_TEMPLATE["distance_int"]),
                                        str(k)
                                        )
        output_directory += "/"

        classifier_class = fit_classifier(x_train, y_train, x_test, y_test, y_true, classifier_name, nb_classes,
                                          output_directory)

        df_metrics_temp = pd.read_csv(os.path.join(classifier_class.output_directory, "df_metrics.csv"))

        df_metrics = pd.concat([df_metrics, df_metrics_temp])

        print('\t\t\t\tDONE')

    # 计算k折交叉的均值和方差

    s = ""
    for n in df_metrics.columns:
        if len(n.split(":")) == 1:  # 保证是想要的字段吗, 因为 df 转过来之后会有不知道啥的冗余第一列(index)
            n_mean = np.mean(df_metrics[n])
            n_std = np.std(df_metrics[n])
            s += 'current database: {}, distance_int: {}, metric: {} \t {:.2f}% (+/- {:.2f}%)\n'.format(
                METHOD_PARAMETER_TEMPLATE["dataset_name"],
                METHOD_PARAMETER_TEMPLATE["distance_int"], n, n_mean, n_std)

    print(s)
    with open(os.path.join(os.path.join(classifier_class.output_directory, ".."), "ksplit_ave.csv"), 'w', encoding="utf-8") as fw:
        fw.writelines(s)

    ############################################### main

if __name__ == '__main__':
    # 数据参数
    dataset_name = "cairo"
    data_lenght = 300
    distance_int = 999

    # 模型公共参数
    nb_epochs = 15
    batch_size = 64

    calculation_unit = "1"

    # 模型私有参数
    model_parameter_dict = INCEPTION_CONSTANT

    dict_config_cus = {
        "datasets_dir": DATASETS_CONSTANT["base_dir"],  # 这是公共数据集常量
        "archive_name": DATASETS_CONSTANT["archive_name"],
        "ksplit": DATASETS_CONSTANT["ksplit"],
        "dataset_name": dataset_name,
        "data_lenght": data_lenght,
        "distance_int": distance_int,

        "nb_epochs": nb_epochs,
        "batch_size": batch_size,

        "reTrain": True,
        "calculation_unit": calculation_unit,
    }

    general.Merge(dict_config_cus, model_parameter_dict)

    train_val(dict_config_cus)
    print("\n\nsuccess all...")
