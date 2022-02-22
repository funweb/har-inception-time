import os
import shutil
import time

op_dir = os.path.join("..")

source_file_name = "103_kyoto7_inception_32_64_1000_1.ipynb"
target_file_name = "{tag}_{dataname}_{method}_{data_length}_{batch_size}_{epochs}_1.ipynb"

tag = 150
dataname = "kyoto11"
method = "inception"
data_length = 1024
batch_size = 64
epochs = 1000
kernel_wide_base = 1
kernel_number_base = 1
net_deep_base = 1

copy_num = 5

dataname_list = ["cairo", "milan", "kyoto7", "kyoto8", "kyoto11"]

source_file = os.path.join(op_dir, source_file_name)
target_file_name_list = []
for i in range(copy_num):
    tag += 1
    dataname = dataname_list[i]

    target_file = os.path.join(op_dir, target_file_name.format(tag=tag, dataname=dataname, method=method,
                                                               data_length=data_length, batch_size=batch_size,
                                                               epochs=epochs))
    assert not os.path.exists(target_file), "文件已存在, 请查验"

    print("{}->\n{}\n\n".format(source_file, target_file))
    target_file_name_list.append(target_file)
    time.sleep(3)
    shutil.copy(source_file, target_file)

print(target_file_name_list)
