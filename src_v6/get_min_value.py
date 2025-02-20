import os
import numpy as np

root_folder = "./output_train"

f = open("statistics.csv", "w")
folders = os.listdir(root_folder)
for folder in folders:
    valid_file = f"{root_folder}/{folder}/Log/Log_Valid.csv"

    valid_inf = np.loadtxt(valid_file, delimiter=",", dtype="float", skiprows=1)
    valid_loss = valid_inf[:, 2].tolist()
    min_value = min(valid_loss)
    min_value_epoch = valid_loss.index(min_value)
    total_epoch = len(valid_loss)

    print(f"{folder:50s}:  {min_value:.6f} {min_value_epoch:5d}/{total_epoch:5d}")

    folder_split = folder.split("_")

    for i in folder_split:
        f.write("%s," % i)

    f.write(",%s,%s,%s\n" % (min_value, min_value_epoch, total_epoch))

f.close()

