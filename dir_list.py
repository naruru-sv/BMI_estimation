# Python program to explain os.listdir() method
import shutil
import os
import sys

import pandas as pd
import numpy as np

# Get the list of all files and directories
# in the root directory
# path = "/home/nata/Documents/Body-BMI_released"
# dir_list = os.listdir(path)
# print(len(dir_list))
#
# print("Files and directories in '", path, "' :")
#
# # print the list
# print(dir_list)
# for i in dir_list:
#     folder_path = path + "/"+ i
#     dir_dir_list = os.listdir(folder_path)
#     for k in dir_dir_list:
#         new_path = "/home/nata/Documents/dataset" + "/"+k
#         old_path = folder_path + "/"+ k
#         os.rename(old_path, new_path)

# path = "/home/nata/pythonProj/STRAPS/"
# dir_list = os.listdir(path)
# # print(dir_list)
# # f1 = open(path + "smpl_result.txt", "r")
# # f2 = open(path + "resnet_results.csv", "r")
# f3 = open(path + "results.txt", "r")
# f4 = open(path + "results_for_krr.txt", "w")
#
# # smpl_data = f1.readlines()
# # res_data = f2.readlines()
# for string in smpl_data:
#     smpl_jpg = string.split(",")
#     smpl_name = smpl_jpg[0]
#     for line in res_data:
#         # print(smpl_jpg[1:])
#         # print(line[:-1].split(","))
#         # sys.exit()
#         if line.startswith(smpl_name):
#             big_str = ",".join(smpl_jpg[1:])
#             big_big_str = line[:-1] + "," + big_str //10q8fn_0mII1_230_67_true_72_0.53.jpg
#             f3.write(big_big_str)
#             break
# results = f3.readlines()
# for line in results:
#     my_str = line.split(",")
#     name = my_str[0]
#     a = name.split("_")
#     height = float(a[3]) / 39.37
#     weight = float(a[2]) / 2.205
#     bmi = weight / (height ** 2)
#     my_str.insert(1, str(bmi))
#     new_str = ",".join(my_str)
#     f4.write(new_str)
# f3.close()
# f4.close()

def getSubstringBetweenTwoChars(ch1, ch2, s):
    return s[s.find(ch1) + 1:s.find(ch2)]


# s = 'Java2Blog'
# s2 = getSubstringBetweenTwoChars('J', 'g', s)
# print(s2)

# f1 = open("/home/nata/pythonProj/STRAPS/smpl_result.txt", "r")
# f2 = open("/home/nata/pythonProj/STRAPS/smpl_result_cleared.txt", "w")
# # i=2
# mess = f1.readlines()
# for i in range(2, 140):
#     for line in mess:
#         number = getSubstringBetweenTwoChars('e', '.', line)
#         if int(number) == i:
#             f2.write(line)
#             break

f1 = open("/home/nata/pythonProj/STRAPS/smpl_result_cleared.txt", "r")
f2 = open("/home/nata/pythonProj/STRAPS/resnet_melissa.csv", "r")
f3 = open("/home/nata/pythonProj/STRAPS/melissa_for_krr.txt", "w")

smpl = f1.readlines()
resnet = f2.readlines()
for line in smpl:
    smpl_jpg = line.split(",")
    smpl_name = smpl_jpg[0]
    for string in resnet:
        if string.startswith(smpl_name):
            big_str = ",".join(smpl_jpg[1:])
            big_big_str = string[:-1] + "," + big_str
            f3.write(big_big_str)
            break

# for string in smpl_data:
#     smpl_jpg = string.split(",")
#     smpl_name = smpl_jpg[0]
#     for line in res_data:
#         # print(smpl_jpg[1:])
#         # print(line[:-1].split(","))
#         # sys.exit()
#         if line.startswith(smpl_name):
#             big_str = ",".join(smpl_jpg[1:])
#             big_big_str = line[:-1] + "," + big_str //10q8fn_0mII1_230_67_true_72_0.53.jpg
#             f3.write(big_big_str)
#             break