import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import glob, os
import pandas as pd
import matplotlib.cm as cm
import matplotlib.animation as animation
import imageio
from matplotlib.animation import FuncAnimation
import time
import sys

def pc2array3d(data, x_range=[-4, 4], x_res = 64, y_range=[0, 5.3], y_res = 256, z_range=[-4, 4], z_res = 64):
    voxelized_array = np.zeros((x_res, y_res, z_res, 4))
    x_binlen = (x_range[1] - x_range[0])/x_res
    y_binlen = (y_range[1] - y_range[0]) / y_res
    z_binlen = (z_range[1] - z_range[0]) / z_res
    for point in data:
        x = int(np.floor((point[0] - x_range[0])/x_binlen))
        y = int(np.floor((point[1] - y_range[0]) / y_binlen))
        z = int(np.floor((point[2] - z_range[0]) / z_binlen))
        if (x >= 0 and y >= 0 and z >= 0) and (x < x_res and y < y_res and z < z_res):
            value = voxelized_array[x][y][z]

            if value[0] != 0:
                value[1] = (value[1] * value[0] + point[3])/(value[0] + 1)
                value[2] = (value[2] * value[0] + point[4]) / (value[0] + 1)
                value[3] = (value[3] * value[0] + point[5]) / (value[0] + 1)
                value[0] = value[0] + 1
            else:
                value[0] = 1
                value[1] = point[3]
                value[2] = point[4]
                value[3] = point[5]
    return voxelized_array

def pc2array2d(data, x_range=[-1.67, 1.67], x_res = 256, z_range=[-1.67, 1.67], z_res = 256, y_range = [0, 5.3], normalized = False, normalized_center = [-20, -20, -20]):
    if normalized:
        x_range = np.array([-1.67, 1.67]) * normalized_center[1]/3.6
        z_range = np.array([-1.67, 1.67]) * normalized_center[1]/3.6
        y_range = [ -1, 1]
        # print(x_range, y_range, z_range)

    voxelized_array = np.zeros((z_res, x_res, 5))
    x_binlen = (x_range[1] - x_range[0])/x_res
    z_binlen = (z_range[1] - z_range[0]) / z_res

    for point in data:
        if point[1] < y_range[0] or point[1] > y_range[1]:
            continue
        if normalized:
            point_depth = point[1] + normalized_center[1]

            
        
        
        x = int(np.floor((point[0] - x_range[0])/x_binlen))
        z = int(np.floor((z_range[1] - point[2]) / z_binlen))
        if (x >= 0 and z >= 0) and (x < x_res and z < z_res):
            value = voxelized_array[z][x]
            if value[0] != 0:
                value[1] = (value[1] * value[0] + point_depth)/(value[0] + 1)
                value[2] = (value[2] * value[0] + point[3]) / (value[0] + 1)
                value[3] = (value[3] * value[0] + point[4]) / (value[0] + 1)
                value[4] = (value[4] * value[0] + point[5]) / (value[0] + 1)
                value[0] = value[0] + 1
            else:
                value[0] = 1
                value[1] = point_depth
                value[2] = point[3]
                value[3] = point[4]
                value[4] = point[5]
    # normalization manual
    voxelized_array[:,:,4] = voxelized_array[:,:,4] / 100
    voxelized_array[:, :, 3] = voxelized_array[:, :, 3] * 10
    voxelized_array[:, :, 2] = voxelized_array[:, :, 2] *1e38

    

    return voxelized_array

def radar_interpolation(data, npoint=3000,x_range=[-1.5, 1.5], y_range = [0, 5.3], normalized = False, normalized_center = [-20, -20, -20]):
    if normalized:
        x_range = [-1.5, 1.5]
        y_range = [ -1., 1.]
        # print(x_range_normalized, y_range_normalized)
    data = data[np.where((data[:,1] <= y_range[1]) & (data[:,1] >= y_range[0]))]
    data = data[np.where((data[:, 0] <= x_range[1]) & (data[:, 0] >= x_range[0]))]
    

    length = data.shape[0]
    if length < npoint:
        for i in range(npoint-length):
            data = np.append(data, np.zeros(data.shape[1]).reshape(1,-1), axis=0)
    else:
        data = data[:npoint, :]
    data[:, 3] = data[:, 3] * 1e38
    data[:, 4] = data[:, 4] * 10
    data[:, 5] = data[:, 5] / 100


    return data


def img_np2tensor(npimage):
    '''
        Input: numpy image of shape n*m*3
        Output: return torch tensor of shape
    '''

    return torch.from_numpy(npimage).permute(2,0,1)



if __name__ == "__main__":
    names = []
    frame = []

    # read radar data
    radar_df = pd.DataFrame(columns=["Frame", "X", "Y", "Z", "Velocity", "Amplitude", "Energy_power"])

    os.chdir("./radar")
    for file in glob.glob("*.npy"):
        frame = int(file.split(".")[0].split("_")[1])
        data = np.load(file)
        print(data.shape)
        print(radar_interpolation(data).shape)
        # array = pc2array2d(data)
        # img = array[:,:,1]
        # plt.figure()
        # plt.imshow(img)
        # plt.show()
    print("radar done")




def fea2tensor2d(pos, data, x_range=[-1.67, 1.67], x_res = 64, z_range=[-1.67, 1.67], z_res = 64, y_range = [0, 5.3]):
    b, fea_n, fea_c = data.size()
    voxelized_array_all = torch.zeros((b, z_res, x_res, fea_c+2)).cuda() # to device
    x_binlen = (x_range[1] - x_range[0])/x_res
    z_binlen = (z_range[1] - z_range[0]) / z_res

    for batch in range(b):
        voxelized_array = voxelized_array_all[batch]
        for point_pos, point_fea in zip(pos, data):
            if point_pos[1] < y_range[0] or point_pos[1] > y_range[1]:
                continue
            x = int(np.floor((point_pos[0] - x_range[0])/x_binlen))
            z = int(np.floor((z_range[1] - point_pos[2]) / z_binlen))


            if (x >= 0 and z >= 0) and (x < x_res and z < z_res):
                    value = voxelized_array[z][x]
                    if value[0] != 0:
                        for i in range(fea_c):
                            value[i+2] = (value[i+2] * value[0] + point_fea[i]) / (value[0] + 1)
                        value[1] = (value[1] * value[0] + point_pos[1]) / (value[0] + 1)
                        value[0] = value[0] + 1
                    else:
                        value[0] = 1
                        value[1] = point_pos[1]
                        for i in range(fea_c):
                            value[i+2] =  point_fea[i]

    return voxelized_array_all



