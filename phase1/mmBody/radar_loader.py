import copy

import os
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as F
import glob
import pandas as pd
import cv2
import numpy as np
from pc2array import pc2array2d, radar_interpolation
from torchvision import transforms, utils
from PIL import Image
from skimage.util import random_noise
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from matplotlib import pyplot as plt


class MMPoseLoader(Dataset):
    def __init__(self,
                 root="../../../../data/mmpose/",
                 split='train',
                 modalities = ["Depth", "RadarImg5", "RGB"],
                 cache_size=10000,
                 normalized=True,
                 transforms_type = None,
                 depth_transforms = False,
                 test_scenario="train",
                 uniform=False,
                 npoint = 72*72):
        self.root = root
        self.npoints = npoint # set limit to the max number of loaded radar point
        self.uniform = uniform
        self.modalities = modalities
        if test_scenario == "all": 
            self.scenario = ["lab1", "lab2", "furnished", "rain", "smoke", "poor_lighting", "occlusion"]
        else: 
            self.scenario = [test_scenario]
        self.transforms = transforms_type
        self.normalized = normalized
        self.normalized_center = [None, None, None]
        self.depth_transform = depth_transforms
        assert (split == 'train' or split == 'test')

        # initialize dataframe
        self.df = pd.DataFrame()
        self.path_df = pd.DataFrame()

        # initialize all data path to be a dict
        self.datapath = {}

        # initialize cache
        self.cache_size = cache_size  # how many data points to cache in memory
        # self.cache = {}  # from index to (point_set, cls) tuple


        # Radar parameters:



        # Depth parameters:
        # 1. Inspect false depth data of dataset
        self.depth_false = list(range(0,0))
        
        self.depth_range = [None,None]
        self.depth_center = [None, None]
        

        # RGB parameters:

        # Train loader
        if split == "train":
            split_path = os.path.join(self.root, split)
            for sub_path in glob.glob(os.path.join(split_path, "*")):
                # If is depth false data, continue
                sequence = int(sub_path.split("/")[-1].split("_")[-1])
                if sequence in self.depth_false:
                    continue
                data_path_df = self._load_mesh(sub_path)
                for modality in self.modalities:
                    if modality == "RadarImg" or modality == "RadarP" or modality =="RadarImg5" or modality == "RadarP4":
                        if "Radar" in data_path_df:
                            continue
                        radar_path_df = self._load_radar(sub_path)
                        data_path_df = data_path_df.set_index(['Sequence', 'Frame']).join(
                            radar_path_df.set_index(['Sequence', 'Frame'])).reset_index(names=['Sequence', 'Frame'])
                    elif modality == "Depth" or modality == "DepthMotion" or modality == "DepthCrop":
                        if "Depth" in data_path_df:
                            continue
                        depth_path_df = self._load_depth(sub_path)
                        data_path_df = data_path_df.set_index(['Sequence', 'Frame']).join(
                            depth_path_df.set_index(['Sequence', 'Frame'])).reset_index(names=['Sequence', 'Frame'])
                    elif modality == "RGB":
                        rgb_path_df = self._load_rgb(sub_path)
                        data_path_df = data_path_df.set_index(['Sequence', 'Frame']).join(
                            rgb_path_df.set_index(['Sequence', 'Frame'])).reset_index(names=['Sequence', 'Frame'])
                    else:
                        print("error")


                self.path_df = pd.concat([self.path_df, data_path_df], ignore_index=True)


        # test loader
        else:
            for s in self.scenario:
                split_path = os.path.join(self.root, split, s)
                print(split_path)
                for sub_path in glob.glob(os.path.join(split_path, "*")):
                    data_path_df = self._load_mesh(sub_path)
                    for modality in self.modalities:
                        if modality == "RadarImg" or modality == "RadarP" or modality =="RadarImg5" or modality == "RadarP4":
                            if "Radar" in data_path_df:
                                continue
                            radar_path_df = self._load_radar(sub_path)
                            data_path_df = data_path_df.set_index(['Sequence', 'Frame']).join(
                                radar_path_df.set_index(['Sequence', 'Frame'])).reset_index(
                                names=['Sequence', 'Frame'])
                        elif modality == "Depth" or modality == "DepthMotion" or modality == "DepthCrop":
                            if "Depth" in data_path_df:
                                continue
                            depth_path_df = self._load_depth(sub_path)
                            data_path_df = data_path_df.set_index(['Sequence', 'Frame']).join(
                                depth_path_df.set_index(['Sequence', 'Frame'])).reset_index(
                                names=['Sequence', 'Frame'])
                        elif modality == "RGB":
                            rgb_path_df = self._load_rgb(sub_path)
                            data_path_df = data_path_df.set_index(['Sequence', 'Frame']).join(
                                rgb_path_df.set_index(['Sequence', 'Frame'])).reset_index(names=['Sequence', 'Frame'])
                        else:
                            print("error")
                    self.path_df = pd.concat([self.path_df, data_path_df], ignore_index=True)






    def __len__(self):
        return self.path_df.shape[0]

    def _get_item(self, index):
        

        if False:
            # Caching
            return self.cache[index]
        else:
            # No caching
            # Initialize experiment ground truth
            sequence = self.path_df.iloc[index]["Sequence"]
            frame = self.path_df.iloc[index]["Frame"]
            mesh_path = self.path_df.iloc[index]["Mesh"]




            # Initialize data of format:
            # ["Sequence", "Frame", "Pose", "Pose_hand", "Shape", "Vertices", "Joints", "modality 1", "modality 2"]
            data = {"Sequence": sequence, "Frame": frame}
            mesh_data = np.load(mesh_path)

            # Load mesh ground truth
            for k in mesh_data.keys():
                data[f"{k}"] = torch.from_numpy(mesh_data[k])
            
            angle = 0
            x_shift = 0
            y_shift = 0
            if self.transforms:
                angle = 0 
                # angle = np.random.randint(low=-15, high=15)
                x_shift = np.random.randint(low=-10, high=10)
                y_shift = np.random.randint(low=-10, high=10)
            
    
            if self.normalized:
                 self.normalized_center = data["joints"][0].numpy()
                 temp = np.subtract(data["joints"].numpy(), self.normalized_center)
                #  print("x min", temp[:, 0].min(), "max", temp[:, 0].max())
                #  print("y min", temp[:, 1].min(), "max", temp[:, 1].max())
                #  print("z min", temp[:, 2].min(), "max", temp[:, 2].max())


            for modality in self.modalities:
                if modality == "RadarImg":
                    radar_path = self.path_df.iloc[index]["Radar"]
                    radar_data = np.load(radar_path)
                    if self.normalized:
                        radar_data[:, :3] = np.subtract(radar_data[:, :3], self.normalized_center)
                        
                    radar_data = pc2array2d(radar_data, normalized=self.normalized, normalized_center=self.normalized_center)
                    radar_data = torch.from_numpy(radar_data).permute(2, 0, 1)  # convert to tensor
                    if self.transforms:
                        radar_data = F.affine(radar_data, angle, (x_shift, y_shift), 1, 0)
                    data["RadarImg"] = radar_data


                if modality == "RadarImg5":
                    radar_path = self.path_df.iloc[index]["Radar"]
                    radar_data = np.load(radar_path)

                    if self.normalized:
                        radar_data[:, :3] = np.subtract(radar_data[:, :3], self.normalized_center)
                    radar_data = pc2array2d(radar_data, normalized=self.normalized, normalized_center=self.normalized_center)
                    radar_data = torch.from_numpy(radar_data).permute(2, 0, 1).unsqueeze(1)  # convert to tensor
                    radar_data_5 = radar_data
                    for i in range(5-1):
                        path = radar_path.replace(str(frame), str(frame - i - 1))
                        if os.path.exists(path) == False: path = radar_path.replace("frame_"+str(frame), "frame_"+str(frame+i))
                        radar_data = np.load(path)

                        if self.normalized:
                            mesh_path_new = mesh_path.replace(str(frame), str(frame - 4 + i))
                            if os.path.exists(mesh_path_new) == False: mesh_path_new = mesh_path.replace("frame_"+str(frame), "frame_"+str(frame+i))
                            self.normalized_center = np.load(mesh_path_new)["joints"][0]
                            
                            radar_data[:, :3] = np.subtract(radar_data[:, :3], self.normalized_center)
                        
                        radar_data = pc2array2d(radar_data, normalized=self.normalized, normalized_center=self.normalized_center)
                        radar_data = torch.from_numpy(radar_data).permute(2, 0, 1).unsqueeze(1)  # convert to tensor
                        radar_data_5 = torch.concatenate([radar_data, radar_data_5], axis=1)
                    if self.transforms:
                        radar_data_5 = F.affine(radar_data_5, angle, (x_shift, y_shift), 1, 0)
                    data["RadarImg5"] = radar_data_5
                    # data["radar"] = radar_interpolation(radar_data)

                if modality == "RadarP":
                    radar_path = self.path_df.iloc[index]["Radar"]
                    radar_data = np.load(radar_path)
                    if self.normalized:
                        radar_data[:, :3] = np.subtract(radar_data[:, :3], self.normalized_center)
                    # convert to tensor
                    radar_data = torch.from_numpy(radar_interpolation(radar_data, npoint=self.npoints, normalized=self.normalized, normalized_center=self.normalized_center))
                    data["RadarP"] = radar_data


                if modality == "RadarP4":
                    radar_path = self.path_df.iloc[index]["Radar"]
                    radar_data = np.load(radar_path)
                    if self.normalized:
                        radar_data[:, :3] = np.subtract(radar_data[:, :3], self.normalized_center)
                    # convert to tensor
                    radar_data = torch.from_numpy(radar_interpolation(radar_data, npoint=self.npoints, normalized=self.normalized, normalized_center=self.normalized_center))
                    radar_data_4 = [radar_data]
                    for i in range(4-1):
                        path = radar_path.replace(str(frame), str(frame - 4 + i))
                        if os.path.exists(path) == False: path = radar_path.replace("frame_"+str(frame), "frame_"+str(frame+i))
                        radar_data = np.load(path)
                        if self.normalized:
                            mesh_path_new = mesh_path.replace(str(frame), str(frame - 4 + i))
                            if os.path.exists(mesh_path_new) == False: mesh_path_new = mesh_path.replace("frame_"+str(frame), "frame_"+str(frame+i))
                            self.normalized_center = np.load(mesh_path_new)["joints"][0]
                            
                            radar_data[:, :3] = np.subtract(radar_data[:, :3], self.normalized_center)
                        radar_data = torch.from_numpy(radar_interpolation(radar_data, npoint=self.npoints, normalized=self.normalized, normalized_center=self.normalized_center))
                        radar_data_4.append(radar_data)
                    data["RadarP4"] = torch.stack(radar_data_4, dim=0)

                if modality == "Depth":
                    depth_path = self.path_df.iloc[index]["Depth"]
                    depth_data = cv2.imread(depth_path)

                    depth_data = self.depth_preprocessing(depth_data)

                    


                            
                        

                    # data augmentation
                    if self.depth_transform:
                        # depth_data = Image.fromarray(depth_data)
                        depth_data = torch.from_numpy(depth_data)
                        depth_data = torch.tensor(random_noise(depth_data, mode='gaussian', mean=0, var=1.0, clip=True)) + depth_data
                        depth_data = depth_data.permute(2,0,1)
                    else:
                        depth_data = torch.from_numpy(depth_data).permute(2,0,1) # convert to tensor
                    depth_data = depth_data[0,:,:].view(1, depth_data.shape[1], depth_data.shape[2])

                    if self.transforms:
                        depth_data = F.affine(depth_data, angle, (x_shift, y_shift), 1, 0)

                    depth_data[depth_data == 0] = 20
                    depth_data = 20 - depth_data

                    # depth_data = transforms.GaussianBlur(11, sigma = 4)(depth_data)
                    



                    # if "RadarImg5" in data.keys():
                        
                    #     radar_mask = (data["RadarImg5"][[4],:,:] + data["RadarImg5"][[9],:,:] + data["RadarImg5"][[14],:,:]+data["RadarImg5"][[19],:,:]+data["RadarImg5"][[24],:,:]).permute(1,2,0)
                    #     radar_mask = radar_mask.numpy()
                    #     radar_mask = cv2.dilate(radar_mask, np.ones((4, 6), np.uint8), iterations=6)
                    #     radar_mask = np.where(radar_mask > 0, 1., 0.)
                    #     depth_mask = cv2.resize(radar_mask, interpolation = cv2.INTER_LINEAR, dsize=(256, 256))
                    #     depth_mask = torch.from_numpy(depth_mask).unsqueeze(0)
                    #     depth_data = depth_data + depth_mask *5
                





                    
                    data["Depth"] = depth_data


                    # generate human segmentation
                    seg_data = copy.deepcopy(depth_data)/10
                    # seg_data[seg_data != 0] = 1
                    data["Segmentation"] = seg_data

                    # generate attention mask
                    selected_joints = [0,4,5,7,8,21,20,19,18, 17, 16, 15] 
                    mask_size_list = [5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8]
                    mask_size = 5
                    x_range=[-1.47, 1.43]
                    x_res = 256
                    z_range=[-0.95, 1.95] 
                    z_res = 256
                    x_binlen = (x_range[1] - x_range[0])/x_res
                    z_binlen = (z_range[1] - z_range[0]) / z_res
                    mask = torch.zeros(x_res, z_res)
                    # print(data["joints"][:,0].min())
                    for i in range(12):
                        joint = data["joints"][selected_joints[i]]
                        mask_size = mask_size_list[i]
                        x = int(np.floor((joint[0] - x_range[0])/x_binlen))
                        z = z_res - int(np.floor((joint[2] - z_range[0]) / z_binlen))
                        mask[x-mask_size:x+mask_size, z-mask_size:z+mask_size] = 1
                        
                    # mask = mask.view(1, x_res * z_res)
                    # mask = nn.Softmax(dim=1)(mask) * 100
                    mask = mask.view(1, x_res, z_res).permute(0,2,1)
                    data["Joint_Mask"] = mask


                if modality == "DepthCrop":
                    if "Depth" in data.keys():
                        if int(sequence) < 10:
                            depth_crop = self.erasor(data["Depth"], h_range=[-50,20], w_range=[-10,10], size=7)
                        else:
                            depth_crop = self.erasor(data["Depth"],h_range=[-70,50], w_range=[-20,20], size=15)  
                        data["DepthCrop"] = depth_crop


                if modality == "DepthMotion":
                    depth_path = self.path_df.iloc[index]["Depth"]
                    depth_data = cv2.imread(depth_path)

                    depth_data_final = None
                    for i in range(4):
                        depth_path_current = depth_path.replace(str(frame ), str(frame + i))
                        depth_path_next = depth_path.replace(str(frame ), str(frame + i + 2))
                        if os.path.exists(depth_path_next) == False: depth_path_next = depth_path
                        if os.path.exists(depth_path_current) == False: depth_path_current = depth_path
                        depth_data_next = cv2.imread(depth_path_next)
                        depth_data_current = cv2.imread(depth_path_current)

                        depth_data_current = self.depth_preprocessing(depth_data_current)
                        depth_data_next = self.depth_preprocessing(depth_data_next)
                        depth_data = self.depth_preprocessing(depth_data)
                        
                        

                        depth_data_current = torch.from_numpy(depth_data_current).permute(2,0,1) # convert to tensor
                        depth_data_current = depth_data_current[0,:,:].view(1, depth_data_current.shape[1], depth_data_current.shape[2])
                        depth_data_next = torch.from_numpy(depth_data_next).permute(2,0,1) # convert to tensor
                        depth_data_next = depth_data_next[0,:,:].view(1, depth_data_next.shape[1], depth_data_next.shape[2])

                        if self.transforms:
                            depth_data_current = F.affine(depth_data_current, angle, (x_shift, y_shift), 1, 0)
                            depth_data_next = F.affine(depth_data_next, angle, (x_shift, y_shift), 1, 0)

                        if depth_data_final == None:
                            depth_data_final = depth_data_next.float() - depth_data_current.float()
                        else:
                            depth_data_final += depth_data_next.float() - depth_data_current.float()

                    
                    
                    
                    depth_data_final = torch.from_numpy(cv2.erode(depth_data_final.numpy(), np.ones((6, 3), np.uint8), iterations=4))
                    depth_data_final = torch.from_numpy(cv2.dilate(depth_data_final.numpy(), np.ones((8, 8), np.uint8), iterations=1))
                    # depth_data_final = torch.from_numpy(cv2.GaussianBlur(depth_data_final.numpy(), (3,3), cv2.BORDER_DEFAULT))
                    
                            

                    
                    data["DepthMotion"] = depth_data_final/4


                    
                    

                if modality == "RGB":

                    rgb_path = self.path_df.iloc[index]["RGB"]
                    rgb_data = cv2.imread(rgb_path)
                    scale_percent = 100  # percent of original size
                    width = int(rgb_data.shape[1] * scale_percent / 100)
                    height = int(rgb_data.shape[0] * scale_percent / 100)
                    # dim = (width, height)

                    
                    

                    # data augmentation
                    if self.transforms:
                        rgb_data = rgb_data[(int(height/2)-336):(int(height/2)+336), (int(width/2)-336):(int(width/2)+336), :]
                        rgb_data = Image.fromarray(rgb_data)
                        rgb_data = self.transforms(rgb_data)
                        rgb_data = (rgb_data * 255).int() #denormalized
                        
                    
                    else:
                        # ## for article ##, should close
                        # rgb_shift = [0, -0] # result: up, left
                        # ## for article ##, should close

                        # x = 124
                        # rgb_data = rgb_data[(int(height/2)+rgb_shift[0]-3*x):(int(height/2)+rgb_shift[0]+3*x), (int(width/2)+rgb_shift[1]-3*x):(int(width/2)+rgb_shift[1]+3*x), :]
                        # dim = (2*x, 2*x)
                        # # resize image
                        # rgb_data = cv2.resize(rgb_data, dim, interpolation=cv2.INTER_AREA)


                        rgb_data = rgb_data[(int(height/2)-336):(int(height/2)+336), (int(width/2)-336):(int(width/2)+336), :]
                        dim = (224, 224)
                        # resize image
                        rgb_data = cv2.resize(rgb_data, dim, interpolation=cv2.INTER_AREA)
                        rgb_data = torch.from_numpy(rgb_data).permute(2, 0, 1)  # convert to tensor
                    
                    
                    rgb_data = rgb_data.view(3, rgb_data.shape[1], rgb_data.shape[2])
                    
                    
                    data["RGB"] = rgb_data

                




            # data_list = [data["sequence"], data["frame"], data["joints"]]
            # for modality in self.modalities:
            #     data_list.append(data[modality])


            # if len(self.cache) < self.cache_size:
            #     self.cache[index] = data

            return data



    def __getitem__(self, index):
        return self._get_item(index)
    
    def depth_preprocessing(self, depth_data):
        scale = depth_data.shape[0]/512
                    
        self.depth_center = [depth_data.shape[0]//2, depth_data.shape[0]//2]
        self.depth_range = [5,17]
        width = int(scale*128)
        depth_center_coor = [0.01, -0.00]
        

        if depth_data.shape[1] == 512: 
            depth_center_coor = [-0.13, -0.45]

        x_start = int((depth_data.shape[1]-depth_data.shape[0])/2)
        depth_data = depth_data[:,x_start:x_start+depth_data.shape[0],:]
        
        
        
        
        

        scale1 = scale * 128  / 1.67  * (4.0/self.normalized_center[1])**1
        # scale1 = 250
        
        if self.normalized:
            self.depth_range = [(self.normalized_center[1] - 0.6)* 3.6, (self.normalized_center[1] + 0.6) * 3.6]
            self.depth_center[0] = depth_data.shape[0]//2 + int((self.normalized_center[0] - depth_center_coor[0]) * scale1)  # * (self.normalized_center[1]/3.5)) # hor
            self.depth_center[1] = depth_data.shape[1]//2 - int((self.normalized_center[2] - depth_center_coor[1]) * scale1) # * (self.normalized_center[1]/3.5))  # ver
            


        depth_data[depth_data < self.depth_range[0]] = 0
        depth_data[depth_data > self.depth_range[1]] = 0


        # # alignment debug
        # temp = np.subtract(data["joints"].numpy(), self.normalized_center)[[0,4,5,7,8,21,20,19,18, 17, 16, 15]]
        # for j in temp:
        #     x_coo = self.depth_center[0]+ int(scale1  * j[0] ) # * (self.normalized_center[1]/3.5)) # hor
        #     y_coo = self.depth_center[1] - int(scale1 * j[2]) # * (self.normalized_center[1]/3.5)) # ver
        #     depth_data[y_coo-5:y_coo+5,x_coo-5:x_coo+5, :] += 20

        

        


        
        depth_data = depth_data[self.depth_center[1]-width:self.depth_center[1]+width, self.depth_center[0]-width:self.depth_center[0]+width, :]
        depth_data = cv2.resize(depth_data, dsize=(256, 256)) # interpolation=cv2.INTER_CUBIC
        
        
        depth_data[210:,:,:] = 0
        depth_data[:45,:,:] = 0
        depth_data[:,225:,:] = 0
        depth_data[:,:25,:] = 0
        
        # depth_data = cv2.erode(depth_data, np.ones((3, 3), np.uint8), iterations=1)
        # depth_data = cv2.dilate(depth_data, np.ones((3, 3), np.uint8), iterations=3)
        depth_data = cv2.medianBlur(depth_data, 3)
        # depth_data = cv2.GaussianBlur(depth_data, (3,3), cv2.BORDER_DEFAULT)
        
        

        return depth_data


    def _from_depth_to_point(self, depth_img, R, t):
        """
            Description: Load the all depth data from .png to np.array(), concatenate all data
            into dataframe. Sort the dataframe according to the frame.

            Input:
            self: inherent from class
            depth_img: shape (512, 512, 3) / (1536, 2048, 3), with duplicated channels, type np.array()


            Output:
            Return a list of point cloud
        """


    def _load_rgb(self, current_path):
        """
        Description: Load the all rgb data from .png to np.array(), concatenate all data
        into dataframe. Sort the dataframe according to the frame.

        Input:
        self: inherent from class
        current_path: indicated the current path of the data to be loaded.
        (Example): ../data/mmpose/train/sequence_9

        Output:
        Return a Tuple of format (sequence, frame, rgb data)
        """

        rgb_path = []
        sequence = int(current_path.split("/")[-1].split("_")[-1])
        for file in glob.glob(os.path.join(current_path, "image", "master", "*.png")):
            frame = int(file.split("/")[-1].split(".")[0].split("_")[1])
            rgb_path.append([sequence, frame, file])

        rgb_path_df = pd.DataFrame(rgb_path, columns = ["Sequence", "Frame", "RGB"])
        rgb_path_df = rgb_path_df.sort_values("Frame", ignore_index=True)

        return rgb_path_df

    def _load_depth(self, current_path):
        """
        Description: Load the all depth data from .png to np.array(), concatenate all data
        into dataframe. Sort the dataframe according to the frame.

        Input:
        self: inherent from class
        current_path: indicated the current path of the data to be loaded.
        (Example): ../data/mmpose/train/sequence_9

        Output:
        Return a Tuple of format (sequence, frame, depth data)
        """

        depth_path = []
        sequence = int(current_path.split("/")[-1].split("_")[-1])
        for file in glob.glob(os.path.join(current_path, "depth", "master", "*.png")):
            frame = int(file.split("/")[-1].split(".")[0].split("_")[1])
            depth_path.append([sequence, frame, file])

        depth_path_df = pd.DataFrame(depth_path, columns = ["Sequence", "Frame", "Depth"])
        depth_path_df = depth_path_df.sort_values("Frame", ignore_index=True)

        return depth_path_df

    def _load_radar(self, current_path):
        """
        Description: Load the all radar data from .npy to np.array(), concatenate all data
        into dataframe. Sort the dataframe according to the frame.

        Input:
        self: inherent from class
        current_path: indicated the current path of the data to be loaded.
        (Example): ../data/mmpose/train/sequence_9

        Output:
        Return a Tuple of format (sequence, frame, radar data)
        """
        radar_path = []
        sequence = int(current_path.split("/")[-1].split("_")[-1])
        for file in glob.glob(os.path.join(current_path, "radar", "*.npy")):
            frame = int(file.split("/")[-1].split(".")[0].split("_")[1])
            radar_path.append([sequence, frame, file])
        radar_path_df = pd.DataFrame(radar_path, columns=["Sequence", "Frame", "Radar"])
        radar_path_df = radar_path_df.sort_values("Frame", ignore_index=True)

        return radar_path_df


    def _load_mesh(self, current_path):
        """
        Description: Load the all mesh data from .npy to np.array(), concatenate all data
        into dataframe. Sort the dataframe according to the frame.

        Input:
        self: inherent from class
        current_path: indicated the current path of the data to be loaded.
        (Example): ../data/mmpose/train/sequence_9

        Output:
        Return a Tuple of format (sequence, frame, mesh data)
        """
        mesh_path = []
        sequence = int(current_path.split("/")[-1].split("_")[-1])
        for file in glob.glob(os.path.join(current_path, "mesh", "*.npz")):
            frame = int(file.split("/")[-1].split(".")[0].split("_")[1])

            mesh_path.append([sequence, frame, file])
        mesh_path_df = pd.DataFrame(mesh_path,
                               columns=["Sequence", "Frame", "Mesh"])
        mesh_path_df = mesh_path_df.sort_values("Frame", ignore_index=True)

        return mesh_path_df
    
    def erasor(self, img, h_range = [-50, 20], w_range = [-10, 10], size = 7):
        img_copy = copy.deepcopy(img)
        h_mid = img.size()[1]//2 + np.random.randint(low=h_range[0], high=h_range[1])
        w_mid = img.size()[2]//2 + np.random.randint(low=w_range[0], high=w_range[1])

        img_copy[:, h_mid-size:h_mid+size, w_mid-size:w_mid+size] = 20
        return img_copy





global depth, radar_tensor

if __name__ == '__main__':
    
    test_scene = "lab1"
    idx = 50
    data = MMPoseLoader('../../data/mmpose/', split='test', uniform=False, normalized=True, test_scenario=test_scene, modalities=["RGB", "RadarP4", "Depth"], depth_transforms=False,
                        )

    data_select = data._get_item(idx)
    depth = data_select["Depth"]
    rgb = data_select["RGB"]
    radar = data_select["RadarP4"]

    radar_tensor = copy.deepcopy(radar)
    depth = depth.permute(1, 2, 0)
    rgb = rgb.permute(1, 2, 0)
    radar = radar[-1]
    print(radar.shape)
    


    plt.figure()
    plt.imshow(depth)
    plt.colorbar(label='Distance to Camera')
    # plt.title(f'Depth image')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.savefig(os.path.join("dataloader_vis", f"depth_article.png"))
    plt.close()

    plt.figure()
    plt.imshow(rgb)
    # plt.title(f'RGB image')
    plt.axis('off')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.savefig(os.path.join("dataloader_vis", f"rgb_article.png"), format="png", bbox_inches='tight', pad_inches=0)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(radar[:,0]+0.0, radar[:,2]-0.22, marker="o", s=1) # -0.22
    ax.set_xlim([-2,2])
    ax.set_ylim([-1,1])
    ext = [-2.0, 2.0, -1.0, 1.0]
    plt.imshow(depth, zorder=0, extent=ext, cmap='binary', vmin=4, vmax=13)
    


    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.scatter(radar[:,0]+0.1, radar[:,2]-0.2, marker="o", c="g", s=1)
    # ax.set_xlim([-2,2])
    # ax.set_ylim([-1,1])
    # ext = [-2.0, 2.0, -1.0, 1.0]
    # plt.imshow(rgb, zorder=0, extent=ext)

    aspect = depth.shape[0] / float(depth.shape[1]) * ((ext[1] - ext[0]) / (ext[3] - ext[2]))
    plt.gca().set_aspect(aspect)
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.axis('off')
    plt.savefig(os.path.join("dataloader_vis", f"radar_article.png"), format="png", bbox_inches='tight', pad_inches=0)
    plt.close()




    # from models import P4Transformer
    # CHECKPOINT_FILE = 'best_model_p4trans_full2.pth'
    # model =P4Transformer(radius=0.1, nsamples=32, spatial_stride=32,
    #               temporal_kernel_size=3, temporal_stride=2,
    #               emb_relu=False,
    #               dim=1024, depth=10, heads=8, dim_head=256,
    #               mlp_dim=2048, num_classes=17*3, dropout1=0.0, dropout2=0.0)

    # path = os.path.join("checkpoints", CHECKPOINT_FILE)
    # checkpoint = torch.load(path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # print('Use pretrain model')
    # model.cuda()
    # model.eval()

    # print("Model complete")

    # radar_input = radar_tensor.expand(1, -1, -1, -1).cuda().float()
    # print(radar_input.shape, type(radar_input))
    # model(radar_input, print_depth = depth, print_radar = radar)

    
    
    # DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=False)
    # for databatch in DataLoader:
    #     sequence = databatch["Sequence"]
    #     frame = databatch["Frame"]
    #     joints = databatch["joints"]
    #     depth = databatch["Depth"]
    #     radar = databatch["RadarImg5"]
    #     rgb = databatch["RGB"]

    #     print(sequence, frame, joints.shape, depth.shape, radar.shape, rgb.shape)
    #     depth = depth.permute(0, 2, 3, 1)
    #     radar = radar.permute(0, 3, 4, 1, 2)
    #     rgb = rgb.permute(0, 2, 3, 1)



    #     for i in range(12):
    #         # img = depth[i][100:400,100:400,0]
    #         img = depth[i]
    #         radar_img = (radar[i][:,:,1, 0] + radar[i][:,:,1, 1] + radar[i][:,:,1, 2] + radar[i][:,:,1, 3] + radar[i][:,:,1, 4])/5
    #         rgbi = rgb[i]

    #         plt.figure()
    #         plt.imshow(img)
    #         plt.colorbar(label='Distance to Camera')
    #         plt.title(f'Depth image {joints[i][0]}')
    #         plt.xlabel('X Pixel')
    #         plt.ylabel('Y Pixel')
    #         plt.savefig(os.path.join("dataloader_vis", f"depth{i}.png"))
    #         plt.close()

    #         plt.figure()
    #         plt.imshow(rgbi)
    #         plt.title(f'rgb image {joints[i][0]}')
    #         plt.xlabel('X Pixel')
    #         plt.ylabel('Y Pixel')

    #         plt.savefig(os.path.join("dataloader_vis", f"rgb{i}.png"))
    #         plt.close()

    #         plt.figure()
    #         plt.imshow(radar_img)
    #         plt.title('radar image')
    #         plt.colorbar(label='number')
    #         plt.xlabel('X Pixel')
    #         plt.ylabel('Y Pixel')

    #         plt.savefig(os.path.join("dataloader_vis", f"radar{i}.png"))
    #         plt.close()
    #     break



    # for i in range(20):
    #     img = np.array(io.v3.imread(os.path.join(f"../data/mmpose/train/sequence_{i}", "depth", "master", "frame_300.png")))[:,:,0]
    
    
    #     plt.figure()
    #     plt.imshow(img)
    #     plt.colorbar(label='Distance to Camera')
    #     plt.title('Depth image')
    #     plt.xlabel('X Pixel')
    #     plt.ylabel('Y Pixel')
    
    #     plt.savefig(os.path.join("depth_img", f"{i}test.png"))
