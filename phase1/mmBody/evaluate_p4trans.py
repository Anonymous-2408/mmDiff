import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from matplotlib import pyplot as plt
import os
import copy
import numpy as np

from metric import  PCK_eval
from metric import MPJPE_max
from metric import MPJPE_mean, p_mpjpe



@torch.inference_mode()
def evaluate(net, dataloader,  epochs, device, amp, batch_size, joint_num = 55, test_flag = False, l1_flag = False, l2_flag = False, l2_lambda=0.001, l1_lambda = 0.001, save_flag = ''):
    net.eval()
    num_val_batches = len(dataloader)
    criterion = nn.MSELoss()
    criterion_likelihood = nn.GaussianNLLLoss()
    val_loss = 0
    # dice_score = 0
    num = 0
    fig_flat = False

    # evaluation metrics
    pck_acc = 0
    mean_pjpe = 0
    max_pjpe = 0
    pa_mpjpe = 0

    joint_gt_array_train = []
    joint_pred_array_train = []
    joint_var_array_train = []
    radar_feat_array_train = []
    radar_input_array_train = []
    action_array_train = []
    sequence_array_train = []
    radar_feature_all_array_train = []
    radar_position_array_train = []


    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            num += 1
            joint_selects =  [0,1, 4, 7, 2, 5, 8, 6, 12, 15, 24,  16, 18,  20, 17, 19, 21]
            # [0, 1, 2, 4, 5, 7, 8, 21, 20, 19, 18, 17, 16, 15, 9, 6]
            selected_weights = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1], device=device, dtype=torch.float32).view(12,1).repeat(1,3)
            joints = batch["joints"][:,joint_selects,:]
            radar = batch["RadarP4"]
            


            radar = radar.to(device=device, dtype=torch.float32)  # B, N, X
            joints = joints.to(device=device, dtype=torch.float32)


            # normalize joints with the center of body
            joints_original = copy.deepcopy(joints)
            joints_centers = joints_original[:, [0], :]

            joints = torch.sub(joints, joints_centers)



            # predict the mask
            joints_predict, joints_var_predict, joint_emb, radar_feature, radar_position = net(radar)

            #append array
            if save_flag[:4] == "test" or save_flag == "train":
                joint_gt_array_train.append(joints.cpu().detach().numpy())
                joint_pred_array_train.append(joints_predict.cpu().detach().numpy())
                radar_feat_array_train.append(joint_emb.cpu().detach().numpy())
                # joint_var_array_train.append(joints_var_predict.cpu().detach().numpy())
                radar_input_array_train.append(radar[:,0,:,:].cpu().detach().numpy())
                action_array_train.append(["Direction"] * 10)
                sequence_array_train.append(batch["Sequence"].cpu().detach().numpy())
                # radar_feature_all_array_train.append(radar_feature.cpu().detach().numpy())
                # radar_position_array_train.append(radar_position.cpu().detach().numpy())
            

            alpha = 0.0 # depth multitask loss
            loss = 100* criterion(joints_predict, joints.float())*1.0 
            # loss = 100* criterion_likelihood(target=joints_predict, input=joints.float(), var=joints_var_predict)*1.0 
            # loss = 100 * LogGaussianLikelihood(input=joints_predict, target=joints.float(), var=joints_var_predict)

                      
            

            joints_centers_predict = joints_centers
            joints_predict[:, 0, :] = torch.zeros(joints_predict[:, 0, :].size())
            # joints_predict = torch.add(joints_predict, joints_centers_predict)
            joints[:, 0, :] = torch.zeros(joints[:, 0, :].size())
            # joints = torch.add(joints, joints_centers)


            val_loss += loss.item()


            current_pck = PCK_eval(joints_predict, joints, thr=0.1)
            pck_acc += current_pck
            mean_pjpe += MPJPE_mean(joints_predict, joints)
            pa_mpjpe += np.mean(p_mpjpe(joints_predict, joints))
            if max_pjpe < MPJPE_max(joints_predict, joints):
                max_pjpe = MPJPE_max(joints_predict, joints)



            #save the fig
            if fig_flat == True:
                if test_flag == False:
                    save_prediction(joints_predict[0], joints_original[0], "train" + str(epochs),
                                    PCK_eval(joints_predict[[0],:,:], joints_original[[0],:,:], thr=0.1))
                    fig_flat = False
                else:
                    save_prediction(joints_predict[0], joints_original[0], "test" + str(num),
                                    PCK_eval(joints_predict[[0],:,:], joints_original[[0],:,:], thr=0.1))
                    if num == 100:
                        fig_flat = False

    if save_flag[:4] == "test":
        print("Saving Test.")
        
        # np transform and save to npz
        with open(os.path.join("data_saver17", f"joint_gt_array_{save_flag}.npy"), "wb") as f:
            nparray = np.concatenate(joint_gt_array_train)
            print(nparray.shape)
            np.save(f, nparray)
        # # np transform and save to npz
        # with open(os.path.join("data_saver17", "joint_var_array_test.npy"), "wb") as f:
        #     nparray = np.concatenate(joint_var_array_train)
        #     print(nparray.shape)
        #     np.save(f, nparray)
        
        # np transform and save to npz
        with open(os.path.join("data_saver17", f"joint_pred_array_{save_flag}.npy"), "wb") as f:
            nparray = np.concatenate(joint_pred_array_train)
            print(nparray.shape)
            np.save(f, nparray)
        
        # np transform and save to npz
        with open(os.path.join("data_saver17", f"radar_feat_array_{save_flag}.npy"), "wb") as f:
            nparray = np.concatenate(radar_feat_array_train)
            print(nparray.shape)
            np.save(f, nparray)
        # # np transform and save to npz
        # with open(os.path.join("data_saver17", "radar_feature_all_array_test.npy"), "wb") as f:
        #     nparray = np.concatenate(radar_feature_all_array_train)
        #     print(nparray.shape)
        #     np.save(f, nparray)
        # # np transform and save to npz
        # with open(os.path.join("data_saver17", "radar_position_array_test.npy"), "wb") as f:
        #     nparray = np.concatenate(radar_position_array_train)
        #     print(nparray.shape)
        #     np.save(f, nparray)
        # np transform and save to npz


        
        nparray = np.concatenate(radar_input_array_train)
        del radar_input_array_train
        bin = nparray.shape[0] // 1 + 1
        for i in range(1):
            with open(os.path.join("data_saver17", f"radar_input_array_{save_flag}.npy"), "wb") as f:
                end = min(nparray.shape[0], bin)
                print(nparray[:end].shape)
                np.save(f, nparray[:end])
                nparray = nparray[end:]
        del nparray
        
        # np transform and save to npz
        with open(os.path.join("data_saver17", f"action_array_{save_flag}.npy"), "wb") as f:
            nparray = np.concatenate(action_array_train)
            print(nparray.shape)
            np.save(f, nparray)
        
        # np transform and save to npz
        with open(os.path.join("data_saver17", f"sequence_array_{save_flag}.npy"), "wb") as f:
            nparray = np.concatenate(sequence_array_train)
            print(nparray.shape)
            np.save(f, nparray)
        
            
    elif save_flag == "train":
        print("Saving Train.")
        # np transform and save to npz
        with open(os.path.join("data_saver17", "joint_gt_array_train.npy"), "wb") as f:
            nparray = np.concatenate(joint_gt_array_train)
            print(nparray.shape)
            np.save(f, nparray)
        # # np transform and save to npz
        # with open(os.path.join("data_saver17", "joint_var_array_train.npy"), "wb") as f:
        #     nparray = np.concatenate(joint_var_array_train)
        #     print(nparray.shape)
        #     np.save(f, nparray)
        # np transform and save to npz
        with open(os.path.join("data_saver17", "joint_pred_array_train.npy"), "wb") as f:
            nparray = np.concatenate(joint_pred_array_train)
            print(nparray.shape)
            np.save(f, nparray)
        # np transform and save to npz
        with open(os.path.join("data_saver17", "radar_feat_array_train.npy"), "wb") as f:
            nparray = np.concatenate(radar_feat_array_train)
            print(nparray.shape)
            np.save(f, nparray)
        # # np transform and save to npz
        # with open(os.path.join("data_saver17", "radar_feature_all_array_train.npy"), "wb") as f:
        #     nparray = np.concatenate(radar_feature_all_array_train)
        #     print(nparray.shape)
        #     np.save(f, nparray)
        # # np transform and save to npz
        # with open(os.path.join("data_saver17", "radar_position_array_train.npy"), "wb") as f:
        #     nparray = np.concatenate(radar_position_array_train)
        #     print(nparray.shape)
        #     np.save(f, nparray)
        # np transform and save to npz
        nparray = np.concatenate(radar_input_array_train)
        del radar_input_array_train
        bin = nparray.shape[0] // 5 + 1
        for i in range(5):
            with open(os.path.join("data_saver17", f"radar_input_array_train{i}.npy"), "wb") as f:
                end = min(nparray.shape[0], bin)
                print(nparray[:end].shape)
                np.save(f, nparray[:end])
                nparray = nparray[end:]
        del nparray
        # np transform and save to npz
        with open(os.path.join("data_saver17", "action_array_train.npy"), "wb") as f:
            nparray = np.concatenate(action_array_train)
            print(nparray.shape)
            np.save(f, nparray)
        # np transform and save to npz
        with open(os.path.join("data_saver17", "sequence_array_train.npy"), "wb") as f:
            nparray = np.concatenate(sequence_array_train)
            print(nparray.shape)
            np.save(f, nparray)
    net.train()
    return val_loss / num,  pck_acc / num, mean_pjpe/num, max_pjpe, pa_mpjpe/num


def save_prediction(pred_tensor, true_tensor, epochs, pck):
    true_copy = copy.deepcopy(true_tensor)
    pred_copy = copy.deepcopy(pred_tensor)
    true = true_copy.detach().cpu().numpy()
    pred = pred_copy.detach().cpu().numpy()

    fig, axes = plt.subplots(nrows = 1, ncols = 4, sharex=False, sharey = False)
    fig.set_figwidth(20)
    axes[0].scatter(true[:, 0], true[:, 2],marker="o")
    axes[0].set_xlabel('Human Pose True [x, z]', labelpad=5)
    axes[1].scatter(pred[:, 0], pred[:, 2], marker="o")
    axes[1].set_xlabel('Human Pose Pred [x, z]', labelpad=5)
    axes[2].scatter(true[:, 1], true[:, 2], marker="o")
    axes[2].set_xlabel('Human Pose True [y, z]', labelpad=5)
    axes[3].scatter(pred[:, 1], pred[:, 2], marker="o")
    axes[3].set_xlabel('Human Pose Pred [y, z]', labelpad=5)

    lines = [[0, 9], [0, 10], [10, 11], [9, 11], [0, 1], [0, 2], [1, 3], [2, 4], [5, 7], [7, 9], [6, 8], [8, 10]]
    color = 0.0
    for i, j in lines:
        color += 0.9 / len(lines)
        axes[0].plot([true[i][0], true[j][0]], [true[i][2], true[j][2]],
                c=plt.cm.gist_ncar(color))
    color = 0.0
    for i, j in lines:
        color += 0.9 / len(lines)
        axes[1].plot([pred[i][0], pred[j][0]], [pred[i][2], pred[j][2]],
                     c=plt.cm.gist_ncar(color))
    color = 0.0
    for i, j in lines:
        color += 0.9 / len(lines)
        axes[2].plot([true[i][1], true[j][1]], [true[i][2], true[j][2]],
                     c=plt.cm.gist_ncar(color))
    color = 0.0
    for i, j in lines:
        color += 0.9 / len(lines)
        axes[3].plot([pred[i][1], pred[j][1]], [pred[i][2], pred[j][2]],
                     c=plt.cm.gist_ncar(color))

    axes[0].set_xlim([-0.6, 0.6])
    axes[0].set_ylim([-1, 1])
    axes[1].set_xlim([-0.6, 0.6])
    axes[1].set_ylim([-1, 1])
    axes[2].set_xlim([3, 4])
    axes[2].set_ylim([-1, 1])
    axes[3].set_xlim([3, 4])
    axes[3].set_ylim([-1, 1])
    plt.title(f"PCK0.1 = {pck}")

    plt.savefig(os.path.join("test_plot", f"prediction_{epochs}.png"))
    plt.close()
