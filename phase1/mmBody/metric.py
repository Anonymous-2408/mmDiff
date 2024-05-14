import torch
import numpy as np
eps = 1e-8





def PCK_eval(pred, true, thr = 0.15, type = None, p=False):
    assert pred.shape == true.shape

    if type == None:
        thr = thr
    elif type == 0.5:
        thr = 0.1
    elif type == 0.2:
        thr = 0.1
    else:
        print("Not implemented metrics")
        return -1

    pred_np = pred.cpu().detach().numpy()
    true_np = true.cpu().detach().numpy()
    dists = calc_dists(pred_np, true_np)

    if p == True:
        print("pred:", pred_np[:,5:7,:])
        print("True", true_np[:,5:7,:])
        print(calc_dists(pred_np, true_np))
    acc = dist_acc(dists, thr)


    return acc

def p_mpjpe(predicted, target):
    predicted = predicted.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)

def MPJPE_mean(pred, true, type = None, p=False):
    assert pred.shape == true.shape


    pred_np = pred.cpu().detach().numpy()
    true_np = true.cpu().detach().numpy()
    dists = calc_dists(pred_np, true_np)

    if p == True:
        print("pred:", pred_np[:,5:7,:])
        print("True", true_np[:,5:7,:])
        print(calc_dists(pred_np, true_np))
    mean = dist_mean(dists)


    return mean

def MPJPE_max(pred, true, type = None, p=False):
    assert pred.shape == true.shape


    pred_np = pred.cpu().detach().numpy()
    true_np = true.cpu().detach().numpy()
    dists = calc_dists(pred_np, true_np)

    if p == True:
        print("pred:", pred_np[:,5:7,:])
        print("True", true_np[:,5:7,:])
        print(calc_dists(pred_np, true_np))
    j_max = dist_max(dists)


    return j_max



def calc_dists(preds, target):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[0], preds.shape[1])) # batch size and joints size
    for b in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            # if target[n, c, 0] > 1 and target[n, c, 1] > 1:
            #     normed_preds = preds[n, c, :] / normalize[n]
            #     normed_targets = target[n, c, :] / normalize[n]
            dists[b, j] = np.linalg.norm(preds[b,j,:] - target[b,j,:])
    return dists

def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)

    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1

def dist_max(dists):
    ''' Return max distance while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.max(dists[dist_cal])
    else:
        return -1

def dist_mean(dists):
    ''' Return mean distance while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.mean(dists[dist_cal])
    else:
        return -1








