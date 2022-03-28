# imports
import sys
sys.path.insert(1, 'C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites')
from imagequalitymetrics import ImageQualityMetrics
import SIREN
import numpy as np
import imageio as io
import os
from numpy.random import default_rng
import normalize
import tifffile as tif
import tensorflow as tf


data_path = 'D:/jo77pihe/Registered/20220203_AutoQuant_Averaged'
res_path = 'D:/jo77pihe/Registered/20220209_SIRENHypTun_z5'
res_path_int_p = 'D:/jo77pihe/Registered/20220224_SIRENInterplane'
res_path_mot_c='D:/jo77pihe/Registered/20220224_SIRENMotCorr'
rep_path='D:/jo77pihe/Registered/20220227_Repetitions_deconved'


# Randomly select 15 images
f = [f for f in os.listdir(data_path) if f.endswith('.tif')]
rng = default_rng(1234)
idx = rng.choice(len(f), size=15, replace=False)
idx_preproc = idx[0:5]
idx_train = idx[5:]
f_train = [f[i] for i in idx_train]
f_preproc = [f[i] for i in idx_preproc]


def hyp_tuning():
    n_layers = [1, 2]
    hidden_dims = [256]
    steps = 1500
    step_to_plot = 1500
    met = ImageQualityMetrics()

    for (idx, img) in enumerate(f_preproc[0:3]):
        stack = np.asarray(io.mimread(os.path.join(data_path, img)), dtype=np.float32)[0:5,:,:]
        res_psnr = np.zeros((len(n_layers), len(hidden_dims)))
        res_ssim = np.zeros((len(n_layers), len(hidden_dims)))
        labs = []
        for (nx, nl) in enumerate(n_layers):
            for (hx, hd) in enumerate(hidden_dims):
                    folder ='_'.join(('Trial', str(img), str(nl), str(hd)))
                    #if not (folder in os.listdir(res_path)) and not (nl==3 and (hd==128 or 256)):
                    args = {}
                    args['hidden_layers'] = nl
                    args['hidden_features'] = hd
                    plane_int = SIREN.application.InterplanePrediction(args)
                    X, _ = plane_int.preprocess(stack, 1)

                    plane_int.train(X, n_steps=steps, steps_to_plot=step_to_plot, batch_size=None,
                                           result_path=os.path.join(res_path, folder))
                    prediction = plane_int.test(X)
                    tif.imwrite(os.path.join(res_path, str(nl) + '_' + str(hd) + '_' + img), prediction)

                    # Evaluation
                    stack -= stack.min()  # * 2 - 1
                    stack = stack / stack.max()

                    res_psnr[nx, hx] = met.psnr(prediction, stack)
                    win_size= stack.shape[0] if stack.shape[0]<7 else None

                    res_ssim[nx, hx] = met.ssim(prediction, stack,win_size=win_size)
                    np.save(os.path.join(res_path, folder,'res_psnr_' + img + '.npy'), res_psnr)
                    np.save(os.path.join(res_path, folder,'res_ssim_' + img + '.npy'), res_ssim)
                    labs.append('Layers: ' + str(nl) + ', Hidden_dim: ' + str(hd))
                    print(str(nl), str(hd))


def eval_interplane_prediction():
    args = {}
    args['hidden_layers'] = 2
    args['hidden_features'] = 128
    steps = 1500
    step_to_plot=1500
    files = f_train[0:5]
    for f in files:
        for n in range(2,9):
            if not((str(n) + '_' + f) in os.listdir(res_path_int_p)):
                stack = np.asarray(io.mimread(os.path.join(data_path, f)), dtype=np.float32)[0:20, :, :]

                plane_int = SIREN.application.InterplanePrediction(args)
                X, Y = plane_int.preprocess(stack, n)

                plane_int.train(X, n_steps=steps, steps_to_plot=step_to_plot, batch_size=None,
                            result_path=res_path_int_p)
                prediction = plane_int.test(Y)
                tif.imwrite(os.path.join(res_path_int_p, str(n) + '_' + f), prediction)

def eval_mot_corr():
    args = {}
    args['hidden_layers'] = 3
    args['hidden_features'] = 64
    steps = 1500
    step_to_plot=1500
    files = [f[:-5] for f in os.listdir(rep_path) if f.endswith('.tif')]
    files_set = set(files)
    files=[f+'.tif' for f in files_set]
    for f in files:
        if not (f in os.listdir(res_path_mot_c)):
            plane_int = SIREN.application.Motion_Correction(args)
            X, _ = plane_int.preprocess(rep_path, f)

            plane_int.train(X, n_steps=steps, steps_to_plot=step_to_plot, batch_size=None,
                        result_path=res_path_mot_c)
            prediction = plane_int.test(X)
            tif.imwrite(os.path.join(res_path_mot_c, f), prediction)

if __name__=='__main__':
    #hyp_tuning()
    # eval_interplane_prediction()
    eval_mot_corr()





