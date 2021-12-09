# import psf_generation as psfg
#
# psf_generator = psfg.PSFGenerator()
# psf_generator.generate(512,512, 21)


import sys
import os

# import tf.keras.callbacks

sys.path.insert(1, 'C:/Users/Johan/Documents/FAU_Masterarbeit/Implementation')
import deconv
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import pickle
# import numpy as np
# pd.options.mode.chained_assignment = None
# from matplotlib.colors import LogNorm
# import pylustrator
from skimage import io
from imagequalitymetrics import ImageQualityMetrics
from data_augmentation import DataAugmenter
import numpy as np

MAX_VAL = 12870
MIN_VAL = -2327


def _rescale_mu_net(img):
    img=(img/MAX_VAL)*(MAX_VAL-MIN_VAL)
    img = img + MIN_VAL
    return img

def _rescale_gt(img, max_val, min_val):
    img = (img/np.max(img))*(max_val-min_val)
    img = img + MIN_VAL
    return img


metrics = ImageQualityMetrics()
files = [f for f in os.listdir('D:/jo77pihe/Registered/Raw_32/Test') if f.endswith('.tif')]
gt_path='D:/jo77pihe/Registered/Deconved'
auto_qu_path = 'D:/jo77pihe/Registered/Deconved_AutoQuant_R2'
care_path = 'D:/jo77pihe/Registered/CARE_res/Predictions'
mu_0 = 'D:/jo77pihe/Registered/Mu_Net_res_0_levels50'
mu_1 = 'D:/jo77pihe/Registered/Mu_Net_res_1_levels50'
mu_2 = 'D:/jo77pihe/Registered/Mu_Net_res_2_levels50'
mu_3 = 'D:/jo77pihe/Registered/Mu_Net_res_3_levels50'

pred_list = [auto_qu_path, care_path, mu_0,mu_1,mu_2,mu_3]
# shape[1] -> Care Img, Mu-Net, Autoquant Img, shape[2] -> Metrics: MSE, SSIM, MSSIM, PSNR, NIQE,BRISQUE, SNR
res=np.zeros((len(files),6, 7))
for (i,f) in enumerate(files):
    gt_img=np.float32(io.imread(os.path.join(gt_path, f)))
    for (j, p) in enumerate(pred_list):
        predicted = np.float32(io.imread(os.path.join(p, f)))
        if j>=2:
            predicted=_rescale_mu_net(predicted)
        if j == 0:
            predicted += MIN_VAL
        gt_img = _rescale_gt(gt_img,max_val=np.max(predicted), min_val=np.min(predicted))
        m=metrics.compute_all(predicted,gt_img)
        res[i,j,:] = np.array([m[y] for y in m.keys()])
########################################################################################################################

# X = io.imread(os.path.join('C:\\Users\\Johan\\Documents\\FAU_Masterarbeit\\Implementation\\Registered\\GT','Alessandro_427_ArcCreERT2_Thy1GFP_Ai9_TRAP_2019-08-31_A2.tif'), as_gray=True)
# X = np.double(X)
# i = ImageQualityMetrics()
# i.niqe(X)
########################################################################################################################
args = {}
args['data_path']= '.\\Registered'
args['source_folder']= 'Raw'
args['target_folder']= 'GT'
args['result_path'] = '..\\NN'
args['psf'] = "C:/Users/Johan/Documents/FAU_Masterarbeit/Implementation/Data/PSF"
args['train'] = True
args['batch_size'] = 1
args['z_shape'] = 32
args['xy_shape'] =128
args['n_patches'] = 50
args['data_augmented_path'] = '.\\Registered\\Data_npz'
args['validation_split']=0.1
args['epochs'] = 10
args['batch_size'] = 8
args['train_steps_p_epoch'] = 10


# deconvolver = deconv.Mu_Net(args)
# deconvolver.train(args['source_folder'])

#############################################################################################################
runner = deconv.CAREDeconv(args)

data_augmenter = DataAugmenter(args)
(augmented_raw, augmented_gt, axes), data_dir = data_augmenter.augment(args['data_path'], args['source_folder'],
                                                                       args['target_folder'], care=True,
                                                                       data_dir=args['data_augmented_path'])

runner.preprocess()
model_dir, mdl_path = runner.train(data_dir, args['validation_split'], args['epochs'], args['batch_size'],
                                   args['train_steps_p_epoch'])

##############################################################################################################

x=io.imread(os.path.join('G:/Ghabiba/25X_1NA_Raw/333_Thy1GFP/2019-02-25/A2/333_Thy1eGFP_A2_zoom10_powe490_gain670_z1_ResGalvo-005',
                         '333_Thy1eGFP_A2_zoom10_powe490_gain670_z1_ResGalvo-005_Cycle00001_Ch4_000001.ome.tif'))

print(x.min())
print(x.max())


####################################################################################
import tensorboard

# tensorboard_callback = keras.callbacks.TensorBoard(log_dir='.\\tb')

########
# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='model.png')

#####################################################################################################################
# checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# # Create a new model instance
# model = create_model()
#
# # Load the previously saved weights
# model.load_weights(latest)
#
#
# ######
# with tf.compat.v1.Session(graph=tf.Graph()) as sess:
#     tf.compat.v1.saved_model.loader.load(sess, ["serve"], 'C:/Users/jo77pihe/Downloads/mdl')
#     graph = tf.get_default_graph()
#     print(graph.get_operations())
#
#
# #####
#
#
# with tf.compat.v1.Session() as sess:
#     new_saver = tf.compat.v1.train.import_meta_graph('C:/Users/jo77pihe/Downloads/mdl')
#     new_saver.restore(sess, tf.train.latest_checkpoint('C:/Users/jo77pihe/Downloads/mdl'))
#
#
# #####
# import tensorflow as tf
# import tensorflow.compat.v1 as tf1
#
# def print_checkpoint(save_path):
#   reader = tf.train.load_checkpoint(save_path)
#   reader.
#   shapes = reader.get_variable_to_shape_map()
#   dtypes = reader.get_variable_to_dtype_map()
#   print(f"Checkpoint at '{save_path}':")
#   for key in shapes:
#     print(f"  (key='{key}', shape={shapes[key]}, dtype={dtypes[key].name}, "
#           f"value={reader.get_tensor(key)})")
#
#
# ####
# batch_sz = 1
# self.model_setup()
# saver = tf.compat.v1.train.Saver()
# sess = tf.compat.v1.Session()
# graph = tf.compat.v1.get_default_graph()
# ckpt_path = saver.restore(sess, 'C:/Users/jo77pihe/Downloads/mdl/')
