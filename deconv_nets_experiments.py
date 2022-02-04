import yaml
from data_augmentation import DataAugmenter, DataProvider
from deconv import REGISTRY
import os
import timeit
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load configuration
with open('.\\config.yaml', "r") as stream:
    args = yaml.safe_load(stream)

timing = np.zeros((5,1))
######################### Mu-Net #######################################################################################

# ######################### 3 Levels #####################################################################################
# Train model
args['n_levels'] = 3
args['result_path'] = './Mu_Net_res_3_levels100'
# data_provider = DataProvider((args['z_shape'], args['xy_shape']), args['data_path'], args['source_folder'],
#                              args['target_folder'],
#                              data_file='C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/data32_128.h5')
deconvolver = REGISTRY['mu-net'](args)
# model_dir, _ = deconvolver.train(data_provider, args['epochs'], args['batch_size'])
#
# # predict image
test_data='D:/jo77pihe/Registered/Deconved_AutoQuant_R2/Test_Raw'
model_dir = 'C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/Mu_Net_res_3_levels100_AQ/model'
start = timeit.default_timer()
deconvolver.predict(test_data, model_dir)
l = timeit.default_timer() -start
print(l)
timing[0,0] = l
np.save('timing.npy', timing)

#
# ######################### 2 Levels #####################################################################################
# # Train model
args['n_levels'] = 2
args['result_path'] = './Mu_Net_res_2_levels100'
# data_provider = DataProvider((args['z_shape'], args['xy_shape']), args['data_path'], args['source_folder'],
#                              args['target_folder'],
#                              data_file='C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/data32_128.h5')
deconvolver = REGISTRY['mu-net'](args)
# model_dir, _ = deconvolver.train(data_provider, args['epochs'], args['batch_size'])
#
# # predict image
test_data='D:/jo77pihe/Registered/Deconved_AutoQuant_R2/Test_Raw'
model_dir = 'C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/Mu_Net_res_2_levels100_AQ/model'
start = timeit.default_timer()
deconvolver.predict(test_data, model_dir)
l = timeit.default_timer() -start
print(l)
timing[1,0] = l
np.save('timing.npy', timing)

#
# ######################## 1 Levels #####################################################################################
# # Train model
args['n_levels'] = 1
args['result_path'] = './Mu_Net_res_1_levels100'
# data_provider = DataProvider((args['z_shape'], args['xy_shape']), args['data_path'], args['source_folder'],
#                              args['target_folder'],
#                              data_file='C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/data32_128.h5')
deconvolver = REGISTRY['mu-net'](args)
# model_dir, _ = deconvolver.train(data_provider, args['epochs'], args['batch_size'])
#
# # predict image
test_data='D:/jo77pihe/Registered/Deconved_AutoQuant_R2/Test_Raw'
model_dir = 'C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/Mu_Net_res_1_levels100_AQ/model'
start = timeit.default_timer()
deconvolver.predict(test_data, model_dir)
l = timeit.default_timer() -start
print(l)
timing[2,0] = l
np.save('timing.npy', timing)

# ######################### 0 Levels #####################################################################################
# # # Train model
args['n_levels'] = 0
args['result_path'] = './Mu_Net_res_0_levels100'
# # data_provider = DataProvider((args['z_shape'], args['xy_shape']), args['data_path'], args['source_folder'],
# #                              args['target_folder'],
# #                              data_file='C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/data32_128.h5')
deconvolver = REGISTRY['mu-net'](args)
# # model_dir, _ = deconvolver.train(data_provider, args['epochs'], args['batch_size'])
#
# # predict image
test_data='D:/jo77pihe/Registered/Deconved_AutoQuant_R2/Test_Raw'
model_dir = 'C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/Mu_Net_res_0_levels100_AQ/model'
start = timeit.default_timer()
deconvolver.predict(test_data, model_dir)
l = timeit.default_timer() -start
print(l)
timing[3,0] = l

np.save('timing.npy', timing)
######################### CARE #########################################################################################

########################################################################################################################

args['result_path'] = './CARE_res_x100'
args['learning_rate'] = 0.0004

# # Generate training data
data_augmenter = DataAugmenter(args)
_, data_dir = data_augmenter.augment(args['data_path'], args['source_folder'], args['target_folder'], care=True,
                                     data_dir=args['data_augmented_path'])
# # Train CARE model
deconvolver = REGISTRY['csbdeep'](args)
#data_dir = './Registered/Data_npz/my_data.npz'
model_dir, mdl_path = deconvolver.train(data_dir, args['validation_split'], args['epochs'], args['batch_size'])

# predict image
test_data='D:/jo77pihe/Registered/Deconved_AutoQuant_R2/Test_Raw'
res_path = 'D:/jo77pihe/Registered/CARE_res_x100'
start = timeit.default_timer()

deconvolver.predict(test_data, model_dir= res_path, name=model_dir, save_res=True)
l = timeit.default_timer() -start
print(l)
timing[4,0] = l
np.save('timing.npy', timing)

###############################################Auto-Encoder#############################################################

########################################################################################################################

# denoiser = REGISTRY['autoencoder'](args)
# model_dir, train_history = denoiser.train(args['epochs'], args['batch_size'])
# X=io.imread('D:/jo77pihe/Registered/Raw_32/Test/Alessandro_427_ArcCreERT2_Thy1GFP_Ai9_TRAP_2019-08-31_A4.tif')
# img = denoiser.predict_img(X,(32,64,64), 8192, 1)
# tifffile.imsave('denoiser.tif', img)
#
# with open('trainj_hist_den.pkl', 'wb') as outfile:
#     pickle.dump(train_history, outfile, pickle.HIGHEST_PROTOCOL)
