import yaml
from data_augmentation import DataAugmenter, DataProvider
from deconv import REGISTRY
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load configuration
with open('.\\config.yaml', "r") as stream:
    args = yaml.safe_load(stream)

######################### Mu-Net #######################################################################################

# ######################### 3 Levels #####################################################################################
# # Train model
# args['n_levels'] = 3
# args['result_path'] = './Mu_Net_res_3_levels50'
# data_provider = DataProvider((args['z_shape'], args['xy_shape']), args['data_path'], args['source_folder'],
#                              args['target_folder'],
#                              data_file='C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/data32_128.h5')
# deconvolver = REGISTRY['mu-net'](args)
# model_dir, _ = deconvolver.train(data_provider, args['epochs'], args['batch_size'])
#
# # predict image
# test_data = 'D:/jo77pihe/Registered/Raw_32/Test'
# deconvolver.predict(test_data, model_dir)
#
# ######################### 2 Levels #####################################################################################
# # Train model
# args['n_levels'] = 2
# args['result_path'] = './Mu_Net_res_2_levels50'
# data_provider = DataProvider((args['z_shape'], args['xy_shape']), args['data_path'], args['source_folder'],
#                              args['target_folder'],
#                              data_file='C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/data32_128.h5')
# deconvolver = REGISTRY['mu-net'](args)
# model_dir, _ = deconvolver.train(data_provider, args['epochs'], args['batch_size'])
#
# # predict image
# test_data = 'D:/jo77pihe/Registered/Raw_32/Test'
# deconvolver.predict(test_data, model_dir)
#
# ######################## 1 Levels #####################################################################################
# # Train model
# args['n_levels'] = 1
# args['result_path'] = './Mu_Net_res_1_levels50'
# data_provider = DataProvider((args['z_shape'], args['xy_shape']), args['data_path'], args['source_folder'],
#                              args['target_folder'],
#                              data_file='C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/data32_128.h5')
# deconvolver = REGISTRY['mu-net'](args)
# model_dir, _ = deconvolver.train(data_provider, args['epochs'], args['batch_size'])
#
# # predict image
# test_data = 'D:/jo77pihe/Registered/Raw_32/Test'
# deconvolver.predict(test_data, model_dir)

# ######################### 0 Levels #####################################################################################
# # # Train model
# # args['n_levels'] = 0
# args['result_path'] = './Mu_Net_res_0_levels50'
# # data_provider = DataProvider((args['z_shape'], args['xy_shape']), args['data_path'], args['source_folder'],
# #                              args['target_folder'],
# #                              data_file='C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/data32_128.h5')
# deconvolver = REGISTRY['mu-net'](args)
# # model_dir, _ = deconvolver.train(data_provider, args['epochs'], args['batch_size'])
#
# # predict image
# model_dir = 'D:/jo77pihe/Registered/Mu_Net_res_0_levels50/model'
# test_data = 'D:/jo77pihe/Registered/Raw_32/Test'
# deconvolver.predict(test_data, model_dir)

######################### CARE #########################################################################################

########################################################################################################################

args['result_path'] = './CARE_res_x'
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
test_data='D:/jo77pihe/Registered/Raw_32/Test'
deconvolver.predict(test_data, model_dir= deconvolver.res_path, name=model_dir, save_res=True)

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
