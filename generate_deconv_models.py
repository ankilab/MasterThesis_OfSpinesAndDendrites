import pickle

import tifffile
import yaml
from data_augmentation import DataAugmenter, DataProvider
from deconv import REGISTRY
from skimage import io
import os
from deconv import REGISTRY



# Load configuration
with open('.\\config.yaml', "r") as stream:
    args=yaml.safe_load(stream)

# # # Generate training data
# data_augmenter = DataAugmenter(args)
# _, data_dir = data_augmenter.augment(args['data_path'], args['source_folder'], args['target_folder'], care=True,
#                                      data_dir=args['data_augmented_path'])
# #
# # Train CARE model
# deconvolver = REGISTRY['csbdeep'](args)
#
# # deconvolver = deconv.CAREDeconv(args)
# # model_dir, mdl_path = deconvolver.train(data_dir, args['validation_split'], args['epochs'], args['batch_size'])
# # print(model_dir)
# model_dir='models/my_model'
#
# # predict image
# # X=io.imread('D:/jo77pihe/Registered/Raw_32/Test/Alessandro_427_ArcCreERT2_Thy1GFP_Ai9_TRAP_2019-08-31_A4.tif')
# test_data='D:/jo77pihe/Registered/Raw_32/Test'
# deconvolver.predict(test_data, model_dir= deconvolver.res_path, name=model_dir, save_res=True)
#
#
#
# # # Generate training data
# data_augmenter = DataAugmenter(args)
# _, data_dir = data_augmenter.augment(args['data_path'], args['source_folder'], args['target_folder'], care=True,
#                                      data_dir=args['data_augmented_path'])

########################################Mu-Net##########################################################################
# # Train model
# data_provider= DataProvider((args['z_shape'],args['xy_shape']), args['data_path'], args['source_folder'], args['target_folder'],  # data_file='C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/data.h5')
#                             data_file='C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/data32_128.h5')
# deconvolver = REGISTRY['mu-net'](args)
# #
# model_dir, train_history=deconvolver.train(data_provider, args['epochs'], args['batch_size'])
# # # train_history = 'C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/model/train_history.npy'
# # deconvolver.plot_training(train_history)
#
# # predict image
# # X=io.imread('D:/jo77pihe/Registered/Raw_32/Test/Alessandro_427_ArcCreERT2_Thy1GFP_Ai9_TRAP_2019-08-31_A4.tif')
# test_data='D:/jo77pihe/Registered/Raw_32/Test'
# model_dir ='C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/model'
# deconvolver.predict(test_data, model_dir)


###############################################Auto-Encoder#############################################################
denoiser = REGISTRY['autoencoder'](args)
model_dir, train_history = denoiser.train(args['epochs'], args['batch_size'])
X=io.imread('D:/jo77pihe/Registered/Raw_32/Test/Alessandro_427_ArcCreERT2_Thy1GFP_Ai9_TRAP_2019-08-31_A4.tif')
img = denoiser.predict_img(X,(32,64,64), file_name='Auto_enc_Alessandro_427_ArcCreERT2_Thy1GFP_Ai9_TRAP_2019-08-31_A4.tif', model_dir=model_dir)
