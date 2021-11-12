import yaml
from data_augmentation import DataAugmenter
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
#
# # Train CARE model
# deconvolver = REGISTRY['csbdeep'](args)
#
# # deconvolver = deconv.CAREDeconv(args)
# model_dir, mdl_path = deconvolver.train(data_dir, args['validation_split'], args['epochs'], args['batch_size'])
# print(model_dir)
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
# Train model
deconvolver = REGISTRY['mu-net'](args)

deconvolver.train(args['epochs'], args['batch_size'])

# predict image
# X=io.imread('D:/jo77pihe/Registered/Raw_32/Test/Alessandro_427_ArcCreERT2_Thy1GFP_Ai9_TRAP_2019-08-31_A4.tif')
test_data='D:/jo77pihe/Registered/Raw_32/Test'
deconvolver.predict(test_data, save_res=True)