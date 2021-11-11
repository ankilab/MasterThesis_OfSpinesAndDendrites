import yaml
from data_augmentation import DataAugmenter
import deconv
from skimage import io
import os


# Load configuration
with open('.\\config.yaml', "r") as stream:
    args=yaml.safe_load(stream)

# # Generate training data
# data_augmenter = DataAugmenter(args)
# _, data_dir = data_augmenter.augment(args['data_path'], args['source_folder'], args['target_folder'], care=True,
#                                      data_dir=args['data_augmented_path'])
#
# # Train CARE model
deconvolver = deconv.CAREDeconv(args)
# model_dir, mdl_path = deconvolver.train(data_dir, args['validation_split'], args['epochs'], args['batch_size'],
#                                    args['train_steps_p_epoch'])
# print(model_dir)
model_dir='models/my_model'
# predict image
X=io.imread('C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/Registered/Alessandro_520_ArcCreERT2_Thy1GFP_Ai9_TRAP_2019-08-31_A2.tif')
deconvolver.predict(X, model_dir= args['result_path'], name=model_dir, save_as='care_rediction.tif')