import os
import sys
sys.path.insert(1, 'C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites')

from deconv import REGISTRY
from data_augmentation import DataAugmenter, DataProvider

import json

p_raw = 'D:/jo77pihe/Registered/20220223_Deconv_comp/Test_data/Raw'
data_path = 'D:/jo77pihe/Registered/20220223_Deconv_comp'
model_appendix = 'models/my_model'

def train_care():
    #raw_data_path = 'D:/jo77pihe/Registered/20220203_Raw'
    data_path = 'D:/jo77pihe/Registered/20220223_Deconv_comp'
    # train_path_d = os.path.join(data_path, 'Train_data', 'Deconved')
    # test_path_d = os.path.join(data_path, 'Test_data', 'Deconved')
    # train_path_r = os.path.join(data_path, 'Train_data', 'Raw')
    test_path_r = os.path.join(data_path, 'Test_data', 'Raw')
    validation_split = 0.1
    source_folder='Train_data/Raw'
    target_folder='Train_data/Deconved'
    # test_folder= test_path_r
    result_path= 'D:/jo77pihe/Registered/20220223_Deconv_comp/CARE'
    data_augmented_path = result_path

    args = {}
    args['z_shape'] = 16
    args['xy_shape'] = 32
    args['n_patches'] = 100
    args['epochs'] = 100
    args['data_path'] = data_path
    args['result_path'] = result_path

    # Generate training data
    data_augmenter = DataAugmenter(args)
    _, data_dir = data_augmenter.augment(data_path, source_folder, target_folder, care=True,
                                         data_dir=data_augmented_path)

    args['learning_rate'] = 0.0004
    args['unet_residual'] = True
    args['batch_size'] = 16

    # Train CARE model
    deconvolver = REGISTRY['csbdeep'](args)
    deconvolver.res_path = result_path
    model_dir, mdl_path = deconvolver.train(data_dir, validation_split, args['epochs'], args['batch_size'],
                                            args['learning_rate'], args['unet_residual'], unet_n_depth=2)
    deconvolver.predict(test_path_r, model_dir,mdl_path,os.path.join(mdl_path,model_appendix))
    with open(os.path.join(deconvolver.res_path, "config_run.json"),'w') as outfile:
        json.dump(args, outfile)

def train_mun():
    args = {}
    args['z_shape'] = 16
    args['xy_shape'] = 64
    args['n_levels'] = 3
    args['data_path'] = data_path
    args['source_folder'] = 'Raw'
    args['target_folder'] = 'Deconved'
    args['lr'] = 0.001
    args['n_patches'] = 100
    args['epochs'] = 100
    args['batch_size'] = 16

    args['result_path'] = os.path.join(data_path, 'MuNet')
    data_provider = DataProvider((args['z_shape'], args['xy_shape']), args['data_path'],
                                 args['source_folder'],
                                 args['target_folder'], n_patches=args['n_patches'],
                                 data_file='data.h5')
    deconvolver = REGISTRY['mu-net'](args)
    model_dir, _ = deconvolver.train(data_provider, args['epochs'], args['batch_size'])
    deconvolver.predict(p_raw,os.path.join(args['result_path'], 'model'))

def predict_blind_rl():
    args = {}
    args['data_path']= ''
    args['source_folder']= p_raw
    args['target_folder']= ''
    args['result_path'] = os.path.join(data_path, 'BlindRL')
    args['psf'] = "C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/PSF"

    blind_rl = REGISTRY['BlindRL'](args)
    blind_rl.predict(args['source_folder'], 5, 1, 3, 1, eval_img_steps = False, save_intermediate_res= True,
                         parallel=True, plot_frequency=1)

def predict_care():
    data_path = 'D:/jo77pihe/Registered/20220223_Deconv_comp'
    test_path_r = os.path.join(data_path, 'Test_data', 'Raw')
    result_path= 'D:/jo77pihe/Registered/20220223_Deconv_comp/CARE'

    args = {}
    args['z_shape'] = 16
    args['xy_shape'] = 32
    args['n_patches'] = 100
    args['epochs'] = 100
    args['data_path'] = data_path
    args['result_path'] = result_path

    deconvolver = REGISTRY['csbdeep'](args)
    deconvolver.predict(test_path_r, result_path, os.path.join(result_path, model_appendix), os.path.join(result_path, model_appendix))


if __name__ == '__main__':
    # multiprocessing.freeze_support()
    # predict_blind_rl()
    # train_care()
    #train_mun()
    predict_care()

