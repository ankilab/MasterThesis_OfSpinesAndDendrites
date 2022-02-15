import gc
import sys
import tensorflow as tf
import tifffile
import multiprocessing
from functools import partial

sys.path.insert(1, 'C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites')

import os
from deconv import REGISTRY
import timeit
from data_augmentation import DataAugmenter
import json

p = 'D:/jo77pihe/Registered/20220207_CARE_HypTuning'
p_raw = 'D:/jo77pihe/Registered/20220203_AutoQuant_NotAveraged/Test_data/Raw'
data_path = 'D:/jo77pihe/Registered/20220203_AutoQuant_NotAveraged'
todo_files = [f for f in os.listdir(p_raw) if f.endswith('.tif')]

model_appendix = 'models/my_model'



def train():
    raw_data_path = 'D:/jo77pihe/Registered/20220203_Raw'
    data_path = 'D:/jo77pihe/Registered/20220203_AutoQuant_NotAveraged'
    train_path_d = os.path.join(data_path, 'Train_data', 'Deconved')
    test_path_d = os.path.join(data_path, 'Test_data', 'Deconved')
    train_path_r = os.path.join(data_path, 'Train_data', 'Raw')
    test_path_r = os.path.join(data_path, 'Test_data', 'Raw')
    validation_split = 0.1
    source_folder='Train_data/Raw'
    target_folder='Train_data/Deconved'
    test_folder= test_path_r
    result_path= 'D:/jo77pihe/Registered/20220207_CARE_HypTuning'
    data_augmented_path = result_path
    epochs=100
    n_patches = 100

    lrs = [0.0004, 0.004, 0.04]
    bzs= [4,8,16]
    unet_residuals =[True,False]
    z_shapes = [8, 16]
    xy_shapes = [32,64]
    #n_layers =[1,3]
    ly=3

    #for ly in n_layers:
    for z in z_shapes:
        for xy in xy_shapes:

            args = {}
            args['z_shape'] = z
            args['xy_shape'] = xy
            args['n_patches'] = n_patches
            args['epochs'] = epochs
            args['n_patches'] = n_patches
            args['data_path'] = data_path
            args['result_path'] = result_path

            # Generate training data
            data_augmenter = DataAugmenter(args)
            _, data_dir = data_augmenter.augment(data_path, source_folder, target_folder, care=True,
                                                 data_dir=data_augmented_path)

            for bz in bzs:
                for lr in lrs:
                    for ur in unet_residuals:
                        if z == 8 and xy==32 and bz==4:
                            pass
                        else:
                            print(z,xy,bz,lr, ur)
                            args['learning_rate'] = lr
                            args['unet_residual'] = ur
                            args['batch_size'] = bz

                            # Train CARE model
                            deconvolver = REGISTRY['csbdeep'](args)
                            folder = 'Trial_' + '_'.join((str(z), str(xy), str(bz), str(lr), str(ur), str(ly)))
                            deconvolver.res_path = os.path.join(result_path, folder)
                            model_dir, mdl_path = deconvolver.train(data_dir, validation_split, epochs, bz, lr, ur, unet_n_depth=ly)
                            with open(os.path.join(deconvolver.res_path, "config_run.json"),'w') as outfile:
                                json.dump(args, outfile)
                            # gc.collect()
                            # # predict image
                            # start = timeit.default_timer()
                            # deconvolver.predict(test_path_r, model_dir=deconvolver.res_path, name=model_dir, save_res=True,
                            #                     res_folder=os.path.join(result_path, folder))
                            # l = timeit.default_timer() - start
                            # print(l)

def post_prediction(dir):
    d = os.path.join(p,dir)

    parts = dir.split('_')
    args = {}
    args['z_shape'] = parts[1]
    args['xy_shape'] = parts[2]
    # args['n_patches'] = 100
    # args['epochs'] = 100
    args['data_path'] = p_raw
    args['result_path'] = d
    print(dir)

    files=[f for f in os.listdir(d) if f.endswith('.tif')]
    deconvolver = REGISTRY['csbdeep'](args)

    for t_file in todo_files:
        if t_file in files:
            pass
        else:
            deconvolver.predict_img(tifffile.imread(os.path.join(p_raw, t_file)),
                                    model_dir=d,name=os.path.join(d,model_appendix),
                                    save_as=os.path.join(d, t_file))
    del deconvolver

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.keras.backend.get_session()
    #tf.keras.backend.
    sess.close()

    gc.collect()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    dirs = os.listdir(p)
    dirs = [dir for dir in dirs if dir.startswith('Trial')]
    for dir in dirs:
        p = multiprocessing.Process(target=post_prediction, args=(dir,))
        p.start()
        p.join()
    # # Launch processes
    # with multiprocessing.Pool(processes=1) as p:
    #     p.map(post_prediction, dirs)


