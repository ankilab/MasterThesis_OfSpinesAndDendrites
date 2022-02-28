import numpy as np
import sys
sys.path.insert(1, 'C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites')
import multiprocessing
from data_augmentation import DataProvider
from deconv import REGISTRY
import os


res_path = 'D:/jo77pihe/Registered/20220214_Mu-Net'
data_path = 'D:/jo77pihe/Registered/20220203_AutoQuant_NotAveraged/Train_data'
test_data_dir = 'D:/jo77pihe/Registered/20220203_AutoQuant_NotAveraged/Test_data/Raw'

levels =[0,1,2,3]
z_shapes=[16]#,32,64]
xy_shapes =[64,128] #[64
lr = [0.01]
bzs = [4, 8]#, 16]


def main():
    for bz in bzs:
        for l in levels:
            for z in z_shapes:
                for xy in xy_shapes:
                    for lrx in lr:
                        args={}
                        args['z_shape']=z
                        args['xy_shape'] = xy
                        args['n_levels'] = l
                        args['data_path'] = data_path
                        args['source_folder'] = 'Raw'
                        args['target_folder'] = 'Deconved'
                        args['lr'] = lrx
                        args['n_patches']=100
                        args['epochs']=100
                        args['batch_size']=bz

                        args['result_path'] = os.path.join(res_path, '_'.join(('Trial', str(l),str(z),str(xy),str(lrx), str(args['batch_size']))))
                        data_provider = DataProvider((args['z_shape'], args['xy_shape']), args['data_path'], args['source_folder'],
                                                      args['target_folder'],n_patches=args['n_patches'], data_file = 'data_'+str(z)+'_'+str(xy)+'.h5')
                        deconvolver = REGISTRY['mu-net'](args)
                        model_dir, _ = deconvolver.train(data_provider, args['epochs'], args['batch_size'])


def post_prediction(dir):
    # runs in PyCharm debugger, but not in console
    path=os.path.join(res_path, dir)
    parts = dir.split('_')
    args = {}
    args['z_shape'] = int(parts[2])
    args['xy_shape'] = int(parts[3])
    args['n_levels'] = int(parts[1])
    args['data_path'] = data_path
    args['source_folder'] = 'Raw'
    args['target_folder'] = 'Deconved'
    args['lr'] = np.float(parts[4])
    args['n_patches'] = 100
    args['epochs'] = 100
    args['batch_size'] = int(parts[5])
    args['result_path']=path
    deconvolver = REGISTRY['mu-net'](args)
    deconvolver.res_path =path
    deconvolver.predict(test_data_dir,os.path.join(path, 'model'))



if __name__ == '__main__':
    main()

    # multiprocessing.freeze_support()
    # dirs = os.listdir(res_path)
    # dirs = [dir for dir in dirs if dir.startswith('Trial')]
    # for dir in dirs:
    #     if 'history_mu_net.pkl' in os.listdir(os.path.join(res_path,dir)):
    #         p = multiprocessing.Process(target=post_prediction, args=(dir,))
    #         p.start()
    #         p.join()