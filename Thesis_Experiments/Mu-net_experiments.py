import sys
sys.path.insert(1, 'C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites')

from data_augmentation import DataAugmenter, DataProvider
from deconv import REGISTRY
import os
import timeit
import numpy as np

res_path = 'D:/jo77pihe/Registered/20220214_Mu-Net'
data_path = 'D:/jo77pihe/Registered/20220203_AutoQuant_NotAveraged/Train_data'
levels =[0,1,2,3]
z_shapes=[16]#,32,64]
xy_shapes =[128] #[64
lr = [0.0001, 0.001]
bzs = [4, 8, 16]


def main():

    for l in levels:
        for z in z_shapes:
            for xy in xy_shapes:
                for lrx in lr:
                    if l == 0 and z==16 and xy==64:
                        pass
                    elif l == 0 and z==16 and xy==128 and lr==0.0001:
                        pass
                    else:
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
                        args['batch_size']=16

                        args['result_path'] = os.path.join(res_path, '_'.join(('Trial', str(l),str(z),str(xy),str(lrx), str(args['batch_size']))))
                        data_provider = DataProvider((args['z_shape'], args['xy_shape']), args['data_path'], args['source_folder'],
                                                      args['target_folder'],n_patches=args['n_patches'], data_file = 'data_'+str(z)+'_'+str(xy)+'.h5')
                        deconvolver = REGISTRY['mu-net'](args)
                        model_dir, _ = deconvolver.train(data_provider, args['epochs'], args['batch_size'])

if __name__ == '__main__':
    main()