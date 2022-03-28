import timeit
import numpy as np
import multiprocessing
import os
from deconv import REGISTRY


p1='D:/jo77pihe/Registered/20220308_Subsampled_AnkiVals/10_128'
p2='D:/jo77pihe/Registered/20220308_Subsampled_AnkiVals/20_256'
p3='D:/jo77pihe/Registered/20220308_Subsampled_AnkiVals/20_512'
p4='D:/jo77pihe/Registered/20220308_Subsampled_AnkiVals/40_512'
p5='D:/jo77pihe/Registered/20220308_Subsampled_AnkiVals/80_512'
ps=[p1,p2,p3,p4,p5]
d_care='D:/jo77pihe/Registered/20220207_CARE_HypTuning/Trial_8_32_16_0.0004_True'
model_appendix = 'models/my_model'
parts = d_care.split('_')
args = {}
args['z_shape'] = parts[1]
args['xy_shape'] = parts[2]
###########mu-net params
ptrial = 'Trial_3_16_64_0.001_16'
d_mu = os.path.join('D:/jo77pihe/Registered/20220214_Mu-Net', ptrial)
parts = ptrial.split('_')
args = {}
args['z_shape'] = int(parts[2])
args['xy_shape'] = int(parts[3])
args['n_levels'] = int(parts[1])

def predict_care(args):
    deconvolver = REGISTRY['csbdeep'](args)
    start = timeit.default_timer()
    deconvolver.predict(args['data_path'],model_dir=d_care,name=os.path.join(d_care,model_appendix),
                                res_folder=args['result_path'])
    t=timeit.default_timer()-start

    return t

def predict_mu(args):
    print('here')
    deconvolver = REGISTRY['mu-net'](args)
    deconvolver.res_path =args['result_path']
    t=deconvolver.predict(args['data_path'],os.path.join(d_mu, 'model'))
    print(t)
    return t

if __name__ == '__main__':
    multiprocessing.freeze_support()

    t_care = np.zeros((5, 4))
    for (ip, p) in enumerate(ps):

        args['data_path'] = p
        args['result_path'] = p + 'results_mu16'
        args['train_flag']=False
        #t_care[ip, 0] = predict_mu(args)
        p = multiprocessing.Process(target=predict_mu, args=(args,))
        p.start()
        p.join()

