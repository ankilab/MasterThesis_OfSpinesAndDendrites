# import psf_generation as psfg
#
# psf_generator = psfg.PSFGenerator()
# psf_generator.generate(512,512, 21)


import sys
import os
sys.path.insert(1, 'C:/Users/Johan/Documents/FAU_Masterarbeit/Implementation')
import deconv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
pd.options.mode.chained_assignment = None
from matplotlib.colors import LogNorm
import pylustrator
from skimage import io
from imagequalitymetrics import ImageQualityMetrics


# X = io.imread(os.path.join('C:\\Users\\Johan\\Documents\\FAU_Masterarbeit\\Implementation\\Registered\\GT','Alessandro_427_ArcCreERT2_Thy1GFP_Ai9_TRAP_2019-08-31_A2.tif'), as_gray=True)
# X = np.double(X)
# i = ImageQualityMetrics()
# i.niqe(X)
args = {}
args['data_path']= ''
args['source_folder']= 'C:\\Users\\Johan\\Documents\\FAU_Masterarbeit\\Implementation\\Registered'
args['target_folder']= ''
args['result_path'] = '..\\Blind_RL'
args['psf'] = "C:/Users/Johan/Documents/FAU_Masterarbeit/Implementation/Data/PSF"
args['train'] = True


deconvolver = deconv.Mu_Net(args)
deconvolver.train(args['source_folder'])

#####################################################################################################################
