from SIREN import SIREN
import numpy as np
import imageio as io

m = SIREN(in_features=3,
         out_features=1,
         hidden_features=256,
         hidden_layers=3,
         outermost_linear=True)

stack = np.asarray(io.mimread(
    r"C:\Users\Johan\Documents\FAU_Masterarbeit\MasterThesis_OfSpinesAndDendrites\Registered\Alessandro_427_ArcCreERT2_Thy1GFP_Ai9_TRAP_2019-08-31_A2.tif"),
    dtype=np.float32)[1::4, ::2, ::2]
X = m.get_mgrid3(stack.shape, z=.05)
y = stack.reshape(-1) / stack.max()# * 2 - 1

steps = 500
step_to_plot = 50

m.train(steps, X,y,step_to_plot, stack.shape)