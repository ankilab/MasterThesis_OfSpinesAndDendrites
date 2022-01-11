from SIREN import SIREN
import numpy as np
import imageio as io
from SIREN import utils
import matplotlib.pyplot as plt


def learn_subsampled(stack, grid, hidden_features, hidden_layers, max_val, n_steps=1000, steps_to_plot=100):
    model = SIREN(in_features=3,
             out_features=1,
             hidden_features=hidden_features,
             hidden_layers=hidden_layers,
             outermost_linear=True)

    y = stack.reshape(-1) / max_val# * 2 - 1
    model.train(n_steps, grid,y,steps_to_plot, stack.shape)
    return model


if __name__ == "__main__":
    stack = np.asarray(io.mimread(
        r"C:\Users\Johan\Documents\FAU_Masterarbeit\MasterThesis_OfSpinesAndDendrites\Registered\Alessandro_427_ArcCreERT2_Thy1GFP_Ai9_TRAP_2019-08-31_A2.tif"),
        dtype=np.float32)[:,::4,::4]
    stack -=stack.min()
    xx, yy, zz = utils.rescale_indices(stack.shape, z=.05)

    # subsample stack and grid
    xx_gt, yy_gt, zz_gt = xx[::2, :, :], yy[::2, :, :], zz[::2, :, :]
    grid_gt=utils.flatten_meshgrid(stack.shape, xx_gt, yy_gt, zz_gt)
    stack_gt = stack[::2, :, :]

    xx_test, yy_test, zz_test = xx[1::2, :, :], yy[1::2, :, :], zz[1::2, :, :]
    grid_test=utils.flatten_meshgrid(stack.shape, xx_test, yy_test, zz_test)
    stack_test = stack[1::2, :, :]

    # Train model
    model=learn_subsampled(stack_gt, grid_gt, 256, 1, stack.max(), n_steps=50, steps_to_plot=10)

    # Test model
    y = stack_test.reshape(-1) / stack.max()# * 2 - 1

    plt.figure()
    plt.imshow(model.model.predict(grid_test, batch_size=len(y)).reshape(xx_test.shape).sum(0), cmap='gray')
    plt.show()
