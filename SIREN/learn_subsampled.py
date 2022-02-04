from SIREN import SIREN
import numpy as np
import imageio as io
from SIREN import utils
import matplotlib.pyplot as plt
import tifffile as tif
from imagequalitymetrics import ImageQualityMetrics


def learn_subsampled(stack, grid, hidden_features, hidden_layers, n_steps=1000, steps_to_plot=100, sine=True):
    model = SIREN(in_features=3,
             out_features=1,
             hidden_features=hidden_features,
             hidden_layers=hidden_layers,
             outermost_linear=True,
                  sine= sine)

    y = stack.reshape(-1)
    model.preprocess(grid.astype(np.float32),y)
    model.train(n_steps, grid,y,steps_to_plot, stack.shape, batch_size=128)
    return model


if __name__ == "__main__":
    stack = np.asarray(io.mimread(
        r"..\Registered\GT\Alessandro_427_ArcCreERT2_Thy1GFP_Ai9_TRAP_2019-08-31_A2.tif"),
        dtype=np.float32)[:20,::2,::2]
    stack -=stack.min() # * 2 - 1
    stack = stack/ stack.max()
    xx, yy, zz = utils.rescale_indices(stack.shape, z=.05)

    # subsample stack and grid
    xx_gt, yy_gt, zz_gt = xx[::2, :, :], yy[::2, :, :], zz[::2, :, :]
    grid_gt=utils.flatten_meshgrid(stack.shape, xx_gt, yy_gt, zz_gt)
    stack_gt = stack[::2, :, :]

    xx_test, yy_test, zz_test = xx[1::2, :, :], yy[1::2, :, :], zz[1::2, :, :]
    grid_test=utils.flatten_meshgrid(stack.shape, xx_test, yy_test, zz_test)
    stack_test = stack[1::2, :, :]

    # Train model
    model=learn_subsampled(stack_gt, grid_gt, 256, 3, n_steps=1500, steps_to_plot=100, sine=True)

    # Test model
    y = stack_test.reshape(-1)
    prediction = model.model.predict(grid_test, batch_size=len(y)).reshape(xx_test.shape)
    tif.imsave('SIREN_pred_batch.tif', prediction)
    tif.imsave('SIREN_gt.tif', stack_test)

    iqm = ImageQualityMetrics()
    print('SSIM TEst', iqm.ssim(prediction,stack_test))
    print('PSNR TEst', iqm.psnr(prediction,stack_test))
    print('MSE TEst', iqm.mse(prediction,stack_test))

    plt.figure()
    plt.imshow(prediction.sum(0), cmap='gray')
    plt.title('Test')
    plt.show()

    plt.figure()
    plt.imshow(stack_test.sum(0), cmap='gray')
    plt.title('Test_GT')
    plt.show()

    y = stack_gt.reshape(-1)
    prediction_gt = model.model.predict(grid_gt, batch_size=len(y)).reshape(xx_gt.shape)

    print('SSIM Train', iqm.ssim(prediction_gt,stack_gt))
    print('PSNR Train', iqm.psnr(prediction_gt,stack_gt))
    print('MSE Train', iqm.mse(prediction_gt,stack_gt))
