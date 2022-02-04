from SIREN import SIREN
import numpy as np
import imageio as io
from SIREN import utils
import matplotlib.pyplot as plt
import tifffile as tif
import os
from imagequalitymetrics import ImageQualityMetrics


MAX_VAL = 12870
MIN_VAL = -2327


def learn_subsampled(y, grid, hidden_features, hidden_layers, n_steps=1000, steps_to_plot=100, sine=True, batch_size=10000):
    model = SIREN(in_features=3,
             out_features=1,
             hidden_features=hidden_features,
             hidden_layers=hidden_layers,
             outermost_linear=True,
                  sine= sine)
    model.preprocess(grid.astype(np.float32),y)
    model.train(n_steps, grid.astype(np.float32),y,steps_to_plot, stack.shape, batch_size=batch_size)
    return model


if __name__ == "__main__":
    path ='D:\\jo77pihe\\Registered\\Repetitions'
    name= 'Alessandro_514_ArcCreERT2_Thy1GFP_Ai9_TRAP_2019-08-31_A1'
    files = [f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and (name in f))]
    ids = [str(x) for x in np.arange(1,len(files)+1)]
    y_x, grid_x =np.array([]), np.array([])
    batch_size = 0
    for i in range(len(ids)):
        stack= np.asarray(io.mimread(os.path.join(path, name + ids[i]+".tif")), dtype=np.float32)[:10,::2,::2]
        stack -=MIN_VAL
        stack = stack/ MAX_VAL
        y = stack.reshape(-1)

        if i ==0:
            xx, yy, zz = utils.rescale_indices(stack.shape, z=.05)
            grid = utils.flatten_meshgrid(stack.shape, xx, yy, zz)
            y_x, grid_x= y, grid
            batch_size= len(y)
        else:
            y_x=np.concatenate((y_x, y), axis=0)
            grid_x = np.concatenate((grid_x, grid), axis=0)

    # Train model
    model=learn_subsampled(y_x, grid_x, 256, 3, n_steps=1000, steps_to_plot=0, sine=True, batch_size=batch_size)

    # Test model
    prediction = model.model.predict(grid, batch_size=len(y)).reshape(xx.shape)
    tif.imsave('SIREN_mc_pred.tif', prediction)

    plt.figure()
    plt.imshow(prediction.sum(0), cmap='gray')
    plt.title('Test')
    plt.show()

