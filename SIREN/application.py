from SIREN.network import SIREN
from SIREN import utils
import numpy as np
import os
import imageio as io
import normalize


MAX_VAL = 12870
MIN_VAL = -2327


class SIREN_application:
    def __init__(self, args):
        hidden_features = args.get('hidden_features', 256)
        hidden_layers = args.get('hidden_layers', 1)

        self.model = SIREN(in_features=3,
                      out_features=1,
                      hidden_features=hidden_features,
                      hidden_layers=hidden_layers,
                      outermost_linear=True,
                      sine=True)

    def preprocess(self, **kwargs):
        pass

    def train(self, X, n_steps, steps_to_plot, batch_size):
        if batch_size is None:
            batch_size = len(X['y'])
        self.model.preprocess(X['grid'].astype(np.float32), X['y'])
        loss =self.model.train(n_steps, X['grid'], X['y'], steps_to_plot, X['original_shape'], batch_size=batch_size)
        return loss

    def test(self, X):
        return self.model.predict(X['grid'], batchsize=len(X['y'])).reshape(X['original_shape'])


class InterplanePrediction(SIREN_application):
    def __init__(self, args):
        super().__init__(args)


    def preprocess(self, img3d, train_plane_step_size=2):
        self.tpss = train_plane_step_size
        img3d -= img3d.min()  # * 2 - 1
        img3d = img3d / img3d.max()
        xx, yy, zz = utils.rescale_indices(img3d.shape, z=.05)

        # Filter planes used for training and testing
        mask = np.zeros(img3d.shape[0], dtype=bool)
        idx=np.arange(0,img3d.shape[0])[::self.tpss]
        mask[idx,] =True

        # subsample stack and grid
        gt = {}
        xx_gt, yy_gt, zz_gt = xx[mask, :, :], yy[mask, :, :], zz[mask, :, :]
        gt['grid'] = utils.flatten_meshgrid(img3d.shape, xx_gt, yy_gt, zz_gt)
        stack_gt = img3d[mask, :, :]
        gt['y'] = stack_gt.reshape(-1)
        gt['original_shape'] = stack_gt.shape

        test_data = {}
        xx_test, yy_test, zz_test = xx[~mask, :, :], yy[~mask, :, :], zz[~mask, :, :]
        test_data['grid'] = utils.flatten_meshgrid(img3d.shape, xx_test, yy_test, zz_test)
        stack_test = img3d[~mask, :, :]
        test_data['y'] = stack_test.reshape(-1)
        test_data['original_shape'] = stack_test.shape
        return gt, test_data

    def train(self, X, n_steps=1000, steps_to_plot=1000, batch_size=None):
        super().train(X, n_steps, steps_to_plot, batch_size)


    def test(self, X):
        return super(InterplanePrediction, self).test(X)


class Motion_Correction(SIREN_application):
    def __init__(self, args):
        super().__init__(args)
        self.normalizer = normalize.PercentileNormalizer()

    def preprocess(self, data_path, img_name):
        files = [f for f in os.listdir(data_path) if (os.path.isfile(os.path.join(data_path, f)) and (img_name in f))]
        ids = [str(x) for x in np.arange(1, len(files) + 1)]
        print(ids)
        y_x, grid_x, stack = np.array([]), np.array([]), np.array([])
        batch_size =0
        for i in range(len(ids)):
            stack = np.asarray(io.mimread(os.path.join(data_path, img_name + ids[i] + ".tif")), dtype=np.float32)[:10, ::2, ::2]
            stack = self.normalizer.normalize(stack)
            # stack -= MIN_VAL
            # stack = stack / MAX_VAL
            y = stack.reshape(-1)

            if i == 0:
                xx, yy, zz = utils.rescale_indices(stack.shape, z=.05)
                grid = utils.flatten_meshgrid(stack.shape, xx, yy, zz)
                y_x, grid_x = y, grid
                batch_size = len(y)
            else:
                y_x = np.concatenate((y_x, y), axis=0)
                grid_x = np.concatenate((grid_x, grid), axis=0)
            print(i)
        X={}
        X['grid'] = grid_x
        X['y'] =y_x
        X['original_shape']=stack.shape
        return X, batch_size

    def train(self, X, n_steps=1000, steps_to_plot=1000, batch_size=None):
        super(Motion_Correction, self).train(X, n_steps, steps_to_plot, batch_size)

    def test(self, X):
        return super(Motion_Correction, self).test(X)



