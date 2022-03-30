from SIREN.network import SIREN
from SIREN import utils
import numpy as np
import os
import imageio as io
import normalize
from pystackreg import StackReg


MAX_VAL = 12870
MIN_VAL = -2327


class SIREN_application:
    """
    Super-class for applications based on SIREN. Each object has a preprocess, train and test function.

    """

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
        """
        Implemeted in derived classes.

        """
        pass

    def train(self, X, n_steps, batch_size, result_path):
        """
        Train SIREN network.

        :param X: Input data providing the coordinates (X['grid']) and the corresponding output value (X['y'])
                    e.g. for images the pixel coordinates (x,y,z) are given and the gray value.
        :type X: dict with 'grid' and 'y'
        :param n_steps: Number of training steps
        :type n_steps: int
        :param batch_size: Batch size
        :type batch_size: int
        :param result_path: Path to folder where to model weights
        :type result_path: str
        :return: Training loss
        :rtype: list[float]
        """

        if batch_size is None:
            batch_size = len(X['y'])
        self.model.preprocess(X['grid'].astype(np.float32), X['y'])
        loss =self.model.train(n_steps, X['grid'], X['y'], batch_size=batch_size)
        self.model.model.save(os.path.join(result_path, 'model'))
        return loss

    def test(self, X):
        """
        Compute predictions for input data.

        :param X: Input data
        :return: Predictions
        """
        return self.model.predict(X['grid_test'], batch_size=len(X['y'])).reshape(X['original_shape'])


class InterplanePrediction(SIREN_application):
    """
    Implements learning the implicit representation of the training data and allows for predicting function
    values at arbitrary positions.

    """

    def __init__(self, args):
        super().__init__(args)


    def preprocess(self, img3d, train_plane_step_size=2):
        """
        Preprocess data: create input grid and split data into training and validation or test set.

        :param img3d: Input data (e.g. image stack)
        :type img3d: nd.array
        :param train_plane_step_size: Offset between training planes. Remaining planes are assigned to validation or
                                        test set.
        :type train_plane_step_size: int
        :return: Training and test data.
        :rtype: dict, dict
        """
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
        gt['grid_test']=gt['grid']
        stack_gt = img3d[mask, :, :]
        gt['y'] = stack_gt.reshape(-1)
        gt['original_shape'] = stack_gt.shape

        test_data = {}
        xx_test, yy_test, zz_test = xx[~mask, :, :], yy[~mask, :, :], zz[~mask, :, :]
        test_data['grid_test'] = utils.flatten_meshgrid(img3d.shape, xx_test, yy_test, zz_test)
        stack_test = img3d[~mask, :, :]
        test_data['y'] = stack_test.reshape(-1)
        test_data['original_shape'] = stack_test.shape
        return gt, test_data

    def train(self, X, n_steps=1000, steps_to_plot=1000, batch_size=None, result_path='./'):
        """
        Calls super-class implementation.

        """

        super().train(X, n_steps, batch_size, result_path)


    def test(self, X):
        """
        Calls super-class implementation.

        """

        return super(InterplanePrediction, self).test(X)


class Motion_Correction(SIREN_application):
    """
    Implements learning the implicit representation of several images of the same scene. We found that this allows for
    correcting motion artifacts.

    """

    def __init__(self, args):
        super().__init__(args)
        self.normalizer = normalize.PercentileNormalizer()

    def preprocess(self, data_path, img_name):
        """
        Create data structures for training. Assumed that the repetitions are named with the same file name and an
        appended index. E.g. stack1.tif, stack2.tif, stack3.tif

        :param data_path: Data path to location where repetitions of one scene are found.
        :param img_name: Name of the image to be processed e.g. stack.tif
        :return: Training data, batch size (= size of one image)
        :rtype: dict, int
        """
        img_name = img_name[:-4]
        files = [f for f in os.listdir(data_path) if (os.path.isfile(os.path.join(data_path, f)) and (img_name in f))]
        ids = [str(x) for x in np.arange(1, len(files) + 1)]
        if len(ids)>4:
            ids = ids[:4]
        print(ids)
        y_x, grid_x, stack = np.array([]), np.array([]), np.array([])
        batch_size =0
        #prev_stack = np.zeros((5,512,512))
        for i in range(len(ids)):
            stack = np.asarray(io.mimread(os.path.join(data_path, img_name + ids[i] + ".tif")), dtype=np.float32)[:5, :,:]
            stack = self.normalizer.normalize(stack)
            if i >0:
                stack_x = np.zeros(stack.shape)
                for p in range(stack.shape[0]):
                    sr = StackReg(StackReg.BILINEAR)
                    stack_x[p,:,:] = sr.register_transform(prev_stack[p,:,:], stack[p,:,:])
                stack = stack_x.copy()
            else:

                prev_stack=stack.copy()
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
        print(img_name)
        X={}
        X['grid'] = grid_x
        X['y'] =y_x
        X['original_shape']=stack.shape
        X['grid_test']=grid
        return X, batch_size

    def train(self, X, n_steps=1000, batch_size=None, result_path='./'):
        """
        Calls super-class implementation.

        """

        super(Motion_Correction, self).train(X, n_steps, batch_size, result_path)

    def test(self, X):
        """
        Calls super-class implementation.

        """

        return super(Motion_Correction, self).test(X)



