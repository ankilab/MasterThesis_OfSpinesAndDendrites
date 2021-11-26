from __future__ import print_function, unicode_literals, absolute_import, division
from .deconvolver import Deconvolver
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
import pickle
import os
import tifffile as tif
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm


class CAREDeconv(Deconvolver):

    def __init__(self, args):

        super().__init__(args)
        self.train_flag = args['train']

    def preprocess(self):
        pass

    def train(self, data_dir, validation_split =0.1, epochs =10, batch_size=8):

        (X, Y), (X_val, Y_val), axes = load_training_data(data_dir,
                                                          validation_split=validation_split, verbose=True)
        train_steps = X.shape[0]//batch_size
        train_steps= train_steps+1 if X.shape[0]%batch_size!=0 else train_steps

        c = axes_dict(axes)['C']
        n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

        config = Config(axes, n_channel_in, n_channel_out, train_batch_size=batch_size,
                        train_steps_per_epoch=train_steps, train_epochs=epochs)
        model_dir = os.path.join(self.res_path,'models')
        model = CARE(config, 'my_model', basedir=model_dir)
        print(model.keras_model.summary())

        history = model.train(X, Y, validation_data=(X_val, Y_val))

        # Save training history
        history.history['lr'] = [float(f) for f in history.history['lr']]
        res_dict = dict(history.history, **history.params)
        # r = os.path.join(self.res_path,'result.json')
        # json.dump(res_dict, open(r, 'w'))

        with open(os.path.join(self.res_path, f'results_care.pkl'), 'wb') \
                as outfile:
            pickle.dump(res_dict, outfile, pickle.HIGHEST_PROTOCOL)

        #Save model
        mdl_path = os.path.join(self.res_path, 'TF_SavedModel.zip')
        model.export_TF(mdl_path)

        # Plot training metrics
        plt.figure(figsize=(16, 5))
        plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mae', 'val_mae'])
        with open(os.path.join(self.res_path, 'history_care.pkl'), 'wb') as outfile:
            pickle.dump(history, outfile, pickle.HIGHEST_PROTOCOL)

        # # Plot exemplary results
        # plt.figure(figsize=(20, 12))
        # _P = model.keras_model.predict(X_val[-5:])
        # if config.probabilistic:
        #     _P = _P[..., :(_P.shape[-1] // 2)]
        # plot_some(X_val[-5:], Y_val[-5:], _P, pmax=99.5)
        # plt.suptitle('5 example validation patches\n'
        #              'top row: input (source),  '
        #              'middle row: target (ground truth),  '
        #              'bottom row: predicted from source')
        # plt.show()
        return model_dir, mdl_path

    def predict(self, data_dir, model_dir, name, save_res=True):
        p = os.path.join(self.res_path, 'Predictions')
        if not os.path.exists(p):
            os.makedirs(p)

        files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
        for f in files:
            X = io.imread(os.path.join(data_dir,f))
            self.predict_img(X, model_dir, name, f)

    def predict_img(self, X, model_dir, name, save_as=None):
        axes = 'ZYX'
        model = CARE(config=None, name=name, basedir=model_dir)
        restored = model.predict(X, axes)

        if save_as is not None:
            tif.imsave(save_as, restored)
        return restored

