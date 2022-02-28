from __future__ import print_function, unicode_literals, absolute_import, division

# import keras.backend
import multiprocessing
from .deconvolver import Deconvolver
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
import pickle
import os
import tifffile as tif
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf
import gc


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# config = tf.compat.v1.ConfigProto(
#     inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(config=config)
#
# print(sess._config)
# tf.compat.v1.keras.backend.set_session(sess)
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
#from numba import cuda


class CAREDeconv(Deconvolver):

    def __init__(self, args):

        super().__init__(args)

    def preprocess(self):
        pass

    def train(self, data_dir, validation_split =0.1, epochs =10, batch_size=8, learning_rate = 0.0004, unet_residual=True,
              unet_n_depth=2):

        (X, Y), (X_val, Y_val), axes = load_training_data(data_dir,
                                                          validation_split=validation_split, verbose=True)
        train_steps = X.shape[0]//batch_size
        train_steps= train_steps+1 if X.shape[0]%batch_size!=0 else train_steps

        c = axes_dict(axes)['C']
        n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

        config = Config(axes, n_channel_in, n_channel_out, train_batch_size=batch_size,
                        train_steps_per_epoch=train_steps, train_epochs=epochs, train_learning_rate=learning_rate,
                        unet_residual=unet_residual, unet_n_depth=unet_n_depth)
        model_dir = os.path.join(self.res_path,'models')

        model = CARE(config, 'my_model', basedir=model_dir)
        model_dir = os.path.join(model_dir, 'my_model')
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
            pickle.dump(history.history, outfile, pickle.HIGHEST_PROTOCOL)

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

    def predict(self, data_dir, model_dir, name, save_res=True, res_folder='./'):
        p = os.path.join(self.res_path, 'Predictions')
        if not os.path.exists(p):
            os.makedirs(p)

        files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
        for f in files:
            X = io.imread(os.path.join(data_dir,f))
            self.predict_img(X, model_dir, name, os.path.join(res_folder,f))
            p = multiprocessing.Process(target=self.predict_img, args=(X, model_dir, name, os.path.join(res_folder,f)),)
            p.start()
            p.join()
            del X
            # cuda.get_current_device().reset()
            # cuda.close()
            # # cuda.select_device(0)
            # # cuda.close()
            tf.keras.backend.clear_session()
            # keras.backend.clear_session()
            # tf.compat.v1.reset_default_graph()
            # sess = tf.compat.v1.keras.backend.get_session()
            # # tf.keras.backend.
            # sess.close()

            gc.collect()

    def predict_img(self, X, model_dir, name, save_as=None):
        axes = 'ZYX'
        # X = tf.convert_to_tensor(X)
        # if self.model is None:
        model = CARE(config=None, name=name, basedir=model_dir)
        restored = model.predict(X, axes)

        if save_as is not None:
            tif.imsave(save_as, restored)
            print(save_as)
        del model
        # device = cuda.get_current_device()
        # device.reset()
        # cuda.close()
        gc.collect()
        return restored

