import random
import numpy as np
import os
from csbdeep.data import RawData, create_patches
import tifffile as tif
from random import shuffle
import flammkuchen as fl
from csbdeep.io import load_training_data
import gc


class DataAugmenter():
    def __init__(self, args):
        self.z_shape = args['z_shape']
        self.xy_shape = args['xy_shape']
        self.n_patches = args['n_patches']

    def augment(self, data_path, source, target, care=False, data_dir=None, save_file_name='my_data.npz', save_h5=None):
        """

        :param data_path:
        :param source:
        :param target:
        :param care:
        :param data_dir:
        :return:
        """
        # If data_dir is None, result is not saved to file
        if data_dir is not None:
            data_dir = os.path.join(data_dir, save_file_name)

        augmented_raw, augmented_gt, axes = [], [], []
        if not care:
            source_files = [f for f in os.listdir(os.path.join(data_path, source)) if f.endswith('.tif')]
            target_files = [f for f in os.listdir(os.path.join(data_path, target)) if f.endswith('.tif')]
            for j in range(len(source_files)):
                raw = tif.imread(os.path.join(data_path, source, source_files[j]))
                try:
                    target = tif.imread(os.path.join(data_path, target, target_files[j]))
                    augmented_raw, augmented_gt, axes = self.augment_img(raw, target)
                    # TODO: stack

                except:
                    print("No corresponding target file found. Image skipped.")

        else:
            raw_data = RawData.from_folder(
                basepath=data_path,
                source_dirs=[source],
                target_dir=target,
                axes='ZYX',
            )

            # Parameter: patch_filter (optional) – Function to determine for each image pair which patches are eligible
            # to be extracted (default: no_background_patches()). Set to None to disable
            augmented_raw, augmented_gt, axes = create_patches(
                raw_data=raw_data,
                patch_size=(self.z_shape, self.xy_shape, self.xy_shape),
                n_patches_per_image=self.n_patches,
                save_file=data_dir)
            if save_h5 != '':
                max_d=np.maximum(np.max(augmented_raw), np.max(augmented_gt))
                min_d=np.minimum(np.min(augmented_raw), np.min(augmented_gt))
                augmented_raw = (augmented_raw-min_d)/max_d
                augmented_gt = (augmented_gt-min_d)/max_d
                fl.save(save_h5, {'raw': np.float32(augmented_raw[:, 0, :, :, :]),
                                  'gt': np.float32(augmented_gt[:, 0, :, :, :])})

        return (augmented_raw, augmented_gt, axes), data_dir

    def augment_img(self, raw_img, gt_img):
        assert len(
            raw_img.shape) == 3, '3-D image required, if using 2-D images add dimension e.g. img[numpy.newaxis,:,:]'
        (z, x, y) = raw_img.shape
        augmented_raw = np.empty((self.n_patches,) + tuple((self.z_shape, self.xy_shape, self.xy_shape)),
                                 dtype=np.float32)
        augmented_gt = np.empty_like(augmented_raw)

        for i in range(self.n_patches):
            x_start = random.randint(0, x - self.xy_shape)
            y_start = random.randint(0, y - self.xy_shape)
            z_start = random.randint(0, z - self.z_shape)

            augmented_raw[i, :, :, :] = raw_img[z_start:(z_start + self.z_shape), x_start:(x_start + self.xy_shape),
                                        y_start:(y_start + self.xy_shape)]
            augmented_gt[i, :, :, :] = gt_img[z_start:(z_start + self.z_shape), x_start:(x_start + self.xy_shape),
                                       y_start:(y_start + self.xy_shape)]

        return augmented_raw, augmented_gt, 'SCZYX'


class DataProvider():
    def __init__(self, size, data_path='', source='', target='', n_patches=50, data_file='data.h5'):
        (sz_z, sz_xy) = size
        args = {}
        args['z_shape'] = sz_z
        args['xy_shape'] = sz_xy
        args['n_patches'] = n_patches
        da = DataAugmenter(args)

        # If no file name is specified generate new data
        if not os.path.isfile(data_file):
            self.data_h5 = data_file
            da.augment(data_path, source, target, care=True, data_dir=None,
                       save_h5=self.data_h5)
        else:
            self.data_h5 = data_file

        # (self.X, self.Y), (self.X_val, self.Y_val), axes = load_training_data(data_dir,
        #                                                   validation_split=0.1, verbose=True)
        # self.X = np.float32(self.X[:,0,:,:,:])
        # self.Y = np.float32(self.Y[:, 0, :, :, :])
        # self.X_val = np.float32(self.X_val[:,0,:,:,:])
        # self.Y_val = np.float32(self.Y_val[:, 0, :, :, :])

        self._reset_idx()
        self.size = fl.meta(self.data_h5, "raw").shape
        self.draw_order = np.arange(self.size[0])
        # self.seed = 0

    def shuffle(self):
        self.draw_order = np.random.choice(np.arange(self.size[0]), self.size[0], replace=False)
        self._reset_idx()

    def _reset_idx(self):
        self.idx = 0

    def get(self, batch_size):
        end_idx = self.idx + batch_size
        end_idx = end_idx if end_idx <= self.size[0] else self.size[0]
        raw = fl.load(self.data_h5, "/raw", sel=fl.aslice[self.draw_order[self.idx:end_idx], :, :, :])
        gt = fl.load(self.data_h5, "/gt", sel=fl.aslice[self.draw_order[self.idx:end_idx], :, :, :])

        # raw = self.X[self.idx:end_idx,:,:,:]
        # gt = self.Y[self.idx:end_idx,:,:,:]
        self.idx += batch_size
        return raw, gt
