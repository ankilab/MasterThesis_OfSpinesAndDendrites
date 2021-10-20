import random
import numpy as np
import os
from csbdeep.data import RawData, create_patches
import tifffile as tif
from random import shuffle

class DataAugmenter():
    def __init__(self, args):
        self.z_shape = args['z_shape']
        self.xy_shape = args['xy_shape']
        self.n_patches = args['n_patches']

    def augment(self, data_path, source, target, care=False, data_dir = None):
        # If data_dir is None, result is not saved to file
        if data_dir is not None:
            data_dir = os.path.join(data_dir,'my_data.npz')

        augmented_raw, augmented_gt, axes = [],[],[]
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

            augmented_raw, augmented_gt, axes = create_patches(
                raw_data=raw_data,
                patch_size=(self.z_shape, self.xy_shape, self.xy_shape),
                n_patches_per_image=self.n_patches,
                save_file=data_dir)

        return (augmented_raw, augmented_gt, axes), data_dir

    def augment_img(self, raw_img, gt_img):
        assert len(raw_img.shape) == 3, '3-D image required, if using 2-D images add dimension e.g. img[numpy.newaxis,:,:]'
        (z,x,y) = raw_img.shape
        augmented_raw = np.empty((self.n_patches,) + tuple((self.z_shape, self.xy_shape, self.xy_shape)), dtype=np.float32)
        augmented_gt = np.empty_like(augmented_raw)

        for i in range(self.n_patches):
            x_start = random.randint(0, x-self.xy_shape)
            y_start = random.randint(0, y - self.xy_shape)
            z_start = random.randint(0, z-self.z_shape)

            augmented_raw[i,:,:,:]= raw_img[z_start:(z_start+self.z_shape),x_start:(x_start+self.xy_shape),
                             y_start:(y_start+self.xy_shape)]
            augmented_gt[i,:,:,:]=gt_img[z_start:(z_start + self.z_shape), x_start:(x_start + self.xy_shape),
                                 y_start:(y_start + self.xy_shape)]

        return augmented_raw, augmented_gt, 'SCZYX'


class DataProvider():
    def __init__(self, size, data_dir):
        (sz_z, sz_xy) = size
        args = {}
        args['z_shape'] = sz_z
        args['xy_shape'] = sz_xy
        args['n_patches'] = 50
        da = DataAugmenter(args)
        (augmented_raw, augmented_gt, _), _ = da.augment(data_dir, 'Raw', 'GT', care=True)
        self.augmented_raw = augmented_raw[:,0,:,:,:]
        self.augmented_gt = augmented_gt[:, 0, :, :, :]
        self.idx = 0
        # self.seed = 0

    def shuffle(self):
        ind_list = [i for i in range(self.augmented_raw.shape[0])]
        shuffle(ind_list)
        self.augmented_raw = self.augmented_raw[ind_list, :, :, :]
        self.augmented_gt = self.augmented_gt[ind_list,:,:,:]

    def get(self, batch_size):
        end_idx = self.idx+batch_size
        raw = self.augmented_raw[self.idx:end_idx,:,:,:]
        gt = self.augmented_gt[self.idx:end_idx,:,:,:]
        self.idx += batch_size
        return raw, gt










