import random
import numpy as np
import os
from csbdeep.data import RawData, create_patches
import tifffile as tif
import flammkuchen as fl


MAX_VAL = 12870
MIN_VAL = -2327


class DataAugmenter():
    """
    Creates patches of data and saves it to file.

    """

    def __init__(self, args):
        """
        Initialize object for data augmentation.

        :param args: 'z_shape' - required patch size in z direction, 'xy_shape - required patch size in x and y direction,
                        'n_patches' - number of patches per images
        :type args: dict(int, int, int)
        """

        self.z_shape = args['z_shape']
        self.xy_shape = args['xy_shape']
        self.n_patches = args['n_patches']

    def augment(self, data_path, source, target, care=True, data_dir=None, save_file_name='my_data.npz', save_h5=None):
        """
        Create patches from images.

        :param data_path: Data path to image pairs
        :type data_path: str
        :param source: Relative path from data path to input data (e.g. distorted data)
        :type source: str
        :param target: Relative path from data path to output data (e.g. undistorted data)
        :type target: str
        :param care: Whether to use the data augmentation provided by CSBDeep package - no other option currently
                     implemented, defaults to True
        :type care: bool, optional
        :param data_dir: Directory where to store npz-file- if None it is not stored as npz-file, defaults to None
        :type data_dir: str or NoneType, optional
        :param save_file_name: File name to store npz-file as, defaults to 'my_data.npz'
        :type save_file_name: str, optional
        :param save_h5: Path and filename to store h5 file as - if None it is not stored as h5-file, defaults to None
        :type save_h5: str or NoneType, optional
        :return: Pairs of augmented distorted and undistorted image patches, data_dir
        :rtype: tuple(augmented distorted data, augmented undistorted data, axes), str
        """

        # If data_dir is None, result is not saved to file
        if data_dir is not None:
            data_dir = os.path.join(data_dir, save_file_name)

        # augmented_raw, augmented_gt, axes = [], [], []
        # if not care:
        #     source_files = [f for f in os.listdir(os.path.join(data_path, source)) if f.endswith('.tif')]
        #     target_files = [f for f in os.listdir(os.path.join(data_path, target)) if f.endswith('.tif')]
        #     for j in range(len(source_files)):
        #         raw = tif.imread(os.path.join(data_path, source, source_files[j]))
        #         try:
        #             target = tif.imread(os.path.join(data_path, target, target_files[j]))
        #             augmented_raw, augmented_gt, axes = self.augment_img(raw, target)
        #             # TODO: stack
        #
        #         except:
        #             print("No corresponding target file found. Image skipped.")
        #
        # else:
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
        if save_h5 is not None:
            augmented_raw = ((augmented_raw-MIN_VAL)/MAX_VAL)*2
            augmented_gt = ((augmented_gt-MIN_VAL)/MAX_VAL)*2
            augmented_raw -= 1.0
            augmented_gt -= 1.0
            augmented_raw = np.clip(augmented_raw, -1, 1)
            augmented_gt = np.clip(augmented_gt, -1, 1)

            fl.save(save_h5, {'raw': np.float32(augmented_raw[:, 0, :, :, :]),
                              'gt': np.float32(augmented_gt[:, 0, :, :, :])})

        return (augmented_raw, augmented_gt, axes), data_dir

    def augment_img(self, raw_img, gt_img):
        """
        Augment single image using own implementation.

        :param raw_img: Raw image
        :type raw_img: nd.array
        :param gt_img: Ground truth image corresponding to raw image
        :type gt_img: nd.array
        :return: Pairs of image patches, axes (here to be consistent with required input for CARE)
        :rtype: nd.array, nd.array, str
        """
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
    """
    Class providing functionality to handle training data. A number of patches can be requested and this class enables
    loading them from a h5 file. (Used for Mu-Net)
    """

    def __init__(self, size, data_path, source, target, n_patches=50, data_file='data.h5'):
        """
        Initialize object.

        :param size: path size in z direction, patch size in x and y direction
        :type size: tuple(int, int)
        :param data_path: Data path to image pairs
        :type data_path: str
        :param source: Relative path from data path to input data (e.g. distorted data)
        :type source: str
        :param target: Relative path from data path to output data (e.g. undistorted data)
        :type target: str
        :param n_patches: number of patches per images, defaults to 50
        :type n_patches: int, optional
        :param data_file: Path and filename to store augmented data as
        :type save_h5: str, optional
        """

        (sz_z, sz_xy) = size
        args = {}
        args['z_shape'] = sz_z
        args['xy_shape'] = sz_xy
        args['n_patches'] = n_patches
        da = DataAugmenter(args)

        # If no existing file name is specified generate new data
        if not os.path.isfile(data_file):
            self.data_h5 = data_file
            da.augment(data_path, source, target, care=True, data_dir=None,
                       save_h5=self.data_h5)
        else:
            self.data_h5 = data_file

        self._reset_idx()
        self.size = fl.meta(self.data_h5, "raw").shape
        self.draw_order = np.arange(self.size[0])
        # self.seed = 0

    def _shuffle(self):
        """
        Shuffle data.

        """

        self.draw_order = np.random.choice(np.arange(self.size[0]), self.size[0], replace=False)
        self._reset_idx()

    def _reset_idx(self):
        """
        Reset index - pointer to start position from where to get next data.
        """
        self.idx = 0

    def get(self, batch_size):
        """
        Get data of specified batch size from file.

        :param batch_size: Batch size
        :type batch_size: int
        :return: Batch - image pairs of raw data and corresponding ground truth
        :rtype: nd.array, nd.array
        """
        end_idx = self.idx + batch_size
        end_idx = end_idx if end_idx <= self.size[0] else self.size[0]
        raw = fl.load(self.data_h5, "/raw", sel=fl.aslice[self.draw_order[self.idx:end_idx], :, :, :])
        gt = fl.load(self.data_h5, "/gt", sel=fl.aslice[self.draw_order[self.idx:end_idx], :, :, :])

        # raw = self.X[self.idx:end_idx,:,:,:]
        # gt = self.Y[self.idx:end_idx,:,:,:]
        self.idx += batch_size
        return raw, gt


class DataProvider_CompleteImg():
    """
    Class providing functionality to handle training data of complete images. A number of images can be requested and t
    his class enables loading them from a h5 file. (Used for Mu-Net)
    """
    def __init__(self, data_path, source, target):
        """
        Initialize object.

        :param data_path: Data path to image pairs
        :type data_path: str
        :param source: Relative path from data path to input data (e.g. distorted data)
        :type source: str
        :param target: Relative path from data path to output data (e.g. undistorted data)
        :type target: str
        """
        self._reset_idx()
        self.files =[f for f in os.listdir(os.path.join(data_path, source)) if f.endswith('.tif')]
        self.draw_order = np.arange(len(self.files))
        self.data_path = data_path
        self.source = source
        self.target = target
        self.size = [len(self.files), 1]
        # self.seed = 0

    def _shuffle(self):
        """
        Shuffle data.

        """
        self.draw_order = np.random.choice(np.arange(len(self.files)), len(self.files), replace=False)
        self._reset_idx()

    def _reset_idx(self):
        """
        Reset index - pointer to start position from where to get next data.
        """

        self.idx = 0

    def get(self, batch_size):
        """
        Get data of specified batch size from file.

        :param batch_size: Batch size
        :type batch_size: int
        :return: Batch - image pairs of raw data and corresponding ground truth
        :rtype: nd.array, nd.array
        """

        end_idx = self.idx + batch_size
        end_idx = end_idx if end_idx <= len(self.files) else len(self.files)
        raw = []
        gt= []
        for x in range(self.idx, end_idx):
            idx = self.draw_order[x]
            if x == self.idx:
                raw=tif.imread(os.path.join(self.data_path,self.source, self.files[idx]))[np.newaxis,...]
                gt = tif.imread(os.path.join(self.data_path,self.target, self.files[idx]))[np.newaxis,...]
            else:
                img=tif.imread(os.path.join(self.data_path,self.source, self.files[idx]))[np.newaxis,...]
                raw=np.concatenate((raw, img), axis=0)
                img = tif.imread(os.path.join(self.data_path,self.target, self.files[idx]))[np.newaxis,...]
                gt=np.concatenate((gt, img), axis=0)

        self.idx += batch_size
        return raw, gt
