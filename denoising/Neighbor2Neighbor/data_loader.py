from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np
from skimage import io


class DataLoader_Imagenet_val(Dataset):
    def __init__(self, data_dir, patch=128):
        super(DataLoader_Imagenet_val, self).__init__()
        self.data_dir = data_dir
        self.patch = patch
        self.train_fns = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
        self.train_fns = [data_dir +'/' + f for f in self.train_fns]
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        # print(fn)
        im = io.imread(fn, plugin='tifffile')
        im = np.array(im, dtype=np.float32)

        # random crop
        # D = im.shape[0]
        H = im.shape[0]
        W = im.shape[1]
        if len(im.shape)==2:
            im = im[:,:,np.newaxis]
        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            im = im[xx:xx + self.patch, :, :]
        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            im = im[:,yy:yy + self.patch, :]
        # if D - self.patch > 0:
        #     zz = np.random.randint(0, D - self.patch)
        #     im = im[zz:zz + self.patch, :,:,:]
        # np.ndarray to torch.tensor
        transformer = transforms.Compose([transforms.ToTensor()])
        im = transformer(im)
        return im

    def __len__(self):
        return len(self.train_fns)