import numpy as np
from skimage import io
import os


def read_psf_file(size_xy, z,psf_dir):
    psf_file = 'PSF_' + str(size_xy) + '_' + str(z) + '.tif'
    g = np.float32(io.imread(os.path.join(psf_dir, psf_file), plugin='pil'))
    # g/=g.sum()
    # g = g**2
    return g