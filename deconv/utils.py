import numpy as np
from skimage import io
import os


def read_psf_file(size_xy, size_z,psf_dir):
    """
    Read PSF from file.

    :param size_xy: Required size in x- and y-direction
    :type size_xy: int
    :param size_z: Required size in z-direction
    :type size_z: int
    :param psf_dir: Directory containing PSF file
    :type psf_dir: str
    :return: PSF
    :rtype: nd.array
    """
    psf_file = 'PSF_' + str(size_xy) + '_' + str(size_z) + '.tif'
    g = np.float32(io.imread(os.path.join(psf_dir, psf_file), plugin='pil'))
    # g/=g.sum()
    # g = g**2
    return g