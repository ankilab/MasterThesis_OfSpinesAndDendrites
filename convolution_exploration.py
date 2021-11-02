# Source: https://numbersmithy.com/2d-and-3d-convolutions-using-numpy/
import numpy as np
from scipy.signal import fftconvolve, convolve
from skimage import io
import os
import timeit
from scipy.signal import fftconvolve, convolve
import tifffile as tif
from cupyx.scipy import ndimage
from cupyx.scipy import signal
import cupy


def checkShape(var, kernel):
    '''Check shapes for convolution
    Args:
        var (ndarray): 2d or 3d input array for convolution.
        kernel (ndarray): 2d or 3d convolution kernel.
    Returns:
        kernel (ndarray): 2d kernel reshape into 3d if needed.
    '''
    var_ndim = np.ndim(var)
    kernel_ndim = np.ndim(kernel)
    if var_ndim not in [2, 3]:
        raise Exception("<var> dimension should be in 2 or 3.")
    if kernel_ndim not in [2, 3]:
        raise Exception("<kernel> dimension should be in 2 or 3.")
    if var_ndim < kernel_ndim:
        raise Exception("<kernel> dimension > <var>.")
    if var_ndim == 3 and kernel_ndim == 2:
        kernel = np.repeat(kernel[:, :, None], var.shape[2], axis=2)
    return kernel


def padArray(var, pad1, pad2=None):
    '''Pad array with 0s
    Args:
        var (ndarray): 2d or 3d ndarray. Padding is done on the first 2 dimensions.
        pad1 (int): number of columns/rows to pad at left/top edges.
    Keyword Args:
        pad2 (int): number of columns/rows to pad at right/bottom edges.
            If None, same as <pad1>.
    Returns:
        var_pad (ndarray): 2d or 3d ndarray with 0s padded along the first 2
            dimensions.
    '''
    if pad2 is None:
        pad2 = pad1
    if pad1+pad2 == 0:
        return var
    var_pad = np.zeros(tuple(pad1+pad2+np.array(var.shape[:2])) + var.shape[2:])
    var_pad[pad1:-pad2, pad1:-pad2] = var
    return var_pad


def asStride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.
    Args:
        arr (ndarray): input array of rank 2 or 3, with shape (m1, n1) or (m1, n1, c).
        sub_shape (tuple): window size: (m2, n2).
        stride (int): stride of windows in both y- and x- dimensions.
    Returns:
        subs (view): strided window view.
    See also skimage.util.shape.view_as_windows()
    '''
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape[:2]
    view_shape = (1+(m1-m2)//stride, 1+(n1-n2)//stride, m2, n2)+arr.shape[2:]
    strides = (stride*s0, stride*s1, s0, s1)+arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)
    return subs

def conv3D3(var, kernel, stride=1, pad=0):
    '''3D convolution by strided view.
    Args:
        var (ndarray): 2d or 3d array to convolve along the first 2 dimensions.
        kernel (ndarray): 2d or 3d kernel to convolve. If <var> is 3d and <kernel>
            is 2d, create a dummy dimension to be the 3rd dimension in kernel.
    Keyword Args:
        stride (int): stride along the 1st 2 dimensions. Default to 1.
        pad (int): number of columns/rows to pad at edges.
    Returns:
        conv (ndarray): convolution result.
    '''
    kernel = checkShape(var, kernel)
    if pad > 0:
        var_pad = padArray(var, pad, pad)
    else:
        var_pad = var
    view = asStride(var_pad, kernel.shape, stride)
    if np.ndim(kernel) == 2:
        conv = np.sum(view*kernel, axis=(2, 3))
    else:
        conv = np.sum(view*kernel, axis=(2, 3, 4))
    return conv

def _get_psf(size_xy, size_z):
    file_path = 'C:\\Users\\Johan\\Documents\\FAU_Masterarbeit\\MasterThesis_OfSpinesAndDendrites\\Data\\PSF'
    z = size_z

    psf_file = 'PSF_' + str(size_xy) + '_' + str(z) + '.tif'

    # Initial guess for PSF
    g = io.imread(os.path.join(file_path,psf_file), plugin='pil')
    offset = int((z - size_z) / 2)
    return g[offset:g.shape[0] - offset, :, :]

def fft(array):
  fft = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(array)))
  return fft

def ifft(array):
  ifft = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(array)))
  return ifft

def conv_3D(array, kernel):
  conv = np.abs(ifft(fft(array)*fft(kernel)))
  return conv


if __name__ == "__main__":
    X = io.imread(os.path.join('C:\\Users\\Johan\\Documents\\FAU_Masterarbeit\\MasterThesis_OfSpinesAndDendrites\\Registered', 'Alessandro_427_ArcCreERT2_Thy1GFP_Ai9_TRAP_2019-08-31_A2.tif'))
    X = np.float32(X)
    X +=X.min()
    X /= X.max()
    z_pad_s = X.shape[0]//2
    y_pad_s = X.shape[2]//2
    x_pad_s = X.shape[1] // 2
    X_pad= np.pad(X, ((z_pad_s, z_pad_s), (x_pad_s, x_pad_s), (y_pad_s, y_pad_s)), 'constant', constant_values =((0,0),(0,0),(0,0)))
    # psf_pad= np.pad(psf, ((z_pad_s, z_pad_s), (x_pad_s, x_pad_s), (y_pad_s, y_pad_s)), 'constant', constant_values =(0,0,0))
    psf = _get_psf(X.shape[1], X.shape[0])

    # # Variante 5:
    # start = timeit.default_timer()
    #
    # conv_5 = signal.fftconvolve(cupy.array(X), cupy.array(psf), mode='same')
    # stop = timeit.default_timer()
    # print( 'Option 5 (cupyx.scipy.fftconvolve) took ' + str(stop-start) +' s')



    # # Variante 1:
    # start = timeit.default_timer()
    # conv = conv_3D(X, psf)
    # stop = timeit.default_timer()
    # print( 'Option 1 took ' + str(stop-start) +' s')
    # # conv3D3(stride=1, pad =)

    # Variante 2:
    start = timeit.default_timer()
    conv_2 = convolve(X, psf, 'same')
    stop = timeit.default_timer()
    print( 'Option 2 (scipy.signal.convolve) took ' + str(stop-start) +' s')

    # Variante 3:
    start = timeit.default_timer()
    conv_3 = fftconvolve(X, psf, 'same')
    stop = timeit.default_timer()
    print( 'Option 3 (scipy.signal.fftconvolve) took ' + str(stop-start) +' s')

    # xt = cupy.isclose(cupy.array(conv_2), conv_4).astype(int)
    # print(cupy.min(xt))
    # n_a = cupy.asnumpy(conv_4)

    # tif.imsave('res_cupy.tif', n_a)

    # Variante 4:
    start = timeit.default_timer()
    xt = cupy.array(X)
    ndimage.convolve(xt, cupy.array(psf), xt, mode='constant')
    stop = timeit.default_timer()
    print( 'Option 4 (cupyx.scipy.ndimage.convolve) took ' + str(stop-start) +' s')
