import numpy as np

def get_mgrid3(shape, x=1, y=1, z=.1):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    xx, yy, zz = rescale_indices(shape, x, y, z)
    return flatten_meshgrid(shape, xx, yy, zz)


def rescale_indices(shape, x=1, y=1, z=.1):
    tensors = (np.linspace(-z, z, num=shape[0]), np.linspace(-y, y, num=shape[1]), np.linspace(-x, x, num=shape[2]))
    xx, yy, zz = np.meshgrid(*tensors, indexing='ij')
    return xx, yy, zz


def flatten_meshgrid(shape, xx, yy, zz):
    M = np.concatenate((xx[..., None], yy[..., None], zz[..., None]), axis=3)
    return M.reshape(-1, len(shape))