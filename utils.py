from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import tifffile as tif
import os
import glob
from pystackreg import StackReg
import pystackreg
from skimage import transform, io, exposure
import pandas as pd

def display(im3d, cmap="gray", step=1):
    n_col = 5
    n_row = np.ceil(im3d.shape[0]/n_col).astype(int)
    _, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(16, 14))

    vmin = im3d.min()
    vmax = im3d.max()

    for ax, image in zip(axes.flatten(), im3d[::step]):
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def save_tif_stack_to_2d(im3d, save_path, start_val = 0):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    n_images = im3d.shape[0]
    names = [f'f_{i}.tif' for i in range(start_val, start_val+n_images)]
    p = save_path+'/'
    names = [p + name for name in names]
    for name, image in zip(names,im3d):
        im = Image.fromarray(image)
        im.save(name)


def save_reduced_tif_stack(search_path, save_path, start_idx,end_idx):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    p = save_path+'/'
    files = glob.glob1(search_path, "*.tif")
    names = [p + name for name in files]
    for (i, f) in enumerate(files):
        im3d = tif.imread(os.path.join(search_path, f))
        tif.imsave(names[i], im3d[start_idx:end_idx,:,:])


def register_3d_stack(img_stack,  transformations):
    tmats = []

    for i, (name, tf) in enumerate(transformations.items()):
        sr = StackReg(tf)
        reference = 'first' if name == 'BILINEAR' else 'previous'
        tmat = sr.register_stack(img_stack, axis=0, reference=reference, verbose=True)
        tmats.append(tmat)
        img_stack = sr.transform_stack(img_stack)

    return img_stack, tmats


def register_3d_stack_from_middle(img_stack, transformation:list):
    """
    Assumed input shape: frames (z-axis) x width x height
    :param transformations:
    :param img_stack:
    :return:
    """
    t_upper = [t.upper() for t in transformation]
    transformations = {
        'TRANSLATION': StackReg.TRANSLATION,
        'RIGID_BODY': StackReg.RIGID_BODY,
        'SCALED_ROTATION': StackReg.SCALED_ROTATION,
        'AFFINE': StackReg.AFFINE,
        'BILINEAR': StackReg.BILINEAR
    }
    filtered_trans = {t: transformations[t] for t in t_upper}

    assert len(filtered_trans.keys())>=1, 'No transformation type specified'

    ref_idx = int(np.ceil(img_stack.shape[0]/2)-1)
    top_stack, _ = register_3d_stack(img_stack[ref_idx:,:,:], filtered_trans)
    bottom_stack, _ = register_3d_stack(np.flip(img_stack[:ref_idx +1,:,:], axis=0), filtered_trans)
    reg_stack = np.concatenate((np.flip(bottom_stack[1:,:,:], axis=0), top_stack), axis=0)

    return reg_stack


def overlay_images(imgs, equalize=False, aggregator=np.mean):

    if equalize:
        imgs = [exposure.equalize_hist(img) for img in imgs]

    imgs = np.stack(imgs, axis=0)

    return aggregator(imgs, axis=0)

def load_stack_from_drive(path, mouse, required_string=''):
    sub_dirs = [x for x in os.listdir(path)]
    img_dirs= [k for k in sub_dirs if mouse[0:4] in k]

    stack = np.array([])
    for (idx, i) in enumerate(img_dirs):
        files = [f for f in os.listdir(os.path.join(path,i)) if f.endswith('.tif')]
        f_name = [f for f in files if required_string in f]
        f = tif.imread(os.path.join(path, i, f_name[0]))
        if len(f.shape) ==3:
            f = f[0,:,:]
        f = f[np.newaxis,:,:]
        if idx == 0:
            stack = f
        else:
            stack = np.concatenate((stack, f), axis = 0)

    return stack


def register_stacks_from_drive(csv_file, result_folder):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    df = pd.read_csv(csv_file, sep = ';')
    reg_files = df[df['Train']=='X']
    for (idx, row) in reg_files.iterrows():
        path = 'D:/' + row['Researcher'] + '/25x_1NA_Raw/' + row['Mouse'] + '/' + row['Date'] +'/' +row['Region']
        stack = load_stack_from_drive(path, row['Mouse'], required_string='000001.ome')
        reg_stack = register_3d_stack_from_middle(stack, ['affine'])
        save_name = os.path.join(result_folder, row['Researcher'] +'_'+ row['Mouse'] + '_' + row['Date']
                                 +'_' +row['Region'] +'.tif')
        tif.imsave(save_name, reg_stack)




# register_stacks_from_drive('C:/Users/Johan/Documents/FAU_Masterarbeit/Implementation/Train_Test.csv', './Registered')

# im = io.imread("./Data/GT/001.tif")
# print(im.shape)
# display(im)
# reg_img = register_3d_stack_from_middle(im, ['affine'])
# tif.imsave('./Data/registered_img.tif', reg_img)
#
# # save_tif_stack_to_2d(im,'./Data/GT_2D', 0)
# # save_reduced_tif_stack("./Data/GT/",'./Data/GT_adjusted', 1,im.shape[0])
#
# im = io.imread("./Data/Raw/001.tif")
# print(im.shape)
# # save_reduced_tif_stack("./Data/Raw/",'./Data/Raw_adjusted', 1,im.shape[0])
# # save_tif_stack_to_2d(im,'./Data/Raw_adjusted', 0)


