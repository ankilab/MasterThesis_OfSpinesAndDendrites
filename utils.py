from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import tifffile as tif
import os
import glob
from pystackreg import StackReg
from skimage import transform, io, exposure
import pandas as pd
from shutil import copy2


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


def save_tif_stack_to_2d(im3d, save_path, start_val = 0, name='f'):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    n_images = im3d.shape[0]
    names = [f'{name}_{i}.tif' for i in range(start_val, start_val+n_images)]
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

        # Weird error: in some folders of TIm's mice there is an additional file name which doe snot correspond to a
        # tif-file e.g ['._117_ArcCre_A1_zoom10_power8_gain500_z1_resgalvo-027-011_Cycle00001_Ch3_000001.ome.tif',
        # '117_ArcCre_A1_zoom10_power8_gain500_z1_resgalvo-027-011_Cycle00001_Ch3_000001.ome.tif']
        if f_name[0][0:2] == '._' and len(f_name)>1:
            fx = f_name[1]
        else:
            fx=f_name[0]

        f = tif.imread(os.path.join(path, i, fx))
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
        path = 'G:/' + row['Researcher'] + '/25x_1NA_Raw/' + row['Mouse'] + '/' + row['Date'] +'/' +row['Region']
        stack = load_stack_from_drive(path, row['Mouse'], required_string='000001.ome')
        reg_stack = register_3d_stack_from_middle(stack, ['affine'])
        save_name = os.path.join(result_folder, row['Researcher'] +'_'+ row['Mouse'] + '_' + row['Date']
                                 +'_' +row['Region'] +'.tif')
        tif.imsave(save_name, reg_stack)


def register_autoquant_images(path, result_folder):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    mice = os.listdir(path)
    for mouse in mice:
        dates = os.listdir(os.path.join(path, mouse))
        date= dates[0]
        # for date in dates:
        regions = os.listdir(os.path.join(path, mouse, date))
        for region in regions:
            stack =np.array([])
            files = [f for f in os.listdir(os.path.join(path, mouse, date, region)) if f.endswith('.tif')]
            files.sort()
            for (idx,fx) in enumerate(files):
                f = tif.imread(os.path.join(path, mouse, date, region, fx))
                if len(f.shape) == 3:
                    f = f[0, :, :]
                f = f[np.newaxis, :, :]
                if idx == 0:
                    stack = f
                else:
                    stack = np.concatenate((stack, f), axis=0)
            # reg_stack = register_3d_stack_from_middle(stack, ['affine'])
            save_name = os.path.join(result_folder, 'Alessandro_' + mouse + '_' + date+ '_' + region+ '.tif')
            tif.imsave(save_name, stack)


def find_all_img_with_min_z(path, min_z, result_folder):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    files = [f for f in os.listdir(path) if f.endswith('.tif')]
    for f in files:
        x=tif.imread(os.path.join(path, f))
        if x.shape[0] >=min_z:
            tif.imsave(os.path.join(result_folder, f), x)


def split_train_test(csv_file, path):
    train_path = os.path.join(path, 'Train')
    test_path = os.path.join(path, 'Test')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    df = pd.read_csv(csv_file, sep = ';')
    reg_files = df[df['Train']=='X']
    for (idx, row) in reg_files.iterrows():
        name = row['Researcher'] +'_' + row['Mouse']+'_'+ row['Date'] +'_' +row['Region'] +'.tif'
        if os.path.isfile(os.path.join(path, name)):
            copy2(os.path.join(path, name), train_path)

    reg_files = df[df['Test']=='X']
    for (idx, row) in reg_files.iterrows():
        name = row['Researcher'] +'_' + row['Mouse']+'_'+ row['Date'] +'_' +row['Region'] +'.tif'
        if os.path.isfile(os.path.join(path, name)):
            copy2(os.path.join(path, name), test_path)


# split_train_test('C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/Train_Test.csv', 'D:/jo77pihe/Registered/Raw_32')
# find_all_img_with_min_z('D:/jo77pihe/Registered/Raw', 32, 'D:/jo77pihe/Registered/Raw_32')
# register_autoquant_images('D:/jo77pihe/Registered/Deconved_AutoQuant', 'D:/jo77pihe/Registered/Deconved_AutoQuant_R2')
files = [f for f in os.listdir('D:/jo77pihe/Registered/Raw_32') if f.endswith('.tif')]

for f in files:
    img = tif.imread(os.path.join('D:/jo77pihe/Registered/Raw_32', f))
    save_tif_stack_to_2d(img,'D:/jo77pihe/Registered/Raw_32/2D', name=f[:-4])

# def predict_img_by_patches(img, predictor, patchsize=(32,128,128)):
#     assert len(patchsize) == 3, 'Please specify a 3-D patch size'
#     (sz, sx, sy) = img.shape
#     z_steps = sz//patchsize[0] + 1 if sz%patchsize[0] !=0 else sz//patchsize[0]
#     x_steps = sx//patchsize[1] + 1 if sx%patchsize[1] !=0 else sx//patchsize[1]
#     y_steps = sy//patchsize[2] + 1 if sy%patchsize[2] !=0 else sy//patchsize[2]
#     z_0, x_0, y_0 = 0,0,0
#
#     for z in range(z_steps):
#         for y in range(y_steps):
#             for x in range(x_steps):
#                 predictor.
#     pass




# register_stacks_from_drive('C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/Train_Test.csv', './Registered')

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


