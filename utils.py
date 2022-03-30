import shutil

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tif
import os
import glob
from pystackreg import StackReg
from skimage import exposure
import pandas as pd
from shutil import copy2
from pathlib import Path
import cv2


def display(im3d, cmap="gray", step=1):
    """
    Diplay stack plane-wise.

    :param im3d: Image stack
    :type im3d: nd.array
    :param cmap: Color map, defaults to 'gray'
    :type cmap: str, optional
    :param step: Step in z direction between displayed planes, defaults to 1
    :type step: int, optional
    """
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
    """
    Save 3D-stack as 2D planes/images.

    :param im3d: Image stack
    :type im3d: nd.array
    :param save_path: Path where to save results
    :type save_path: str
    :param start_val: Index of first plane in new file name, defaults to 0
    :type start_val: int, optional
    :param name: file name for acquired 2D planes, defaults to 'f'
    :type name: str, optional
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    n_images = im3d.shape[0]
    names = [f'{name}_{i}.tif' for i in range(start_val, start_val+n_images)]
    p = save_path+'/'
    names = [p + name for name in names]
    for name, image in zip(names,im3d):
        im = Image.fromarray(image)
        im.save(name)


def restore_3d_stack_from_2d_images(source_folder, result_folder):
    """
    Stacks that were previously splitted by save_tif_stack_to_2d() are restored to stacks.

    :param source_folder: Folder with 2D images
    :param result_folder: Folder were restored stackes should be stored in
    """
    files = [f for f in os.listdir(source_folder) if f.endswith('.tif')]
    unique_files = [f for f in files if f.endswith('_0.tif')]
    Path(result_folder).mkdir(parents=True, exist_ok=True)

    for cur_f in unique_files:
        names = [f for f in files if f.split('.tif')[0]==cur_f.split('.tif')[0]]
        # ensure right order (avoid 10 being after 1)
        names.sort(key=lambda x:int(x.split('.tif_')[1].split('.tif')[0]))
        im3d = []
        for (i,name) in enumerate(names):
            img = tif.imread(os.path.join(source_folder, name))
            if i ==0:
                im3d = img[np.newaxis,:,:]
            else:
                im3d = np.append(im3d, img[np.newaxis,:,:], axis=0)
        tif.imsave(os.path.join(result_folder,names[0].split('.tif')[0] +'.tif'),im3d)


def save_reduced_tif_stack(search_path, save_path, start_idx,end_idx):
    """
    Reduce all stacks in one folder to specified number of z planes.

    :param search_path: Folder containing images that are to be reduced
    :type search_path: str
    :param save_path: Folder where reduced stacks are to be stored
    :type save_path: str
    :param start_idx: index of z plane being the first plane of the reduced stack
    :type start_idx: int
    :param end_idx: index of z plane being the last plane of the reduced stack
    :type end_idx: int
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    p = save_path+'/'
    files = glob.glob1(search_path, "*.tif")
    names = [p + name for name in files]
    for (i, f) in enumerate(files):
        im3d = tif.imread(os.path.join(search_path, f))
        tif.imwrite(names[i], im3d[start_idx:end_idx,:,:])


def register_3d_stack(img_stack,  transformations):
    """
    Register image planes of 3D stack.

    :param img_stack: Image stack
    :type img_stack: nd.array
    :param transformations: Allowed transformations
    :type transformations: dict of StackReg Transformations
    :return: Transformed stack, transformation matrix
    :rtype: nd.array, nd.array
    """

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
    Register 3D stack starting from middle plane.
    Assumed input shape: frames (z-axis) x width x height.

    :param img_stack: Image stack
    :type img_stack: nd.array
    :param transformation: Allowed transformations
    :type transformation: dict of StackReg Transformations
    :return: Registered stack
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


# def overlay_images(imgs, equalize=False, aggregator=np.mean):
#     """
#     Overlay images.
#
#     :param imgs: Images
#     :type imgs: list[nd.array]
#     :param equalize: Whether to equalize the histograms, defaults to False
#     :type equalize: bool, optional
#     :param aggregator:
#     :type aggregator: function, optional
#     :return:
#     """
#
#     if equalize:
#         imgs = [exposure.equalize_hist(img) for img in imgs]
#
#     imgs = np.stack(imgs, axis=0)
#
#     return aggregator(imgs, axis=0)


def load_stack_from_drive(path, mouse, required_string='', aq=False):
    """
    Create image stacks from original data which is stored as separate image planes in different folders.

    :param path: Path to data
    :type path: str
    :param mouse: Name of required Mouse
    :type mouse: str
    :param required_string: Mandatory string to filter right image planes, defaults to ''
    :type required_string: str, optional
    :param aq: Flag whether the planes are already deconvolved by AutoQuant, defaults to False
    :type aq: bool, optional
    :return: Stacked image planes
    :rtype: nd.array
    """

    sub_dirs = [x for x in os.listdir(path)]
    img_dirs= [k for k in sub_dirs if mouse[0:4] in k]

    stack = np.array([])
    for (idx, i) in enumerate(img_dirs):
        if aq:
            f = tif.imread(os.path.join(path, i))
            f = f[np.newaxis, :, :]

        else:
            files = [f for f in os.listdir(os.path.join(path,i)) if f.endswith('.tif')]
            f_name = [f for f in files if required_string in f]

            # Weird error: in some folders of TIm's mice there is an additional file name which doe snot correspond to a
            # tif-file e.g ['._117_ArcCre_A1_zoom10_power8_gain500_z1_resgalvo-027-011_Cycle00001_Ch3_000001.ome.tif',
            # '117_ArcCre_A1_zoom10_power8_gain500_z1_resgalvo-027-011_Cycle00001_Ch3_000001.ome.tif']
            if len(f_name) >0:
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


def register_stacks_from_drive(csv_file, result_folder, required_string='000001.ome', save_name_id=''):
    """
    Create and register stacks from original data.

    :param csv_file: Path and name to csv file that specifies, which stacks should be registered(Train_Test.csv)
    :type csv_file: str
    :param result_folder: Path to folder, where to store the results
    :type result_folder: str
    :param required_string: String required to identify the right image planes from the original data,
                            defaults to '000001.ome'
    :type required_string: str, optional
    :param save_name_id: Additional string to be included in result stack name, defaults to ''
    :type save_name_id: str, optional
    """

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    df = pd.read_csv(csv_file, sep = ';')
    reg_files = df[df['Train']=='X']
    for (idx, row) in reg_files.iterrows():
        path = 'G:/' + row['Researcher'] + '/25x_1NA_Raw/' + row['Mouse'] + '/' + row['Date'] +'/' +row['Region']
        stack = load_stack_from_drive(path, row['Mouse'], required_string=required_string)
        if stack.size>0:
            reg_stack = register_3d_stack_from_middle(stack, ['affine'])
            save_name = os.path.join(result_folder, row['Researcher'] +'_'+ row['Mouse'] + '_' + row['Date']
                                     +'_' +row['Region'] +save_name_id+'.tif')
            tif.imwrite(save_name, reg_stack)


def register_all_repetitions(csv_file, result_folder):
    """
    Create and register (all) available repetitions of the same scene from the original data.
    (Required for motion correction)

    :param csv_file: Path and name to csv file that specifies, which stacks should be registered(Train_Test.csv)
    :type csv_file: str
    :param result_folder: Path to folder where to store the results
    :type result_folder: str
    """

    required_strings=['000001.ome','000002.ome','000003.ome','000004.ome','000005.ome','000006.ome','000007.ome']
    save_name_ids= [str(x) for x in np.arange(1,8)]
    for i in range(len(required_strings)):
        register_stacks_from_drive(csv_file, result_folder, required_strings[i], save_name_ids[i])


def register_autoquant_images(path, result_folder):
    """
    Register stacks that where deconvolved with AutoQuant from original data.

    :param path: Path to original data
    :type path: str
    :param result_folder: Path to folder where to store the results
    :type result_folder: str
    """

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
            tif.imwrite(save_name, stack)


def find_all_img_with_min_z(path, min_z, result_folder):
    """
    Iterate over tif-files within a folder and find images that have at least min_z z-planes.

    :param path: Path to data
    :type path: str
    :param min_z: Minimum number of z-planes
    :type min_z: int
    :param result_folder: Folder where to store files that fit the requested amount of z-planes
    :type result_folder: str
    """

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    files = [f for f in os.listdir(path) if f.endswith('.tif')]
    for f in files:
        x=tif.imread(os.path.join(path, f))
        if x.shape[0] >=min_z:
            tif.imwrite(os.path.join(result_folder, f), x)

def create_subsampled_images(path, result_folder, z_shape=10, xy_factor=4):
    """
    Subsample stacks.

    :param path: Path to input files.
    :type path: str
    :param result_folder: Path to folder where t store results
    :type result_folder: str
    :param z_shape: Number of z-planes in subsampled stack
    :type z_shape: int
    :param xy_factor: Factor by which to reduce x and y dimension
    :type xy_factor: int
    """

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    files = [f for f in os.listdir(path) if f.endswith('.tif')]
    for f in files:
        x = tif.imread(os.path.join(path, f))
        subsampled=x[0:z_shape,::xy_factor, ::xy_factor]
        tif.imwrite(os.path.join(result_folder, f), subsampled)


def split_train_test(csv_file, path):
    """
    Split data into training and test set by copying the data in distinct folders.

    :param csv_file: Path and name to csv file that specifies, which stacks should be assigned to training and test set
                    (Train_Test.csv)
    :type csv_file: str
    :param path: Path where to create the folders for training and test set
    :type path: str
    """
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


def get_extrem_val(path):
    """
    Get extrem values of all tif-files within one folder.

    :param path: Path to folder where tif-files are  stored
    :type path: str
    :return: Minimum, Maximum value
    :rtype: float, float
    """

    files = []
    for p in Path(path).rglob('*.tif'):
        files.append(p)
    min_val = 1000
    max_val = 0
    for f in files:
        img = tif.imread(os.path.join(path, f))
        if img.max()>max_val:
            max_val=img.max()
        if img.min() < min_val:
            min_val = img.min()
    print('Max Value: ' + str(max_val))
    print('Min Value: ' + str(min_val))
    return min_val, max_val


def copy_rename_matching_autoquant_images(raw_data, result_folder):
    """
    Copy and rename original AutoQuant stacks.

    :param raw_data: Path to folder where to find the data
    :type raw_data: str
    :param result_folder: Path to folder where to store renamed files
    :type result_folder: str
    """
    files = [f for f in os.listdir(raw_data) if f.endswith('.tif')]
    for f in files:
        parts=f.split('_')
        # TODO: needs to be adjusted as data is copied to new directory now

        path = 'G:/'
        path = os.path.join(path, parts[0])
        deconved_dir = [d for d in os.listdir(path) if '1NA_Deconvolved' in d][0]
        path = os.path.join(path, deconved_dir)
        mouse = parts[1] +'_'+parts[2] if parts[0] !='Ghabiba' else parts[1]
        mouse_dir = [d for d in os.listdir(path) if mouse in d][0]
        path = os.path.join(path, mouse_dir)
        date_dir = [d for d in os.listdir(path) if parts[len(parts)-2] in d]
        if len(date_dir) >=1:
            date_dir = date_dir[0]
            # append data directory and region
            path = os.path.join(path, date_dir)
            x=os.listdir(path)
            if parts[-1][:-4] in x and os.path.isdir(os.path.join(path,parts[-1][:-4])):
                path = os.path.join(path,parts[-1][:-4])
            f_d = [f for f in os.listdir(path) if f ==parts[-1]]
            if len(f_d) >=1:
                shutil.copy2(os.path.join(path,f_d[0]), os.path.join(result_folder,f))
            else:
                print(f)
        else:
            print(f)


def generate_axial_psf(file, save_as=None):
    """
    Basis for creating axial (x-z) image of PSF.

    :param file: Path and filename of PSF
    :type file: str
    :param save_as: File name and path as which to store the axial PSF, if None, not saved, defaults to None
    :type save_as: str or NoneType, optional
    :return: Axial PSF
    :rtype: nd.array
    """
    img =tif.imread(file)
    img = img[:, ::4, ::4]
    y = 63
    n = 512
    psf_ax = img[:,y,:]
    psf_ax=cv2.resize(psf_ax, (n, n))
    if not save_as is None:
        tif.imwrite(save_as, psf_ax)
    return psf_ax

#generate_axial_psf('C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/Blind_RL_Test/PSF_Check/Confocal/131Alessandro_427_ArcCreERT2_Thy1GFP_Ai9_TRAP_2019-08-31_A2_psf.tif',
#                   'C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/Blind_RL_Test/PSF_Check/xz.tif')
#create_subsampled_images('D:/jo77pihe/Registered/20220203_Raw_80', 'D:/jo77pihe/Registered/20220223_Subsampled/10_128', z_shape=10, xy_factor=4)
#find_all_img_with_min_z('D:/jo77pihe/Registered/20220203_Raw', 80, 'D:/jo77pihe/Registered/20220203_Raw_80')
# copy_rename_matching_autoquant_images('D:/jo77pihe/Registered/20220203_Raw', 'D:/jo77pihe/Registered/20220203_AutoQuant_Averaged')

# restore_3d_stack_from_2d_images('D:/jo77pihe/Registered/20220203_Raw_2D/test_test', 'D:/jo77pihe/Registered/20220203_Raw_2D/test_test/res')
#register_all_repetitions('C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/Train_Test.csv', 'D:/jo77pihe/Registered/20220227_Repetitions')
# get_extrem_val('D:/jo77pihe/Registered/Raw')

# split_train_test('C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/Train_Test.csv', 'D:/jo77pihe/Registered/Raw_32')
# find_all_img_with_min_z('D:/jo77pihe/Registered/Raw', 32, 'D:/jo77pihe/Registered/Raw_32')
# register_autoquant_images('D:/jo77pihe/Registered/Deconved_AutoQuant', 'D:/jo77pihe/Registered/Deconved_AutoQuant_R2')
# files = [f for f in os.listdir('D:/jo77pihe/Registered/Raw_32') if f.endswith('.tif')]
#
# for f in files:
#     img = tif.imread(os.path.join('D:/jo77pihe/Registered/Raw_32', f))
#     save_tif_stack_to_2d(img,'D:/jo77pihe/Registered/Raw_32/2D', name=f[:-4])


# register_stacks_from_drive('C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites/Train_Test.csv', './Registered/20220203_Raw')

# register_stacks_from_drive('C:/Users/Johan/Documents/FAU_Masterarbeit/Implementation/Train_Test.csv', './Registered')

# im = io.imread("./Data/GT/001.tif")
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


