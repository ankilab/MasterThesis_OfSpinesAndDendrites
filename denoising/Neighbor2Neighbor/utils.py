import torch
import os
import numpy as np
from Neighbor2Neighbor.arch_unet import UNet
from skimage import io
from skimage.metrics import structural_similarity as sk_ssim
import torch.optim as optim

MAX_VAL = 12870
MIN_VAL = -2327
operation_seed_counter = 0

def checkpoint(net, epoch, name, opt, systime):
    """
    Save training checkpoint.

    :param net: PyTorch model
    :param epoch: Epoch number
    :type epoch: int
    :param name: Model name
    :type name: str
    :param opt: Information about where to store checkpoint
    :param systime: System time
    :return: Path where checkpoint ist stored
    """
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))
    return save_model_path


def load_checkpoint(path, lr =3e-4, n_channel=1, n_feature=48):
    """
    Load checkpoint from file.

    :param path: Path to checkpoint.
    :type path: str
    :param lr: Learning rate, defaults to 3e-4
    :type lr: float, optional
    :param n_channel: Number of channels, defaults to 1
    :type n_channel: int, optional
    :param n_feature: Number of features, defaults to 48
    :type n_feature: int, optional
    :return: Pytorch network
    """
    net = UNet(in_nc=n_channel,
               out_nc=n_channel,
               n_feature=n_feature)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint)  # ['model'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    net.eval()
    return net


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0, high=8, size=(n * h // 2 * w // 2, ), generator=get_generator(), out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


def load_val_images(dataset_dir):
    fns = [f for f in os.listdir(dataset_dir) if f.endswith('.tif')]
    fns.sort()
    return fns


def load_img(dataset_dir, name):
    im = io.imread(os.path.join(dataset_dir, name))
    return im


def ssim(prediction, target):
    """
    Calculate SSIM.

    :param prediction: Predicted image
    :type prediction: nd.array
    :param target: Target image
    :type target: nd.array
    :return: SSIM
    """

    return sk_ssim(prediction, target)


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref):
    """
    Calculate SSIM.

    :param target: Predicted image
    :type target: nd.array
    :param ref: Target image
    :type ref: nd.array
    :return: SSIM
    """

    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(MAX_VAL * MAX_VAL / np.mean(np.square(diff)))
    return psnr