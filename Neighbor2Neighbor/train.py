from __future__ import division

import multiprocessing
import os
import time
import glob
import datetime
import argparse
import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as sk_ssim
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tifffile as tif
from arch_unet import UNet

parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default="gauss25")
parser.add_argument('--data_dir', type=str, default='./Imagenet_val')
parser.add_argument('--val_dir', type=str, default='./validation')
parser.add_argument('--res_dir', type=str, default='./N2N_result')
parser.add_argument('--save_model_path', type=str, default='.\\results')
parser.add_argument('--log_name', type=str, default='unet_gauss25_b4e100r02')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=1)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n_epoch', type=int, default=30)
parser.add_argument('--n_snapshot', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--patchsize', type=int, default=256)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=1.0)
parser.add_argument("--increase_ratio", type=float, default=2.0)

MAX_VAL = 8192
opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices


def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


class AugmentNoise(object):
    def __init__(self, style):
        print(style)
        if style.startswith('gauss'):
            self.params = [
                float(p) / MAX_VAL for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0,
                         std=std,
                         generator=get_generator(),
                         out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_std - min_std) + min_std
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised

    def add_valid_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)


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
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
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


class DataLoader_Imagenet_val(Dataset):
    def __init__(self, data_dir, patch=128):
        super(DataLoader_Imagenet_val, self).__init__()
        self.data_dir = data_dir
        self.patch = patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = io.imread(fn)
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


def load_val_images(dataset_dir):
    fns = [f for f in os.listdir(dataset_dir) if f.endswith('.tif')]
    fns.sort()
    return fns

def load_img(dataset_dir, name):
    im = io.imread(os.path.join(dataset_dir, name))
    return im



def ssim(prediction, target):
    # C1 = (0.01 * MAX_VAL)**2
    # C2 = (0.03 * MAX_VAL)**2
    # img1 = prediction.astype(np.float64)
    # img2 = target.astype(np.float64)
    # kernel = cv2.getGaussianKernel(11, 1.5)
    # window = np.outer(kernel, kernel.transpose())
    # mu1 = cv2.filter3D(img1, -1, window)[5:-5, 5:-5]  # valid
    # mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    # mu1_sq = mu1**2
    # mu2_sq = mu2**2
    # mu1_mu2 = mu1 * mu2
    # sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    # sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    # sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    # ssim_map = ((2 * mu1_mu2 + C1) *
    #             (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
    #                                    (sigma1_sq + sigma2_sq + C2))
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
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(MAX_VAL * MAX_VAL / np.mean(np.square(diff)))
    return psnr


if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Training Set
    TrainingDataset = DataLoader_Imagenet_val(opt.data_dir, patch=opt.patchsize)
    TrainingLoader = DataLoader(dataset=TrainingDataset,
                                num_workers=8,
                                batch_size=opt.batchsize,
                                shuffle=True,
                                pin_memory=False,
                                drop_last=True)

    # Validation Set
    Kodak_dir = opt.val_dir

    names = load_val_images(Kodak_dir)
    valid_dict = {
        "Names": names
    }

    # Noise adder
    noise_adder = AugmentNoise(style=opt.noisetype)

    # Network
    network = UNet(in_nc=opt.n_channel,
                   out_nc=opt.n_channel,
                   n_feature=opt.n_feature)
    if opt.parallel:
        network = torch.nn.DataParallel(network)
    #network = network.cuda()

    # about training scheme
    num_epoch = opt.n_epoch
    ratio = num_epoch / 100
    optimizer = optim.Adam(network.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[
                                             int(20 * ratio) - 1,
                                             int(40 * ratio) - 1,
                                             int(60 * ratio) - 1,
                                             int(80 * ratio) - 1
                                         ],
                                         gamma=opt.gamma)
    print("Batchsize={}, number of epoch={}".format(opt.batchsize, opt.n_epoch))

    checkpoint(network, 0, "model")
    print('init finish')

    for epoch in range(1, opt.n_epoch + 1):
        cnt = 0

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

        network.train()
        for iteration, clean in enumerate(TrainingLoader):
            st = time.time()
            clean = clean / MAX_VAL
            clean = clean.cuda()
            noisy = noise_adder.add_train_noise(clean)

            optimizer.zero_grad()

            mask1, mask2 = generate_mask_pair(noisy)
            noisy_sub1 = generate_subimages(noisy, mask1)
            noisy_sub2 = generate_subimages(noisy, mask2)
            with torch.no_grad():
                noisy_denoised = network(noisy)
            noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
            noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

            noisy_output = network(noisy_sub1)
            noisy_target = noisy_sub2
            Lambda = epoch / opt.n_epoch * opt.increase_ratio
            diff = noisy_output - noisy_target
            exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

            loss1 = torch.mean(diff**2)
            loss2 = Lambda * torch.mean((diff - exp_diff)**2)
            loss_all = opt.Lambda1 * loss1 + opt.Lambda2 * loss2

            loss_all.backward()
            optimizer.step()
            print(
                '{:04d} {:05d} Loss1={:.6f}, Lambda={}, Loss2={:.6f}, Loss_Full={:.6f}, Time={:.4f}'
                .format(epoch, iteration, np.mean(loss1.item()), Lambda,
                        np.mean(loss2.item()), np.mean(loss_all.item()),
                        time.time() - st))

        scheduler.step()

        if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch:
            network.eval()
            # save checkpoint
            checkpoint(network, epoch, "model")
            # validation
            save_model_path = os.path.join(opt.save_model_path, opt.log_name,
                                           systime)
            validation_path = os.path.join(save_model_path, "validation")
            res_dir = os.path.join(save_model_path, "results_n2n")
            os.makedirs(res_dir, exist_ok=True)
            os.makedirs(validation_path, exist_ok=True)
            np.random.seed(101)
            valid_repeat_times = 1

            psnr_result = []
            ssim_result = []
            repeat_times = valid_repeat_times
            for i in range(repeat_times):
                for idx, valid_name in enumerate(valid_dict["Names"]):
                    im = load_img(Kodak_dir, valid_name)
                    origin255 = im.copy()
                    origin255 = origin255.astype(np.float32)
                    im = np.array(im, dtype=np.float32) / MAX_VAL
                    noisy_im = noise_adder.add_valid_noise(im)
                    if epoch == opt.n_snapshot:
                        noisy255 = noisy_im.copy()
                        noisy255 = np.clip(noisy255 * MAX_VAL + 0.5, 0,
                                           MAX_VAL).astype(np.float32)
                    # padding to square
                    H = noisy_im.shape[0]
                    W = noisy_im.shape[1]
                    val_size = (max(H, W) + 31) // 32 * 32
                    # noisy_im = np.pad(
                    #     noisy_im,
                    #     [[0, val_size - H], [0, val_size - W], [0, 0]],
                    #     'reflect')
                    transformer = transforms.Compose([transforms.ToTensor()])
                    noisy_im = transformer(noisy_im)
                    noisy_im = torch.unsqueeze(noisy_im, 0)
                    noisy_im = noisy_im.cuda()
                    with torch.no_grad():
                        prediction = network(noisy_im)
                        prediction = prediction[:, :, :H, :W]
                    prediction = prediction.permute(0, 2, 3, 1)
                    prediction = prediction.cpu().data.clamp(0, 1).numpy()
                    prediction = prediction.squeeze()
                    pred255 = np.clip(prediction * MAX_VAL + 0.5, 0,
                                      MAX_VAL).astype(np.float32)
                    # calculate psnr
                    cur_psnr = calculate_psnr(origin255.astype(np.float32),
                                              pred255.astype(np.float32))
                    psnr_result.append(cur_psnr)
                    cur_ssim = calculate_ssim(origin255.astype(np.float32),
                                              pred255.astype(np.float32))
                    ssim_result.append(cur_ssim)

                    # visualization
                    if i == 0 and epoch == opt.n_snapshot:
                        save_path = os.path.join(
                            res_dir,
                            valid_name[:-4]+"_"+str(epoch)+"_clean.tif")
                        tif.imsave(save_path, origin255)

                        save_path = os.path.join(
                            res_dir,
                            "{}_{:03d}_noisy.tif".format(
                                valid_name[:-4], epoch))
                        tif.imsave(save_path, noisy255)

                    if i == 0:
                        save_path = os.path.join(
                            res_dir,
                            "{}_{:03d}_denoised.tif".format(
                                valid_name[idx][:-4], epoch))
                        tif.imsave(save_path, pred255)

            psnr_result = np.array(psnr_result)
            avg_psnr = np.mean(psnr_result)
            avg_ssim = np.mean(ssim_result)
            log_path = os.path.join(validation_path,
                                    "A_log.csv")
            with open(log_path, "a") as f:
                f.writelines("{},{},{}\n".format(epoch, avg_psnr, avg_ssim))
