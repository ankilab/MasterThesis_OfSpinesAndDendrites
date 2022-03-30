from __future__ import division

import multiprocessing
import time
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import tifffile as tif
from denoising.Neighbor2Neighbor.arch_unet import UNet
import datetime
from denoising.Neighbor2Neighbor.noise_augmetation import *
import denoising.Neighbor2Neighbor.utils as utils
from denoising.Neighbor2Neighbor.data_loader import *
import os


systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
gpu_devices = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices

# MAX_VAL = 12870
# MIN_VAL = -2327
MAX_VAL = 1
MIN_VAL = 0


class Neighbor2Neighbor:
    def __init__(self):
        self.noisetype = "gauss25"
        self.save_model_path = '.\\results'
        self.log_name = 'unet_gauss25_b4e100r02'
        self.gpu_devices = gpu_devices
        self.parallel = False
        self.n_feature = 48
        self.n_channel =1
        self.gamma = 0.5
        self.n_snapshot=25
        self.patchsize = 256
        self.Lambda1 =1.0
        self.Lambda2 = 1.0
        self.increase_ratio = 2.0
        self.batchsize = 4
        self.n_epoch =100
        self.lr =3e-4
        self.data_dir = 'D:/jo77pihe/Registered/Raw_32/2D/subset'
        self.val_dir = 'D:/jo77pihe/Registered/Raw_32/2D/validation'
        self.res_dir = 'D:/jo77pihe/N2N_result'

    def train(self, train_data, val_data, res_path, lr=3e-4, n_epochs=100, batch_size=4):
        self.batchsize = batch_size
        self.n_epoch = n_epochs
        self.lr = lr
        self.res_dir = res_path
        self.val_dir = val_data
        self.data_dir = train_data
        multiprocessing.freeze_support()

        # Training Set
        TrainingDataset = DataLoader_Imagenet_val(self.data_dir, patch=self.patchsize)
        TrainingLoader = DataLoader(dataset=TrainingDataset,
                                    num_workers=8,
                                    batch_size=self.batchsize,
                                    shuffle=True,
                                    pin_memory=False,
                                    drop_last=True)

        # Validation Set
        val_dir = self.val_dir

        # Get validation set files names
        names = utils.load_val_images(val_dir)
        valid_dict = {
            "Names": names
        }

        # Noise adder
        noise_adder = AugmentNoise(style=self.noisetype)

        # Network
        network = UNet(in_nc=self.n_channel,
                       out_nc=self.n_channel,
                       n_feature=self.n_feature)
        if self.parallel:
            network = torch.nn.DataParallel(network)
        # network = network.cuda()

        # about training scheme
        num_epoch = self.n_epoch
        ratio = num_epoch / 100
        optimizer = optim.Adam(network.parameters(), lr=self.lr)
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=[
                                                 int(20 * ratio) - 1,
                                                 int(40 * ratio) - 1,
                                                 int(60 * ratio) - 1,
                                                 int(80 * ratio) - 1
                                             ],
                                             gamma=self.gamma)
        print("Batchsize={}, number of epoch={}".format(self.batchsize, self.n_epoch))

        save_model_path = utils.checkpoint(network, 0, "model", self, systime)
        print('init finish')

        for epoch in range(1, self.n_epoch + 1):
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

            network.train()
            for iteration, clean in enumerate(TrainingLoader):
                st = time.time()
                clean = clean - MIN_VAL
                clean = clean / MAX_VAL
                clean = clean.cuda()
                noisy=clean
                #noisy = noise_adder.add_train_noise(clean)

                optimizer.zero_grad()

                mask1, mask2 = utils.generate_mask_pair(noisy)
                noisy_sub1 = utils.generate_subimages(noisy, mask1)
                noisy_sub2 = utils.generate_subimages(noisy, mask2)
                with torch.no_grad():
                    noisy_denoised = network(noisy)
                noisy_sub1_denoised = utils.generate_subimages(noisy_denoised, mask1)
                noisy_sub2_denoised = utils.generate_subimages(noisy_denoised, mask2)

                noisy_output = network(noisy_sub1)
                noisy_target = noisy_sub2
                Lambda = epoch / self.n_epoch * self.increase_ratio
                diff = noisy_output - noisy_target
                exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

                loss1 = torch.mean(diff ** 2)
                loss2 = Lambda * torch.mean((diff - exp_diff) ** 2)
                loss_all = self.Lambda1 * loss1 + self.Lambda2 * loss2

                loss_all.backward()
                optimizer.step()
                print(
                    '{:04d} {:05d} Loss1={:.6f}, Lambda={}, Loss2={:.6f}, Loss_Full={:.6f}, Time={:.4f}'
                        .format(epoch, iteration, np.mean(loss1.item()), Lambda,
                                np.mean(loss2.item()), np.mean(loss_all.item()),
                                time.time() - st))

            scheduler.step()

            if epoch % self.n_snapshot == 0 or epoch == self.n_epoch:
                network.eval()
                # save checkpoint
                utils.checkpoint(network, epoch, "model", self, systime)
                # validation
                save_model_path = os.path.join(self.save_model_path, self.log_name,
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
                        im = utils.load_img(val_dir, valid_name)
                        im = im - MIN_VAL
                        origin255 = im.copy()
                        origin255 = origin255.astype(np.float32)
                        im = np.array(im, dtype=np.float32) / MAX_VAL
                        noisy_im = im
                        # noisy_im = noise_adder.add_valid_noise(im)
                        if epoch == self.n_snapshot:
                            noisy255 = noisy_im.copy()
                            noisy255 = np.clip(noisy255 * MAX_VAL + 0.5, 0,
                                               MAX_VAL).astype(np.float32)
                        # padding to square
                        H = noisy_im.shape[0]
                        W = noisy_im.shape[1]
                        # val_size = (max(H, W) + 31) // 32 * 32
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
                        cur_psnr = utils.calculate_psnr(origin255.astype(np.float32),
                                                  pred255.astype(np.float32))
                        psnr_result.append(cur_psnr)
                        cur_ssim = utils.calculate_ssim(origin255.astype(np.float32),
                                                  pred255.astype(np.float32))
                        ssim_result.append(cur_ssim)

                        # visualization
                        if i == 0 and epoch == self.n_snapshot:
                            save_path = os.path.join(
                                res_dir,
                                valid_name[:-4] + "_" + str(epoch) + "_clean.tif")
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
                                    valid_name[:-4], epoch))
                            tif.imsave(save_path, pred255)

                psnr_result = np.array(psnr_result)
                avg_psnr = np.mean(psnr_result)
                avg_ssim = np.mean(ssim_result)
                log_path = os.path.join(validation_path, "A_log.csv")
                with open(log_path, "a") as f:
                    f.writelines("{},{},{}/n".format(epoch, avg_psnr, avg_ssim))
        return save_model_path

    def predict(self, checkpoint_dir, input, res_path):
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        net = utils.load_checkpoint(checkpoint_dir)
        files = [f for f in os.listdir(input) if f.endswith('.tif')]

        for f in files:
            noisy = io.imread(os.path.join(input, f))
            res=self.predict_image(checkpoint_dir,noisy, res_path,f, net)
            tif.imsave(os.path.join(res_path, f), res)

    def predict_image(self, checkpoint_dir, img, net=None):
        if net is None:
            net = utils.load_checkpoint(checkpoint_dir)

        noisy = img - MIN_VAL
        # noisy = (noisy-noisy.min()) / MAX_VAL
        # noisy = np.clip(noisy * MAX_VAL + 0.5, 0,
        #                    MAX_VAL).astype(np.float32)
        res = np.zeros(noisy.shape)
        for p in range(noisy.shape[0]):
            img = noisy[p, :, :]
            transformer = transforms.Compose([transforms.ToTensor()])
            img = transformer(img)
            noisy_im = img / MAX_VAL
            noisy_im = torch.unsqueeze(noisy_im, 0)
            noisy_im = noisy_im.cuda()
            with torch.no_grad():
                prediction = net(noisy_im.type(torch.float32))
            prediction = prediction.permute(0, 2, 3, 1)
            prediction = prediction.cpu().data.clamp(0, 1).numpy()
            prediction = prediction.squeeze()
            res[p, :, :] = np.clip(prediction * MAX_VAL + 0.5, 0,
                                   MAX_VAL).astype(np.float32)
        return res

