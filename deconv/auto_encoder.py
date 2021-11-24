from .deconvolver import Deconvolver
from denoising_ae.auto_encoder_net import Autoencoder
import data_augmentation as da
import numpy as np
import torch
import torch.optim as optim
import os


class AE(Deconvolver):
    def __init__(self, args):
        super().__init__(args)
        self.net = Autoencoder(args)
        self.data_provider = da.DataProvider((args['z_shape'], args['xy_shape']),'data.h5')
        # the loss function
        self.criterion = torch.nn.MSELoss()
        # the optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=float(args['learning_rate']))

    def preprocess(self):
        pass

    def train(self, epochs=10, batch_size=8):
        train_loss = []
        n_steps_p_epoch = np.ceil(self.data_provider.size[0]/batch_size).astype(int)
        for epoch in range(epochs):
            running_loss = 0.0
            self.data_provider.shuffle()
            for i in range(n_steps_p_epoch):
                raw, gt = self.data_provider.get(batch_size)
                raw = raw[:, np.newaxis, :, :, :]
                img_noisy = torch.tensor(raw).to(self.net.device)
                self.optimizer.zero_grad()
                outputs = self.net(img_noisy)
                loss = self.criterion(outputs, img_noisy)
                # backpropagation
                loss.backward()
                # update the parameters
                self.optimizer.step()
                running_loss += loss.item()

            loss = running_loss / n_steps_p_epoch
            train_loss.append(loss)
            print('Epoch {} of {}, Train Loss: {:.3f}'.format(
                epoch + 1, epochs, loss))
        model_dir = os.path.join(self.res_path, 'model.pth')
        torch.save(self.net, model_dir)
        return model_dir, train_loss

    def predict(self, X, model_dir, save_as=None):
        """

        :param X: Input image
        :param model_dir:
        :param save_as: If not None, result is saved as file with the name specified (e.g. 'denoised_img.tif')
        :return:
        """

        pass

    def predict_img(self, img, sampling_step, max_value, batch_sz):
        img_sz = img.shape
        sc = max_value / 2.0
        cnn_input_sz = [batch_sz, sampling_step[0], sampling_step[1], sampling_step[2], 1]
        input_tensor = np.zeros(cnn_input_sz, 'float32')

        wz = [sampling_step[0], sampling_step[1], sampling_step[2]]
        wz = np.int_(wz)

        x_loc = np.zeros(batch_sz, 'int32')
        y_loc = np.zeros(batch_sz, 'int32')
        z_loc = np.zeros(batch_sz, 'int32')
        count = 0

        img = img.astype('float32')
        recon_img = np.zeros(img_sz)
        recon_img = recon_img.astype('float32')
        occupancy = np.zeros(img_sz)
        occupancy = occupancy.astype('float32')

        x_direction = (0, 1, 0, 1, 0, 1, 0, 1)
        y_direction = (0, 0, 1, 1, 0, 0, 1, 1)
        z_direction = (0, 0, 0, 0, 1, 1, 1, 1)

        for i in range(0, 8):
            print('%g%%' % (i / 8 * 100))
            x_dir = x_direction[i]
            y_dir = y_direction[i]
            z_dir = z_direction[i]

            if z_dir == 0:
                for z in range(0, img_sz[0] - wz[0], sampling_step[0]):
                    if y_dir == 0:
                        for y in range(0, img_sz[1] - wz[1], sampling_step[1]):
                            if x_dir == 0:
                                for x in range(0, img_sz[2] - wz[2], sampling_step[2]):
                                    patch = img[z:z + wz[0], y:y + wz[1], x:x + wz[2]]
                                    patch = get_resized_patches(patch, max_value)
                                    input_patch, input_tensor, x_loc, y_loc, z_loc = get_batch(patch, input_tensor,
                                                                                               x, y, z, x_loc,
                                                                                               y_loc, z_loc, count)
                                    count = count + 1
                                    if count % batch_sz == 0:
                                        pred_patch = self.net(input_tensor).cpu().detach().numpy()
                                        pred_patch = np.clip(pred_patch, -1, 1)
                                        pred_patch = np.reshape((pred_patch + 1) * sc, cnn_input_sz[0:4])
                                        for k in range(0, count):
                                            prev_patch = recon_img[z_loc[k]:z_loc[k] + wz[0],
                                                         y_loc[k]:y_loc[k] + wz[1], x_loc[k]:x_loc[k] + wz[2]]
                                            recon_img[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1],
                                            x_loc[k]:x_loc[k] + wz[2]] = pred_patch[k] + prev_patch
                                            occupancy[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1],
                                            x_loc[k]:x_loc[k] + wz[2]] = occupancy[z_loc[k]:z_loc[k] + wz[0],
                                                                         y_loc[k]:y_loc[k] + wz[1],
                                                                         x_loc[k]:x_loc[k] + wz[2]] + 1
                                            count = 0

                            elif x_dir == 1:
                                for x in range(img_sz[2] - wz[2], 0, -sampling_step[2]):
                                    patch = img[z:z + wz[0], y:y + wz[1], x:x + wz[2]]
                                    patch = get_resized_patches(patch, max_value)
                                    input_patch, input_tensor, x_loc, y_loc, z_loc = get_batch(patch, input_tensor,
                                                                                               x, y, z, x_loc,
                                                                                               y_loc, z_loc, count)
                                    count = count + 1
                                    if count % batch_sz == 0:
                                        pred_patch = self.net(input_tensor).cpu().detach().numpy()
                                        pred_patch = np.clip(pred_patch, -1, 1)
                                        pred_patch = np.reshape((pred_patch + 1) * sc, cnn_input_sz[0:4])
                                        for k in range(0, count):
                                            prev_patch = recon_img[z_loc[k]:z_loc[k] + wz[0],
                                                         y_loc[k]:y_loc[k] + wz[1], x_loc[k]:x_loc[k] + wz[2]]
                                            recon_img[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1],
                                            x_loc[k]:x_loc[k] + wz[2]] = pred_patch[k] + prev_patch
                                            occupancy[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1],
                                            x_loc[k]:x_loc[k] + wz[2]] = occupancy[z_loc[k]:z_loc[k] + wz[0],
                                                                         y_loc[k]:y_loc[k] + wz[1],
                                                                         x_loc[k]:x_loc[k] + wz[2]] + 1
                                            count = 0
                    elif y_dir == 1:
                        for y in range(img_sz[1] - wz[1], 0, -sampling_step[1]):
                            if x_dir == 0:
                                for x in range(0, img_sz[2] - wz[2], sampling_step[2]):
                                    patch = img[z:z + wz[0], y:y + wz[1], x:x + wz[2]]
                                    patch = get_resized_patches(patch, max_value)
                                    input_patch, input_tensor, x_loc, y_loc, z_loc = get_batch(patch, input_tensor,
                                                                                               x, y, z, x_loc,
                                                                                               y_loc, z_loc, count)
                                    count = count + 1
                                    if count % batch_sz == 0:
                                        pred_patch = self.net(input_tensor).cpu().detach().numpy()
                                        pred_patch = np.clip(pred_patch, -1, 1)
                                        pred_patch = np.reshape((pred_patch + 1) * sc, cnn_input_sz[0:4])
                                        for k in range(0, count):
                                            prev_patch = recon_img[z_loc[k]:z_loc[k] + wz[0],
                                                         y_loc[k]:y_loc[k] + wz[1], x_loc[k]:x_loc[k] + wz[2]]
                                            recon_img[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1],
                                            x_loc[k]:x_loc[k] + wz[2]] = pred_patch[k] + prev_patch
                                            occupancy[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1],
                                            x_loc[k]:x_loc[k] + wz[2]] = occupancy[z_loc[k]:z_loc[k] + wz[0],
                                                                         y_loc[k]:y_loc[k] + wz[1],
                                                                         x_loc[k]:x_loc[k] + wz[2]] + 1
                                            count = 0
                            elif x_dir == 1:
                                for x in range(img_sz[2] - wz[2], 0, -sampling_step[2]):
                                    patch = img[z:z + wz[0], y:y + wz[1], x:x + wz[2]]
                                    patch = get_resized_patches(patch, max_value)
                                    input_patch, input_tensor, x_loc, y_loc, z_loc = get_batch(patch, input_tensor,
                                                                                               x, y, z, x_loc,
                                                                                               y_loc, z_loc, count)
                                    count = count + 1
                                    if count % batch_sz == 0:
                                        pred_patch = self.net(input_tensor).cpu().detach().numpy()
                                        pred_patch = np.clip(pred_patch, -1, 1)
                                        pred_patch = np.reshape((pred_patch + 1) * sc, cnn_input_sz[0:4])
                                        for k in range(0, count):
                                            prev_patch = recon_img[z_loc[k]:z_loc[k] + wz[0],
                                                         y_loc[k]:y_loc[k] + wz[1], x_loc[k]:x_loc[k] + wz[2]]
                                            recon_img[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1],
                                            x_loc[k]:x_loc[k] + wz[2]] = pred_patch[k] + prev_patch
                                            occupancy[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1],
                                            x_loc[k]:x_loc[k] + wz[2]] = occupancy[z_loc[k]:z_loc[k] + wz[0],
                                                                         y_loc[k]:y_loc[k] + wz[1],
                                                                         x_loc[k]:x_loc[k] + wz[2]] + 1
                                            count = 0
            elif z_dir == 1:
                for z in range(img_sz[0] - wz[0], 0, -sampling_step[0]):
                    if y_dir == 0:
                        for y in range(0, img_sz[1] - wz[1], sampling_step[1]):
                            if x_dir == 0:
                                for x in range(0, img_sz[2] - wz[2], sampling_step[2]):
                                    patch = img[z:z + wz[0], y:y + wz[1], x:x + wz[2]]
                                    patch = get_resized_patches(patch, max_value)
                                    input_patch, input_tensor, x_loc, y_loc, z_loc = get_batch(patch, input_tensor,
                                                                                               x, y, z, x_loc,
                                                                                               y_loc, z_loc, count)
                                    count = count + 1
                                    if count % batch_sz == 0:
                                        pred_patch = self.net(input_tensor).cpu().detach().numpy()
                                        pred_patch = np.clip(pred_patch, -1, 1)
                                        pred_patch = np.reshape((pred_patch + 1) * sc, cnn_input_sz[0:4])
                                        for k in range(0, count):
                                            prev_patch = recon_img[z_loc[k]:z_loc[k] + wz[0],
                                                         y_loc[k]:y_loc[k] + wz[1], x_loc[k]:x_loc[k] + wz[2]]
                                            recon_img[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1],
                                            x_loc[k]:x_loc[k] + wz[2]] = pred_patch[k] + prev_patch
                                            occupancy[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1],
                                            x_loc[k]:x_loc[k] + wz[2]] = occupancy[z_loc[k]:z_loc[k] + wz[0],
                                                                         y_loc[k]:y_loc[k] + wz[1],
                                                                         x_loc[k]:x_loc[k] + wz[2]] + 1
                                            count = 0
                            elif x_dir == 1:
                                for x in range(img_sz[2] - wz[2], 0, -sampling_step[2]):
                                    patch = img[z:z + wz[0], y:y + wz[1], x:x + wz[2]]
                                    patch = get_resized_patches(patch, max_value)
                                    input_patch, input_tensor, x_loc, y_loc, z_loc = get_batch(patch, input_tensor,
                                                                                               x, y, z, x_loc,
                                                                                               y_loc, z_loc, count)
                                    count = count + 1
                                    if count % batch_sz == 0:
                                        pred_patch = self.net(input_tensor).cpu().detach().numpy()
                                        pred_patch = np.clip(pred_patch, -1, 1)
                                        pred_patch = np.reshape((pred_patch + 1) * sc, cnn_input_sz[0:4])
                                        for k in range(0, count):
                                            prev_patch = recon_img[z_loc[k]:z_loc[k] + wz[0],
                                                         y_loc[k]:y_loc[k] + wz[1], x_loc[k]:x_loc[k] + wz[2]]
                                            recon_img[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1],
                                            x_loc[k]:x_loc[k] + wz[2]] = pred_patch[k] + prev_patch
                                            occupancy[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1],
                                            x_loc[k]:x_loc[k] + wz[2]] = occupancy[z_loc[k]:z_loc[k] + wz[0],
                                                                         y_loc[k]:y_loc[k] + wz[1],
                                                                         x_loc[k]:x_loc[k] + wz[2]] + 1
                                            count = 0
                    elif y_dir == 1:
                        for y in range(img_sz[1] - wz[1], 0, -sampling_step[1]):
                            if x_dir == 0:
                                for x in range(0, img_sz[2] - wz[2], sampling_step[2]):
                                    patch = img[z:z + wz[0], y:y + wz[1], x:x + wz[2]]
                                    patch = get_resized_patches(patch, max_value)
                                    input_patch, input_tensor, x_loc, y_loc, z_loc = get_batch(patch, input_tensor,
                                                                                               x, y, z, x_loc,
                                                                                               y_loc, z_loc, count)
                                    count = count + 1
                                    if count % batch_sz == 0:
                                        pred_patch = self.net(input_tensor).cpu().detach().numpy()
                                        pred_patch = np.clip(pred_patch, -1, 1)
                                        pred_patch = np.reshape((pred_patch + 1) * sc, cnn_input_sz[0:4])
                                        for k in range(0, count):
                                            prev_patch = recon_img[z_loc[k]:z_loc[k] + wz[0],
                                                         y_loc[k]:y_loc[k] + wz[1], x_loc[k]:x_loc[k] + wz[2]]
                                            recon_img[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1],
                                            x_loc[k]:x_loc[k] + wz[2]] = pred_patch[k] + prev_patch
                                            occupancy[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1],
                                            x_loc[k]:x_loc[k] + wz[2]] = occupancy[z_loc[k]:z_loc[k] + wz[0],
                                                                         y_loc[k]:y_loc[k] + wz[1],
                                                                         x_loc[k]:x_loc[k] + wz[2]] + 1
                                            count = 0
                            elif x_dir == 1:
                                for x in range(img_sz[2] - wz[2], 0, -sampling_step[2]):
                                    patch = img[z:z + wz[0], y:y + wz[1], x:x + wz[2]]
                                    patch = get_resized_patches(patch, max_value)
                                    input_patch, input_tensor, x_loc, y_loc, z_loc = get_batch(patch, input_tensor,
                                                                                               x, y, z, x_loc,
                                                                                               y_loc, z_loc, count)
                                    count = count + 1
                                    if count % batch_sz == 0:
                                        pred_patch = self.net(input_tensor).cpu().detach().numpy()
                                        pred_patch = np.clip(pred_patch, -1, 1)
                                        pred_patch = np.reshape((pred_patch + 1) * sc, cnn_input_sz[0:4])
                                        for k in range(0, count):
                                            prev_patch = recon_img[z_loc[k]:z_loc[k] + wz[0],
                                                         y_loc[k]:y_loc[k] + wz[1], x_loc[k]:x_loc[k] + wz[2]]
                                            recon_img[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1],
                                            x_loc[k]:x_loc[k] + wz[2]] = pred_patch[k] + prev_patch
                                            occupancy[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1],
                                            x_loc[k]:x_loc[k] + wz[2]] = occupancy[z_loc[k]:z_loc[k] + wz[0],
                                                                         y_loc[k]:y_loc[k] + wz[1],
                                                                         x_loc[k]:x_loc[k] + wz[2]] + 1
                                            count = 0

        pred_patch = self.net(input_tensor).cpu().detach().numpy()
        pred_patch = np.clip(pred_patch, -1, 1)
        pred_patch = np.reshape((pred_patch + 1) * sc, cnn_input_sz[0:4])
        for k in range(0, count):
            prev_patch = recon_img[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1], x_loc[k]:x_loc[k] + wz[2]]
            recon_img[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1], x_loc[k]:x_loc[k] + wz[2]] = pred_patch[
                                                                                                             k] + prev_patch
            occupancy[z_loc[k]:z_loc[k] + wz[0], y_loc[k]:y_loc[k] + wz[1], x_loc[k]:x_loc[k] + wz[2]] = occupancy[
                                                                                                         z_loc[k]:
                                                                                                         z_loc[k] +
                                                                                                         wz[0],
                                                                                                         y_loc[k]:
                                                                                                         y_loc[k] +
                                                                                                         wz[1],
                                                                                                         x_loc[k]:
                                                                                                         x_loc[k] +
                                                                                                         wz[2]] + 1
            count = 0

        recon_img = np.divide(recon_img, occupancy)
        print('done')
        return recon_img


def get_resized_patches(patch, max_value):
    patch = patch.astype('float32')
    sc = max_value/2.0
    return patch.astype('float32')/sc-1.0


def get_batch(patch, tensor, x,y,z, x_loc, y_loc, z_loc, count):
    patch_sz = patch.shape
    patch = np.reshape(patch, [1, patch_sz[0], patch_sz[1], patch_sz[2], 1] )
    tensor[count, 0:patch_sz[0], 0:patch_sz[1], 0:patch_sz[2], 0:1] = patch
    x_loc[count] = x
    y_loc[count] = y
    z_loc[count] = z
    return patch, tensor, x_loc, y_loc, z_loc

