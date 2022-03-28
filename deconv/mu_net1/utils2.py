# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 20:27:53 2018

@author: sehyung
"""
import numpy as np
import sys


def _printProgressBar(i, max_bars, postText):
    """
    Print progress bar
    TODO: needs fix

    :param i: Current status
    :type i: float
    :param max_bars: Maximum number of bars
    :type max_bars: float
    :param postText: Text to be printed
    :type postText: string
    """
    n_bar = max_bars
    j = i / max_bars
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()


def _get_normalized_patch(patch, max_value, min_value):
    """
    Normalize image patch.

    :param patch: Image patch
    :type patch: nd.array
    :param max_value: Maximum value of image range
    :type max_value: float
    :param min_value: Minimum value of image range
    :type min_value: float
    :return: Normalized patch
    """
    patch = patch.astype('float32') -min_value
    sc = max_value / 2.0
    return patch.astype('float32') / sc - 1.0


def predict_patch(model, img, recon_img, occupancy, patch_sz, max_value, min_value, idx, level_idx=0):
    """
    Deconvolve image patch.

    :param model: Trained deconvolution Mu-Net model
    :type model: TF model
    :param img: Complete image to be deconvolved
    :param recon_img: Interim result
    :type recon_img: nd.array
    :param occupancy: Matrix indicating which pixel was predicted for how often
    :type occupancy: nd.array
    :param patch_sz: Patch size
    :param max_value: Maximum value of image range
    :type max_value: float
    :param min_value: Minimum value of image range
    :type min_value: float
    :param idx: Index of patch start in complete image
    :param level_idx: Index of model level
    :type level_idx: int, optional
    :return: Updated interim result, occupancy
    :rtype: nd.array, nd.array
    """
    (z, y, x) = idx
    patch = img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]]
    patch = _get_normalized_patch(patch, max_value, min_value)
    pred_patch = model.predict(np.reshape(patch, [1, patch_sz[0], patch_sz[1], patch_sz[2], 1]))[level_idx]
    pred_patch = np.reshape(pred_patch, [patch_sz[0], patch_sz[1], patch_sz[2]])
    pred_patch = np.clip(pred_patch, -1, 1)
    recon_img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]] += pred_patch
    occupancy[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]] += 1

    return recon_img, occupancy


def window_sliding(self, img, sampling_step, patch_sz, max_value, min_value, batch_sz=1, n_levels=1):
    """
    Deconvolve image by sliding a window over the input.

    :param self: Mu-Net object
    :type self: deconv.mu_net1.denoiser_only_mu.Denoiser
    :param img: Input image
    :type img: nd.array
    :param sampling_step: Sampling step at which to predict patches
    :type sampling_step: int
    :param patch_sz: x-y-, z-dimension of patch
    :type patch_sz: tuple(int,int,int)
    :param max_value: Maximum value of image range
    :type max_value: float
    :param min_value: Minimum value of image range
    :type min_value: float
    :param n_levels: Number of levels of Mu-Net model
    :type n_levels: int, optional
    :return: Deconvolved image
    :rtype: nd.array
    """
    img_sz = img.shape
    sampling_step = sampling_step.astype(int)
    patch_sz = patch_sz.astype(int)
    recon_img = np.zeros(img_sz, dtype=np.float32)
    occupancy = np.zeros(img_sz, dtype=np.float32)
    level_idx = n_levels - 1

    num_total_bars = 50
    num_total_iters = int(img_sz[0]*8 / sampling_step[0]) * int(img_sz[1]*8 / sampling_step[1]) * int(img_sz[2] *8/ sampling_step[2])
    iter_count = 0

    for z in range(0, img_sz[0] - patch_sz[0] + 1, sampling_step[0]):
        for y in range(0, img_sz[1] - patch_sz[1] + 1, sampling_step[1]):
            for x in range(0, img_sz[2] - patch_sz[2] + 1, sampling_step[2]):
                recon_img, occupancy = predict_patch(self.model, img, recon_img, occupancy, patch_sz, max_value,min_value,
                                                     (z, y, x), level_idx)
                iter_count = iter_count + 1
                _printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "progress")

    for z in range(img_sz[0] - patch_sz[0], -1, -sampling_step[0]):
        for y in range(0, img_sz[1] - patch_sz[1] + 1, sampling_step[1]):
            for x in range(0, img_sz[2] - patch_sz[2] + 1, sampling_step[2]):
                recon_img, occupancy = predict_patch(self.model, img, recon_img, occupancy, patch_sz, max_value,min_value,
                                                     (z, y, x), level_idx)
                iter_count = iter_count + 1
                _printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "progress")

    for z in range(0, img_sz[0] - patch_sz[0] + 1, sampling_step[0]):
        for y in range(img_sz[1] - patch_sz[1], -1, -sampling_step[1]):
            for x in range(0, img_sz[2] - patch_sz[2] + 1, sampling_step[2]):
                recon_img, occupancy = predict_patch(self.model, img, recon_img, occupancy, patch_sz, max_value,min_value,
                                                     (z, y, x), level_idx)
                iter_count = iter_count + 1
                _printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "progress")

    for z in range(img_sz[0] - patch_sz[0], -1, -sampling_step[0]):
        for y in range(img_sz[1] - patch_sz[1], -1, -sampling_step[1]):
            for x in range(0, img_sz[2] - patch_sz[2] + 1, sampling_step[2]):
                recon_img, occupancy = predict_patch(self.model, img, recon_img, occupancy, patch_sz, max_value,min_value,
                                                     (z, y, x), level_idx)
                iter_count = iter_count + 1
                _printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "progress")

    for z in range(0, img_sz[0] - patch_sz[0] + 1, sampling_step[0]):
        for y in range(0, img_sz[1] - patch_sz[1] + 1, sampling_step[1]):
            for x in range(img_sz[2] - patch_sz[2], -1, -sampling_step[2]):
                recon_img, occupancy = predict_patch(self.model, img, recon_img, occupancy, patch_sz, max_value,min_value,
                                                     (z, y, x), level_idx)
                iter_count = iter_count + 1
                _printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "progress")

    for z in range(img_sz[0] - patch_sz[0], -1, -sampling_step[0]):
        for y in range(0, img_sz[1] - patch_sz[1] + 1, sampling_step[1]):
            for x in range(img_sz[2] - patch_sz[2], -1, -sampling_step[2]):
                recon_img, occupancy = predict_patch(self.model, img, recon_img, occupancy, patch_sz, max_value,min_value,
                                                     (z, y, x), level_idx)

                iter_count = iter_count + 1
                _printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "progress")

    for z in range(0, img_sz[0] - patch_sz[0] + 1, sampling_step[0]):
        for y in range(img_sz[1] - patch_sz[1], -1, -sampling_step[1]):
            for x in range(img_sz[2] - patch_sz[2], -1, -sampling_step[2]):
                recon_img, occupancy = predict_patch(self.model, img, recon_img, occupancy, patch_sz, max_value, min_value,
                                                     (z, y, x), level_idx)
                iter_count = iter_count + 1
                _printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "progress")

    for z in range(img_sz[0] - patch_sz[0], -1, -sampling_step[0]):
        for y in range(img_sz[1] - patch_sz[1], -1, -sampling_step[1]):
            for x in range(img_sz[2] - patch_sz[2], -1, -sampling_step[2]):
                recon_img, occupancy = predict_patch(self.model, img, recon_img, occupancy, patch_sz, max_value, min_value,
                                                     (z, y, x), level_idx)
                iter_count = iter_count + 1
                _printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "progress")

    recon_img = np.divide(recon_img, occupancy)
    recon_img = (recon_img + 1.) * max_value / 2.
    recon_img = np.clip(recon_img, 0, max_value)
    _printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "complete\n")

    return recon_img
