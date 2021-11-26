# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 20:27:53 2018

@author: sehyung
"""
import numpy as np
import sys


def printProgressBar(i, max_bars, postText):
    n_bar = max_bars
    j = i / max_bars
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()


def read_patches(batch_sz, max_value):
    # implement image batch read for your dataset
    return


def get_normalized_patch(patch, max_value):
    patch = patch.astype('float32')
    sc = max_value / 2.0
    return patch.astype('float32') / sc - 1.0


def window_sliding(self, img, sampling_step, patch_sz, max_value, batch_sz, n_levels=2):
    img_sz = img.shape
    sampling_step = sampling_step.astype(int)
    patch_sz = patch_sz.astype(int)
    recon_img = np.zeros(img_sz, dtype=np.float32)
    occupancy = np.zeros(img_sz, dtype=np.float32)
    level_idx= n_levels-1

    num_total_bars = 50
    num_total_iters = int(img_sz[0] / patch_sz[0]) * int(img_sz[1] / patch_sz[1]) * int(img_sz[2] / patch_sz[2]) * 8
    iter_count = 0

    for z in range(0, img_sz[0] - patch_sz[0] + 1, sampling_step[0]):
        for y in range(0, img_sz[1] - patch_sz[1] + 1, sampling_step[1]):
            for x in range(0, img_sz[2] - patch_sz[2] + 1, sampling_step[2]):
                patch = img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]]
                patch = get_normalized_patch(patch, max_value)
                pred_patch = self.model.predict(np.reshape(patch, [1, patch_sz[0], patch_sz[1], patch_sz[2], 1]))[level_idx]
                pred_patch = np.reshape(pred_patch, [patch_sz[0], patch_sz[1], patch_sz[2]])
                pred_patch = np.clip(pred_patch, -1, 1)
                recon_img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]] += pred_patch
                occupancy[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]] += 1
                iter_count = iter_count + 1
                printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "progress")

    for z in range(img_sz[0] - patch_sz[0], -1, -sampling_step[0]):
        for y in range(0, img_sz[1] - patch_sz[1] + 1, sampling_step[1]):
            for x in range(0, img_sz[2] - patch_sz[2] + 1, sampling_step[2]):
                patch = img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]]
                patch = get_normalized_patch(patch, max_value)
                pred_patch = self.model.predict(np.reshape(patch, [1, patch_sz[0], patch_sz[1], patch_sz[2], 1]))[level_idx]
                pred_patch = np.reshape(pred_patch, [patch_sz[0], patch_sz[1], patch_sz[2]])
                pred_patch = np.clip(pred_patch, -1, 1)
                recon_img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]] += pred_patch
                occupancy[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]]  += 1
                iter_count = iter_count + 1
                printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "progress")

    for z in range(0, img_sz[0] - patch_sz[0] + 1, sampling_step[0]):
        for y in range(img_sz[1] - patch_sz[1], -1, -sampling_step[1]):
            for x in range(0, img_sz[2] - patch_sz[2] + 1, sampling_step[2]):
                patch = img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]]
                patch = get_normalized_patch(patch, max_value)
                pred_patch = self.model.predict(np.reshape(patch, [1, patch_sz[0], patch_sz[1], patch_sz[2], 1]))[level_idx]
                pred_patch = np.reshape(pred_patch, [patch_sz[0], patch_sz[1], patch_sz[2]])
                pred_patch = np.clip(pred_patch, -1, 1)
                recon_img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]] += pred_patch
                occupancy[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]] += 1
                iter_count = iter_count + 1
                printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "progress")

    for z in range(img_sz[0] - patch_sz[0], -1, -sampling_step[0]):
        for y in range(img_sz[1] - patch_sz[1], -1, -sampling_step[1]):
            for x in range(0, img_sz[2] - patch_sz[2] + 1, sampling_step[2]):
                patch = img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]]
                patch = get_normalized_patch(patch, max_value)
                pred_patch = self.model.predict(np.reshape(patch, [1, patch_sz[0], patch_sz[1], patch_sz[2], 1]))[level_idx]
                pred_patch = np.reshape(pred_patch, [patch_sz[0], patch_sz[1], patch_sz[2]])
                pred_patch = np.clip(pred_patch, -1, 1)
                recon_img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]] += pred_patch
                occupancy[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]]  += 1
                iter_count = iter_count + 1
                printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "progress")

    for z in range(0, img_sz[0] - patch_sz[0] + 1, sampling_step[0]):
        for y in range(0, img_sz[1] - patch_sz[1] + 1, sampling_step[1]):
            for x in range(img_sz[2] - patch_sz[2], -1, -sampling_step[2]):
                patch = img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]]
                patch = get_normalized_patch(patch, max_value)
                pred_patch = self.model.predict(np.reshape(patch, [1, patch_sz[0], patch_sz[1], patch_sz[2], 1]))[level_idx]
                pred_patch = np.reshape(pred_patch, [patch_sz[0], patch_sz[1], patch_sz[2]])
                pred_patch = np.clip(pred_patch, -1, 1)
                recon_img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]] += pred_patch
                occupancy[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]]  += 1
                iter_count = iter_count + 1
                printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "progress")

    for z in range(img_sz[0] - patch_sz[0], -1, -sampling_step[0]):
        for y in range(0, img_sz[1] - patch_sz[1] + 1, sampling_step[1]):
            for x in range(img_sz[2] - patch_sz[2], -1, -sampling_step[2]):
                patch = img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]]
                patch = get_normalized_patch(patch, max_value)
                pred_patch = self.model.predict(np.reshape(patch, [1, patch_sz[0], patch_sz[1], patch_sz[2], 1]))[level_idx]
                pred_patch = np.reshape(pred_patch, [patch_sz[0], patch_sz[1], patch_sz[2]])
                pred_patch = np.clip(pred_patch, -1, 1)
                recon_img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]] += pred_patch
                occupancy[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]]  += 1
                iter_count = iter_count + 1
                printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "progress")

    for z in range(0, img_sz[0] - patch_sz[0] + 1, sampling_step[0]):
        for y in range(img_sz[1] - patch_sz[1], -1, -sampling_step[1]):
            for x in range(img_sz[2] - patch_sz[2], -1, -sampling_step[2]):
                patch = img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]]
                patch = get_normalized_patch(patch, max_value)
                pred_patch = self.model.predict(np.reshape(patch, [1, patch_sz[0], patch_sz[1], patch_sz[2], 1]))[level_idx]
                pred_patch = np.reshape(pred_patch, [patch_sz[0], patch_sz[1], patch_sz[2]])
                pred_patch = np.clip(pred_patch, -1, 1)
                recon_img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]] += pred_patch
                occupancy[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]]  += 1
                iter_count = iter_count + 1
                printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "progress")

    for z in range(img_sz[0] - patch_sz[0], -1, -sampling_step[0]):
        for y in range(img_sz[1] - patch_sz[1], -1, -sampling_step[1]):
            for x in range(img_sz[2] - patch_sz[2], -1, -sampling_step[2]):
                patch = img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]]
                patch = get_normalized_patch(patch, max_value)
                pred_patch = self.model.predict(np.reshape(patch, [1, patch_sz[0], patch_sz[1], patch_sz[2], 1]))[level_idx]
                pred_patch = np.reshape(pred_patch, [patch_sz[0], patch_sz[1], patch_sz[2]])
                pred_patch = np.clip(pred_patch, -1, 1)
                recon_img[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]] += pred_patch
                occupancy[z:z + patch_sz[0], y:y + patch_sz[1], x:x + patch_sz[2]]  += 1
                iter_count = iter_count + 1
                printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "progress")

    recon_img = np.divide(recon_img, occupancy)
    recon_img = (recon_img + 1.) * max_value / 2.
    recon_img = np.clip(recon_img, 0, max_value)
    printProgressBar(iter_count / num_total_iters * num_total_bars, num_total_bars, "complete\n")

    return recon_img
