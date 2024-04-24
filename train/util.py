import cv2
import torch
import numpy as np
import random
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_start_iters(start_iter, dataset_size):
    if start_iter == 0:
        return 0, 0
    start_epoch = (start_iter + 1) // dataset_size
    start_iter  = (start_iter + 1) %  dataset_size
    return start_epoch, start_iter


class results_saver():
    def __init__(self, opt):
        path = os.path.join(opt.results_dir, opt.name)
        self.path_label = os.path.join(path, "image_real")
        self.path_image = os.path.join(path, "image_reco")
        self.path_to_save = {"image_real": self.path_label, "image_reco": self.path_image}
        os.makedirs(self.path_label, exist_ok=True)
        os.makedirs(self.path_image, exist_ok=True)

    def __call__(self, label, generated, mr_image, name):
        assert len(label) == len(generated)
        for i in range(len(label)):
            im = tens_to_lab_color(label[i], self.num_cl)
            self.save_im(im, "label", name[i])
            im = tens_to_im(generated[i]) * 255
            self.save_im(im, "image", name[i])
            fake = (tens_to_im(generated[i]) * 255).astype(np.uint8)
            mr = (tens_to_im(mr_image[i]) * 255).astype(np.uint8)
            heatmap_array, mae = self.calculate_mae(mr, fake)
            self.mmae.append(mae)
            self.save_im(heatmap_array, "mae", name[i])

        print('mean MAE:', sum(self.mmae)/len(self.mmae))

    def save_im(self, im, mode, name):
        im = Image.fromarray(im.astype(np.uint8))
        #print(name.split("/")[-1])
        im.save(os.path.join(self.path_to_save[mode], name.split("/")[-1]).replace('.jpg', '.png'))


class combined_images_saver():
    def __init__(self, opt):
        path = os.path.join(opt.results_dir, opt.name)
        self.path_combined = os.path.join(path, "combined_images")
        self.path_mae = os.path.join(path, "combined_mae")
        os.makedirs(self.path_combined, exist_ok=True)
        os.makedirs(self.path_mae, exist_ok=True)
        self.num_cl = opt.label_nc + 2

    def __call__(self, label, generated1, generated2, generated3, generated4, mr_image, ct_image, name):
        assert len(label) == len(generated1)
        for i in range(len(label)):
            im_label = tens_to_lab_color(label[i], self.num_cl)
            im_image1 = (tens_to_im(generated1[i]) * 255).astype(np.uint8)
            im_image2 = (tens_to_im(generated2[i]) * 255).astype(np.uint8)
            im_image3 = (tens_to_im(generated3[i]) * 255).astype(np.uint8)
            im_image4 = (tens_to_im(generated4[i]) * 255).astype(np.uint8)
            im_image5 = (tens_to_im(mr_image[i]) * 255).astype(np.uint8)
            im_image6 = (tens_to_im(ct_image[i]) * 255).astype(np.uint8)
            hm1 = self.calculate_mae(im_image5, im_image1)
            hm2 = self.calculate_mae(im_image5, im_image2)
            hm3 = self.calculate_mae(im_image5, im_image3)
            hm4 = self.calculate_mae(im_image5, im_image4)
            combined_image = self.combine_images(im_label, im_image1, im_image2, im_image3, im_image4, im_image5, im_image6)
            combined_heatmap = self.combine_images_all(im_label, im_image1, im_image2, im_image3, im_image4, im_image5, im_image6, hm1,
                               hm2, hm3, hm4)
            self.save_combined_image(combined_heatmap, self.path_mae, name[i])
            self.save_combined_image(combined_image, self.path_combined, name[i])

    def combine_images(self, im_label, im_image1, im_image2, im_image3, im_image4, im_image5, im_image6):
        width, height = im_label.shape[1], im_label.shape[0]
        combined_image = Image.new("RGB", (width * 7, height))
        combined_image.paste(Image.fromarray(im_label), (0, 0))
        combined_image.paste(Image.fromarray(im_image6), (width, 0))
        combined_image.paste(Image.fromarray(im_image5), (width * 2, 0))
        combined_image.paste(Image.fromarray(im_image1), (width * 3, 0))
        combined_image.paste(Image.fromarray(im_image2), (width * 4, 0))
        combined_image.paste(Image.fromarray(im_image3), (width * 5, 0))
        combined_image.paste(Image.fromarray(im_image4), (width * 6, 0))

        return combined_image

    def combine_images_all(self, im_label, im_image1, im_image2, im_image3, im_image4, im_image5, im_image6, hm1, hm2, hm3, hm4):
        width, height = im_label.shape[1], im_label.shape[0]
        combined_image = Image.new("RGB", (width * 7, height*2))
        combined_image.paste(Image.fromarray(im_label), (0, 0))
        combined_image.paste(Image.fromarray(im_image6), (width, 0))
        combined_image.paste(Image.fromarray(im_image5), (width * 2, 0))
        combined_image.paste(Image.fromarray(im_image1), (width * 3, 0))
        combined_image.paste(Image.fromarray(im_image2), (width * 4, 0))
        combined_image.paste(Image.fromarray(im_image3), (width * 5, 0))
        combined_image.paste(Image.fromarray(im_image4), (width * 6, 0))
        combined_image.paste(Image.fromarray(im_label), (0, height))
        combined_image.paste(Image.fromarray(im_image6), (width, height))
        combined_image.paste(Image.fromarray(im_image5), (width * 2, height))
        combined_image.paste(Image.fromarray(hm1), (width * 3, height))
        combined_image.paste(Image.fromarray(hm2), (width * 4, height))
        combined_image.paste(Image.fromarray(hm3), (width * 5, height))
        combined_image.paste(Image.fromarray(hm4), (width * 6, height))

        return combined_image

    def calculate_mae(self, image1, image2):

        absolute_error = np.abs(image1 - image2)
        mae = np.mean(absolute_error)*100

        heatmap_image = cv2.applyColorMap(absolute_error.astype(np.uint8), cv2.COLORMAP_JET)

        heatmap_array = cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB)

        return heatmap_array

    def save_combined_image(self, combined_image, save_path, name):
        combined_image.save(os.path.join(save_path, name.split("/")[-1]).replace('.jpg', '.png'))



class results_saver_mid_training():
    def __init__(self, opt,current_iteration):
        path = os.path.join(opt.results_dir, opt.name, current_iteration)
        self.path_label = os.path.join(path, "label")
        self.path_image = os.path.join(path, "image")
        self.path_to_save = {"label": self.path_label, "image": self.path_image}
        os.makedirs(self.path_label, exist_ok=True)
        os.makedirs(self.path_image, exist_ok=True)
        self.num_cl = opt.label_nc + 2

    def __call__(self, label, generated, name):
        assert len(label) == len(generated)
        for i in range(len(label)):
            im = tens_to_lab(label[i], self.num_cl)
            self.save_im(im, "label", name[i])
            im = tens_to_im(generated[i]) * 255
            self.save_im(im, "image", name[i])

    def save_im(self, im, mode, name):
        im = Image.fromarray(im.astype(np.uint8))
        im.save(os.path.join(self.path_to_save[mode], name.split("/")[-1]).replace('.jpg', '.png'))

class timer():
    def __init__(self, opt):
        self.prev_time = time.time()
        self.prev_epoch = 0
        self.num_epochs = opt.num_epochs
        self.file_name = os.path.join(opt.checkpoints_dir, opt.name, "progress.txt")

    def __call__(self, epoch, cur_iter):
        if cur_iter != 0:
            avg = (time.time() - self.prev_time) / (cur_iter - self.prev_epoch)
        else:
            avg = 0
        self.prev_time = time.time()
        self.prev_epoch = cur_iter

        with open(self.file_name, "a") as log_file:
            log_file.write('[epoch %d/%d - iter %d], time:%.3f \n' % (epoch, self.num_epochs, cur_iter, avg))
        print('[epoch %d/%d - iter %d], time:%.3f' % (epoch, self.num_epochs, cur_iter, avg))
        return avg


class losses_saver():
    def __init__(self, opt):
        self.name_list = ["recon_loss",
                          'aeloss',
                          'perceptual_loss',
                          'gan_feat_loss',
                          'd_image_loss',
                          'd_video_loss'
                          ]
        self.opt = opt
        self.freq_smooth_loss = opt.freq_smooth_loss
        self.freq_save_loss = opt.freq_save_loss
        self.losses = dict()
        self.cur_estimates = np.zeros(len(self.name_list))
        print(len(self.name_list))
        self.path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses")
        self.is_first = True
        os.makedirs(self.path, exist_ok=True)
        for name in self.name_list:
            if opt.continue_train:
                self.losses[name] = np.load(self.path+"/losses.npy", allow_pickle = True).item()[name]
            else:
                self.losses[name] = list()

    def __call__(self, epoch, losses):
        for i, loss in enumerate(losses):
            if loss is None:
                self.cur_estimates[i] = None
            else:
                self.cur_estimates[i] += loss.detach().cpu().numpy()
        if epoch % self.freq_smooth_loss == self.freq_smooth_loss-1:
            for i, loss in enumerate(losses):
                if not self.cur_estimates[i] is None:
                    self.losses[self.name_list[i]].append(self.cur_estimates[i]/self.opt.freq_smooth_loss)
                    self.cur_estimates[i] = 0
        if epoch % self.freq_save_loss == self.freq_save_loss-1:
            self.plot_losses()
            np.save(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", "losses"), self.losses)

    def plot_losses(self):
        for curve in self.losses:
            fig,ax = plt.subplots(1)
            n = np.array(range(len(self.losses[curve])))*self.opt.freq_smooth_loss
            plt.plot(n[1:], self.losses[curve][1:])
            plt.ylabel('loss')
            plt.xlabel('epochs')

            plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", '%s.png' % (curve)),  dpi=600)
            plt.close(fig)

        fig,ax = plt.subplots(1)
        for curve in self.losses:
            if np.isnan(self.losses[curve][0]):
                continue
            plt.plot(n[1:], self.losses[curve][1:], label=curve)
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", 'combined.png'), dpi=600)
        plt.close(fig)


def save_networks(opt, cur_iter, model, latest=False, best=False):
    path = os.path.join(opt.checkpoints_dir, opt.name, "models")
    os.makedirs(path, exist_ok=True)
    if latest:
        try:
            torch.save(model.module.encoder.state_dict(), path + '/%s_Encoder.pth' % ("latest"))
        except:
            print('fail to save encoder')

        try:
            torch.save(model.module.decoder.state_dict(), path+'/%s_Decoder.pth' % ("latest"))
        except:
            print('fail to save ')

        try:
            torch.save(model.module.codebook.state_dict(), path+'/%s_Codebook.pth' % ("latest"))

        except:
            print('fail to save codebook ')

        try:
            torch.save(model.module.image_discriminator.state_dict(), path + '/%s_Image_D.pth' % ("latest"))
            torch.save(model.module.video_discriminator.state_dict(), path + '/%s_Video_D.pth' % ("latest"))
        except:
            print('fail to save discriminator')

        with open(os.path.join(opt.checkpoints_dir, opt.name)+"/latest_iter.txt", "w") as f:
            f.write(str(cur_iter))
    elif best:
        try:
            torch.save(model.module.encoder.state_dict(), path + '/%s_Encoder.pth' % ("best"))
        except:
            print('fail to save encoder')

        try:
            torch.save(model.module.decoder.state_dict(), path + '/%s_Decoder.pth' % ("best"))
        except:
            print('fail to save ')

        try:
            torch.save(model.module.codebook.state_dict(), path + '/%s_Codebook.pth' % ("best"))

        except:
            print('fail to save codebook ')

        try:
            torch.save(model.module.image_discriminator.state_dict(), path + '/%s_image_D.pth' % ("best"))
            torch.save(model.module.video_discriminator.state_dict(), path + '/%s_video_D.pth' % ("best"))
        except:
            print('fail to save discriminator')

        with open(os.path.join(opt.checkpoints_dir, opt.name)+"/best_iter.txt", "w") as f:
            f.write(str(cur_iter))


class image_saver():
    def __init__(self, opt):
        self.cols = 4
        self.rows = 3
        self.grid = 5
        self.path = os.path.join(opt.checkpoints_dir, opt.name, "images")+"/"
        self.opt = opt
        self.num_cl = opt.label_nc + 2
        os.makedirs(self.path, exist_ok=True)

    def visualize_batch(self, image, recon, cur_iter):
        self.save_images(image, "real", cur_iter)
        self.save_images(recon, "recon", cur_iter)

    def save_images(self, batch, name, cur_iter, is_label=False):
        fig = plt.figure()
        for i in range(min(self.rows * self.cols, len(batch))):
            if is_label:
                im = tens_to_lab_color(batch[i], self.num_cl)
            else:
                im = tens_to_im(batch[i])
            plt.axis("off")
            fig.add_subplot(self.rows, self.cols, i+1)
            plt.axis("off")
            plt.imshow(im)
        fig.tight_layout()
        plt.savefig(self.path+str(cur_iter)+"_"+name)
        plt.close()


def tens_to_im(tens):
    out = (tens + 1) / 2
    out.clamp(0, 1)
    return np.transpose(out.detach().cpu().numpy(), (1, 2, 0))


def tens_to_lab(tens, num_cl):
    label_tensor = GreyScale(tens, num_cl)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy

def tens_to_lab_color(tens, num_cl):
    label_tensor = Colorize(tens, num_cl)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy

###############################################################################
# Code below from
# https://github.com/visinf/1-stage-wseg/blob/38130fee2102d3a140f74a45eec46063fcbeaaf8/datasets/utils.py
# Modified so it complies with the Cityscapes label map colors (fct labelcolormap)
###############################################################################


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def GreyScale(tens, num_cl):
    cmap = labelcolormap(num_cl)
    cmap = torch.from_numpy(cmap[:num_cl])
    size = tens.size()
    color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
    tens = torch.argmax(tens, dim=0, keepdim=True)

    for label in range(0, len(cmap)):
        mask = (label == tens[0]).cpu()
        color_image[0][mask] = label
        color_image[1][mask] = label
        color_image[2][mask] = label
    return color_image


def Colorize(tens, num_cl):
    cmap = labelcolormap(num_cl)
    cmap = torch.from_numpy(cmap[:num_cl])
    size = tens.size()
    color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
    tens = torch.argmax(tens, dim=0, keepdim=True)

    for label in range(0, len(cmap)):
        mask = (label == tens[0]).cpu()
        color_image[0][mask] = cmap[label][0]
        color_image[1][mask] = cmap[label][1]
        color_image[2][mask] = cmap[label][2]
    return color_image


def labelcolormap(N):
    if N == 39 or N == 33:
        cmap = np.array([(0, 0, 0), (111, 74, 0), (81, 0, 81), (50, 80, 100), (0, 100, 230), (119, 60, 50), (70, 40, 142),
                  (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156),
                  (190, 153, 153),
                  (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30),
                  (220, 220, 0),
                  (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                  (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142),
                  (150, 250, 90), (0, 153, 140), (119, 11, 32), (0, 0, 142), (150, 250, 90), (0, 153, 140)],
                 dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


