"""Adapted from https://github.com/SongweiGe/TATS"""
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from vq_gan_3d.sync_batchnorm import DataParallelWithCallback
from vq_gan_3d.utils import shift_dim, adopt_weight, comp_getattr
from vq_gan_3d.model.lpips import LPIPS
from vq_gan_3d.model.codebook import Codebook


def silu(x):
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
            torch.mean(torch.nn.functional.softplus(-logits_real)) +
            torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def put_on_multi_gpus(model, opt):
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        model = DataParallelWithCallback(model, device_ids=gpus).cuda()
    else:
        model.module = model
    assert len(opt.gpu_ids.split(",")) == 0 or opt.batch_size % len(opt.gpu_ids.split(",")) == 0
    return model


class VQGAN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.opt.downsample = [4, 4, 4]
        self.opt.upsample = [4, 4, 4]
        self.embedding_dim = opt.embedding_dim
        self.n_codes = self.opt.n_codes
        self.opt.enc_out_ch = self.opt.n_hiddens * 2 ** (max(self.opt.downsample))
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)

        self.codebook = Codebook(self.opt.n_codes, self.opt.embedding_dim,
                                 no_random_restart=self.opt.no_random_restart, restart_thres=self.opt.restart_thres)

        self.gan_feat_weight = self.opt.gan_feat_weight
        # TODO: Changed batchnorm from sync to normal
        self.image_discriminator = NLayerDiscriminator(
            self.opt.image_channel, self.opt.disc_channels, self.opt.disc_layers, norm_layer=nn.BatchNorm2d)
        self.video_discriminator = NLayerDiscriminator3D(
            self.opt.image_channel, self.opt.disc_channels, self.opt.disc_layers, norm_layer=nn.BatchNorm3d)

        if self.opt.disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif self.opt.disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss

        self.perceptual_model = LPIPS().eval()

        self.image_gan_weight = self.opt.image_gan_weight
        self.video_gan_weight = self.opt.video_gan_weight

        self.perceptual_weight = self.opt.perceptual_weight

        self.l1_weight = self.opt.l1_weight
        self.load_checkpoints()

    def encode(self, x, include_embeddings=False, quantize=True):
        h = self.encoder(x)
        if quantize:
            vq_output = self.codebook(h)
            if include_embeddings:
                return vq_output['embeddings'], vq_output['encodings']
            else:
                return vq_output['encodings']
        return h

    def decode(self, latent, quantize=False):
        if quantize:
            vq_output = self.codebook(latent)
            latent = vq_output['encodings']
        h = F.embedding(latent, self.codebook.embeddings)
        h = shift_dim(h, -1, 1)
        return self.decoder(h)

    def forward(self, x, mode, log_image=False):

        B, C, T, H, W = x.shape
        z = self.encoder(x)
        vq_output = self.codebook(z)
        x_recon = self.decoder(vq_output['embeddings'])

        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight

        # Selects one random 2D image from each 3D Image
        frame_idx = torch.randint(0, T, [B]).cuda()
        frame_idx_selected = frame_idx.reshape(-1,
                                               1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        if log_image:
            return frames, frames_recon, x, x_recon

        if mode == 'losses_G':
            # Autoencoder - train the "generator"
            # Perceptual loss
            perceptual_loss = 0
            if self.perceptual_weight > 0:
                perceptual_loss = self.perceptual_model(
                    frames, frames_recon).mean() * self.perceptual_weight

            # Discriminator loss (turned on after a certain epoch)
            logits_image_fake, pred_image_fake = self.image_discriminator(
                frames_recon)
            logits_video_fake, pred_video_fake = self.video_discriminator(
                x_recon)
            g_image_loss = -torch.mean(logits_image_fake)
            g_video_loss = -torch.mean(logits_video_fake)
            g_loss = self.image_gan_weight * g_image_loss + self.video_gan_weight * g_video_loss
            disc_factor = adopt_weight(
                self.global_step, threshold=self.opt.discriminator_iter_start)
            aeloss = disc_factor * g_loss

            # GAN feature matching loss - tune features such that we get the same prediction result on the discriminator
            image_gan_feat_loss = 0
            video_gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1)
            if self.image_gan_weight > 0:
                logits_image_real, pred_image_real = self.image_discriminator(
                    frames)
                for i in range(len(pred_image_fake) - 1):
                    image_gan_feat_loss += feat_weights * \
                                           F.l1_loss(pred_image_fake[i], pred_image_real[i].detach(
                                           )) * (self.image_gan_weight > 0)
            if self.video_gan_weight > 0:
                logits_video_real, pred_video_real = self.video_discriminator(
                    x)
                for i in range(len(pred_video_fake) - 1):
                    video_gan_feat_loss += feat_weights * \
                                           F.l1_loss(pred_video_fake[i], pred_video_real[i].detach(
                                           )) * (self.video_gan_weight > 0)
            gan_feat_loss = disc_factor * self.gan_feat_weight * (image_gan_feat_loss + video_gan_feat_loss)

            commitment_loss = vq_output['commitment_loss']

            self.log("train/g_image_loss", g_image_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/g_video_loss", g_video_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/image_gan_feat_loss", image_gan_feat_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/video_gan_feat_loss", video_gan_feat_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/perceptual_loss", perceptual_loss,
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/recon_loss", recon_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/aeloss", aeloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/commitment_loss", vq_output['commitment_loss'],
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log('train/perplexity', vq_output['perplexity'],
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)

            loss_G = recon_loss + aeloss + perceptual_loss + gan_feat_loss + commitment_loss

            return loss_G, [recon_loss,  aeloss, perceptual_loss, gan_feat_loss], frames, frames_recon

        if mode == 'losses_D':
            # Train discriminator
            logits_image_real, _ = self.image_discriminator(frames.detach())
            logits_video_real, _ = self.video_discriminator(x.detach())

            logits_image_fake, _ = self.image_discriminator(
                frames_recon.detach())
            logits_video_fake, _ = self.video_discriminator(x_recon.detach())

            d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
            d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
            disc_factor = adopt_weight(
                self.global_step, threshold=self.opt.discriminator_iter_start)
            discloss = disc_factor * \
                       (self.image_gan_weight * d_image_loss +
                        self.video_gan_weight * d_video_loss)

            self.log("train/logits_image_real", logits_image_real.mean().detach(),
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_image_fake", logits_image_fake.mean().detach(),
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_video_real", logits_video_real.mean().detach(),
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_video_fake", logits_video_fake.mean().detach(),
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/d_image_loss", d_image_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/d_video_loss", d_video_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/discloss", discloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)

            return discloss, [d_image_loss, d_video_loss]

    def training_step(self, batch,  optimizer_idx):
        x = batch['data']
        if optimizer_idx == 0:
            recon_loss, _, vq_output, aeloss, perceptual_loss, gan_feat_loss = self.forward(
                x, optimizer_idx)
            commitment_loss = vq_output['commitment_loss']
            loss = recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss
        if optimizer_idx == 1:
            discloss = self.forward(x, optimizer_idx)
            loss = discloss
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']  # TODO: batch['stft']
        recon_loss, _, vq_output, perceptual_loss = self.forward(x)
        self.log('val/recon_loss', recon_loss, prog_bar=True)
        self.log('val/perceptual_loss', perceptual_loss, prog_bar=True)
        self.log('val/perplexity', vq_output['perplexity'], prog_bar=True)
        self.log('val/commitment_loss',
                 vq_output['commitment_loss'], prog_bar=True)

    def configure_optimizers(self):
        lr = self.opt.lr
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  #list(self.pre_vq_conv.parameters()) +
                                  #list(self.post_vq_conv.parameters()) +
                                  list(self.codebook.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(list(self.image_discriminator.parameters()) +
                                    list(self.video_discriminator.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch['data']
        x = x.to(self.device)
        frames, frames_rec, _, _ = self(x, log_image=True)
        log["inputs"] = frames
        log["reconstructions"] = frames_rec
        # log['mean_org'] = batch['mean_org']
        # log['std_org'] = batch['std_org']
        return log

    def log_videos(self, batch, **kwargs):
        log = dict()
        x = batch['data']
        _, _, x, x_rec = self(x, log_image=True)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        # log['mean_org'] = batch['mean_org']
        # log['std_org'] = batch['std_org']
        return log

    def load_checkpoints(self):
        if self.opt.phase == "test":
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", "best_")
            self.encoder.load_state_dict(torch.load(path + "Encoder.pth"))
            self.decoder.load_state_dict(torch.load(path + "Decoder.pth"))
            self.codebook.load_state_dict(torch.load(path + "codebook.pth"))

        elif self.opt.continue_train:
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", "latest_")
            try:
                self.encoder.load_state_dict(torch.load(path + "Encoder.pth"))
                print('Encoder successfully loaded')
            except:
                print('Encoder.pth not found', path + "G.pth")
            try:
                self.decoder.load_state_dict(torch.load(path + "Decoder.pth"))
                print('Decoder successfully loaded')
            except:
                print('Decoder.pth not found', path + "G.pth")

            try:
                self.codebook.load_state_dict(torch.load(path + "Codebook.pth"))
                print('Codebook successfully loaded')
            except:
                print('Codebook.pth not found', path + "G.pth")

            try:
                self.image_discriminator.load_state_dict(torch.load(path + "Image_D.pth"))
                print('image_discriminator successfully loaded')
            except:
                print('Image_Discriminator.pth not found', path + "Image_D.pth")

            try:
                self.video_discriminator.load_state_dict(torch.load(path + "Video_D.pth"))
                print('video_discriminator successfully loaded')
            except:
                print('Video_Discriminator.pth not found', path + "Video_D.pth")


def Normalize(in_channels, norm_type='group', num_groups=32):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        # TODO Changed num_groups from 32 to 8
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)


class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        n_times_downsample = np.array([int(math.log2(d)) for d in self.opt.downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = max(n_times_downsample)  # .max()
        self.pre_vq_conv = SamePadConv3d(
            self.opt.enc_out_ch, self.opt.embedding_dim, 1, padding_type=self.opt.padding_type)
        self.conv_first = SamePadConv3d(
            self.opt.image_channel, self.opt.n_hiddens, kernel_size=3, padding_type=self.opt.padding_type)

        for i in range(max_ds):
            block = nn.Module()
            in_channels = self.opt.n_hiddens * 2 ** i
            out_channels = self.opt.n_hiddens * 2 ** (i + 1)
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            block.down = SamePadConv3d(
                in_channels, out_channels, 4, stride=stride, padding_type=self.opt.padding_type)
            block.res = ResBlock(
                out_channels, out_channels, norm_type=self.opt.norm_type, num_groups=self.opt.num_groups)
            self.conv_blocks.append(block)
            n_times_downsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, self.opt.norm_type, num_groups=self.opt.num_groups),
            SiLU()
        )

        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)
        h = self.pre_vq_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt= opt
        n_times_upsample = np.array([int(math.log2(d)) for d in self.opt.upsample])
        max_us = max(n_times_upsample)  # .max()

        in_channels = self.opt.n_hiddens * 2 ** max_us
        self.final_block = nn.Sequential(
            Normalize(in_channels, self.opt.norm_type, num_groups=self.opt.num_groups),
            SiLU()
        )
        self.enc_out_ch = self.opt.enc_out_ch
        self.conv_blocks = nn.ModuleList()
        for i in range(max_us):
            block = nn.Module()
            in_channels = in_channels if i == 0 else self.opt.n_hiddens * 2 ** (max_us - i + 1)
            out_channels = self.opt.n_hiddens * 2 ** (max_us - i)
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            block.up = SamePadConvTranspose3d(
                in_channels, out_channels, 4, stride=us)
            block.res1 = ResBlock(
                out_channels, out_channels, norm_type=self.opt.norm_type, num_groups=self.opt.num_groups)
            block.res2 = ResBlock(
                out_channels, out_channels, norm_type=self.opt.norm_type, num_groups=self.opt.num_groups)
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.post_vq_conv = SamePadConv3d(
            self.opt.embedding_dim, self.enc_out_ch, 1)
        self.conv_last = SamePadConv3d(
            out_channels, self.opt.image_channel, kernel_size=3)

    def forward(self, x):
        h = self.post_vq_conv(x)
        h = self.final_block(h)
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_last(h)
        return h


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group',
                 padding_type='replicate', num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = SamePadConv3d(
            in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv2 = SamePadConv3d(
            out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(
                in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x + h


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3  # (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim, step = -1
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False,
                 getIntermFeat=True):
        # def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input)


class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False,
                 getIntermFeat=True):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input)
