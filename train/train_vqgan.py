"Adapted from https://github.com/SongweiGe/TATS"
import sys

from vq_gan_3d.model.vqgan import put_on_multi_gpus

sys.path.append('/misc/no_backups/s1449/medical_image_synthesis_3d')
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
# from ddpm.diffusion import default
from vq_gan_3d.model import VQGAN
from train.callbacks import ImageLogger, VideoLogger
from train.get_dataset import get_dataset
import hydra
from omegaconf import DictConfig, open_dict
from vq_gan_3d.sync_batchnorm import DataParallelWithCallback
import torch
import util as utils
import config_set as config
# --- read options ---#

opt = config.read_arguments(train=True)
print("nb of gpus: ", torch.cuda.device_count())

# --- create utils ---#
timer = utils.timer(opt)
visualizer_losses = utils.losses_saver(opt)
im_saver = utils.image_saver(opt)
train_dataset, val_dataset, sampler = get_dataset(opt)
dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size,
                              num_workers=opt.num_workers, sampler=sampler)
val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size,
                            shuffle=False, num_workers=opt.num_workers)

model = VQGAN(opt)
model = put_on_multi_gpus(model, opt)

lr = opt.lr
optimizerG = torch.optim.Adam(list(model.module.encoder.parameters()) +
                              list(model.module.decoder.parameters()) +
                              list(model.module.codebook.parameters()),
                              lr=lr, betas=(0.5, 0.9))
optimizerD = torch.optim.Adam(list(model.module.image_discriminator.parameters()) +
                                list(model.module.video_discriminator.parameters()),
                                lr=lr, betas=(0.5, 0.9))


def loopy_iter(dataset):
    while True:
        for item in dataset:
            yield item

# --- the training loop ---#
already_started = True
start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader))

for epoch in range(start_epoch, opt.num_epochs):
    for i, data in enumerate(dataloader):
        if not already_started and i < start_iter:
            continue
        already_started = True
        cur_iter = epoch*len(dataloader) + i

        # --- generator unconditional update ---#
        model.module.netG.zero_grad()
        loss_G, losses_G_list, frames, frames_recon = model(data, "losses_G")
        loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]
        loss_G.backward()
        optimizerG.step()

        # --- discriminator update ---#
        model.module.netDu.zero_grad()
        loss_D, losses_D_list = model(data, "losses_D")
        loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]
        loss_D.backward()
        optimizerD.step()

        # --- stats update ---#

        if cur_iter % opt.freq_print == 0:
            im_saver.visualize_batch(frames, frames_recon,  cur_iter)
            timer(epoch, cur_iter)
        if cur_iter % opt.freq_save_latest == 0:
            utils.save_networks(opt, cur_iter, model, latest=True)
        if cur_iter % opt.freq_fid == 0 and cur_iter > 0:
            #is_best = fid_computer.update(model, cur_iter)
            #metrics_computer.update_metrics(model, cur_iter)
            #if #is_best:
            utils.save_networks(opt, cur_iter, model, best=True)
        visualizer_losses(cur_iter, losses_G_list+losses_D_list)

# --- after training ---#
utils.save_networks(opt, cur_iter, model)
utils.save_networks(opt, cur_iter, model, latest=True)
#is_best = fid_computer.update(model, cur_iter)
#metrics_computer.update_metrics(model, cur_iter)
#if is_best:
    #utils.save_networks(opt, cur_iter, model, best=True)

print("The training has successfully finished")



if __name__ == '__main__':
    run()
