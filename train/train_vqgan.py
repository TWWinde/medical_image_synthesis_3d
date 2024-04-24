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
                              list(model.module.pre_vq_conv.parameters()) +
                              list(model.module.post_vq_conv.parameters()) +
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
        # image.shape torch.Size([2, 3, 256, 256])
        # label.shape torch.Size([2, 38, 256, 256])
        loss_G, losses_G_list = model(data, "losses_G")
        loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]
        loss_G.backward()
        optimizerG.step()

        # --- discriminator update ---#
        model.module.netDu.zero_grad()
        loss_D = model(data, "losses_D")
        loss_D.backward()
        optimizerD.step()

        # --- stats update ---#

        if not opt.no_EMA:
            utils.update_EMA(model, cur_iter, dataloader, opt)
        if cur_iter % opt.freq_print == 0:
            im_saver.visualize_batch(model, image, label, cur_iter)
            timer(epoch, cur_iter)
        if cur_iter % opt.freq_save_latest == 0:
            utils.save_networks(opt, cur_iter, model, latest=True)
        if cur_iter % opt.freq_fid == 0 and cur_iter > 0:
            is_best = fid_computer.update(model, cur_iter)
            metrics_computer.update_metrics(model, cur_iter)
            if is_best:
                utils.save_networks(opt, cur_iter, model, best=True)
        visualizer_losses(cur_iter, losses_G_list+losses_S_list+losses_Du_list+losses_reg_list)

# --- after training ---#
utils.update_EMA(model, cur_iter, dataloader, opt, force_run_stats=True)
utils.save_networks(opt, cur_iter, model)
utils.save_networks(opt, cur_iter, model, latest=True)
is_best = fid_computer.update(model, cur_iter)
metrics_computer.update_metrics(model, cur_iter)
if is_best:
    utils.save_networks(opt, cur_iter, model, best=True)

print("The training has successfully finished")
def run(opt):

    train_dataset, val_dataset, sampler = get_dataset(opt)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size,
                                  num_workers=opt.num_workers, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size,
                                shuffle=False, num_workers=opt.num_workers)

    # automatically adjust learning rate
    bs, base_lr, ngpu, accumulate = cfg.model.batch_size, cfg.model.lr, cfg.model.gpus, cfg.model.accumulate_grad_batches

    with open_dict(cfg):
        cfg.model.lr = accumulate * (ngpu / 8.) * (bs / 4.) * base_lr
        cfg.model.default_root_dir = os.path.join(
            cfg.model.default_root_dir, cfg.dataset.name, cfg.model.default_root_dir_postfix)
    print(
        "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(
            cfg.model.lr, accumulate, ngpu / 8, bs / 4, base_lr))

    model = VQGAN(cfg)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                                     save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000,
                                     save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1,
                                     filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    callbacks.append(ImageLogger(
        batch_frequency=750, max_images=4, clamp=True))
    callbacks.append(VideoLogger(
        batch_frequency=1500, max_videos=4, clamp=True))

    # load the most recent checkpoint file
    base_dir = os.path.join(cfg.model.default_root_dir, 'lightning_logs')
    if os.path.exists(base_dir):
        log_folder = ckpt_file = ''
        version_id_used = step_used = 0
        for folder in os.listdir(base_dir):
            version_id = int(folder.split('_')[1])
            if version_id > version_id_used:
                version_id_used = version_id
                log_folder = folder
        if len(log_folder) > 0:
            ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            for fn in os.listdir(ckpt_folder):
                if fn == 'latest_checkpoint.ckpt':
                    ckpt_file = 'latest_checkpoint_prev.ckpt'
                    os.rename(os.path.join(ckpt_folder, fn),
                              os.path.join(ckpt_folder, ckpt_file))
            if len(ckpt_file) > 0:
                cfg.model.resume_from_checkpoint = os.path.join(
                    ckpt_folder, ckpt_file)
                print('will start from the recent ckpt %s' %
                      cfg.model.resume_from_checkpoint)

    accelerator = None
    if cfg.model.gpus > 1:
        accelerator = 'ddp'


    trainer = pl.Trainer(
        gpus=2,#cfg.model.gpus,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        default_root_dir=cfg.model.default_root_dir,
        resume_from_checkpoint=cfg.model.resume_from_checkpoint,
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        gradient_clip_val=cfg.model.gradient_clip_val,
        accelerator=accelerator,
    )

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    run()
