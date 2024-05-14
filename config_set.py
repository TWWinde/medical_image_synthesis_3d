import argparse
import pickle
import os
import train.util as utils


def read_arguments(train=True):
    parser = argparse.ArgumentParser()
    parser = add_all_arguments(parser, train)
    parser.add_argument('--phase', type=str, default='train')
    opt = parser.parse_args()
    if train:
        if opt.continue_train:
            update_options_from_file(opt, parser)
    opt = parser.parse_args()
    opt.phase = 'train' if train else 'test'
    if train:
        opt.loaded_latest_iter = 0 if not opt.continue_train else load_iter(opt)
    utils.fix_seed(opt.seed)
    print_options(opt, parser)
    if train:
        save_options(opt, parser)
    return opt


def add_all_arguments(parser, train):
    #--- train VQ-GAN---
    parser.add_argument('--name', type=str, default='VQ-GAN', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--dataset_mode', type=str, default='SynthRAD2023', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--num_workers', type=int, default=30, help='random seed')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
    parser.add_argument('--root_dir', type=str, default='/misc/data/private/autoPET/Task1/pelvis/', help='path to dataset root')
    parser.add_argument('--checkpoints_dir', type=str, default='/misc/no_backups/d1502/medical_image_synthesis_3d/checkpoints', help='path to dataset root')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
    parser.add_argument('--no_netDu', action='store_true', help='if specified, no undconditional discriminator')
    parser.add_argument('--single_image', action='store_true', help='if specified, only train single image')
    parser.add_argument('--embedding_dim', type=int, default=256, help='if specified, only train single image')
    parser.add_argument('--n_codes', type=int, default=2048, help='if specified, only train single image')
    parser.add_argument('--n_hiddens', type=int, default=240, help='if specified, only train single image')
    parser.add_argument('--lr', type=float, default=0.0003, help='if specified, only train single image')
    parser.add_argument('--disc_channels', type=int, default=64, help='if specified, only train single image')
    parser.add_argument('--disc_layers', type=int, default=3, help='if specified, only train single image')
    parser.add_argument('--discriminator_iter_start', type=int, default=50000, help='if specified, only train single image')
    parser.add_argument('--disc_loss_type', type=str, default='hinge', help='name of the experiment.')
    parser.add_argument('--image_gan_weight', type=int, default=1.0, help='if specified, only train single image')
    parser.add_argument('--video_gan_weight', type=int, default=1.0, help='if specified, only train single image')
    parser.add_argument('--gan_feat_weight', type=int, default=0.0, help='if specified, only train single image')
    parser.add_argument('--l1_weight', type=int, default=4.0, help='if specified, only train single image')
    parser.add_argument('--perceptual_weight', type=int, default=0.0, help='if specified, only train single image')
    parser.add_argument('--restart_thres', type=int, default=1.0, help='if specified, only train single image')
    parser.add_argument('--norm_type', type=str, default='group', help='name of the experiment.')
    parser.add_argument('--padding_type', type=str, default='replicate', help='name of the experiment.')
    parser.add_argument('--num_groups', type=int, default=32, help='if specified, only train single image')
    parser.add_argument('--i3d_feat', action='store_true', help='if specified, only train single image')
    parser.add_argument('--no_random_restart', action='store_true', help='if specified, only train single image')
    parser.add_argument('--image_channel', type=int, default=1, help='if specified, only train single image')
    parser.add_argument('--num_res_blocks', type=int, default=6, help='number of residual blocks in G and D')
    parser.add_argument('--channels_G', type=int, default=64, help='# of gen filters in first conv layer in generator')
    parser.add_argument('--param_free_norm', type=str, default='syncbatch', help='which norm to use in generator before SPADE')
    parser.add_argument('--spade_ks', type=int, default=3, help='kernel size of convs inside SPADE')
    parser.add_argument('--no_EMA', action='store_true', help='if specified, do *not* compute exponential moving averages')
    parser.add_argument('--EMA_decay', type=float, default=0.9999, help='decay in exponential moving averages')
    parser.add_argument('--no_3dnoise', action='store_true', default=False, help='if specified, do *not* concatenate noise to label maps')
    parser.add_argument('--z_dim', type=int, default=64, help="dimension of the latent z vector")
    parser.add_argument('--progressive_growing', action='store_true', help="progressive model or normal")
    parser.add_argument('--netG', type=str, default="default", help="generator architecture")
    parser.add_argument('--mixed_images', action='store_true', help='mix images for compeletey unpaired training')
    parser.add_argument('--model_supervision', type=int, default=0,help='0 unsupervised, 1 semi-supervised, 2 supervised')
    parser.add_argument('--lambda_segment',type=int, default= 1,help ='weight of the segmentation loss for the generator')
    parser.add_argument('--generate_seg', action='store_true', help='if specified, generate output of segmentator')
    parser.add_argument('--trunc_normal', action='store_true', help='if specified, sample noise from truncated normal ditribution during test')

    if train:
        parser.add_argument('--freq_print', type=int, default=1000, help='frequency of showing training results')
        parser.add_argument('--freq_save_ckpt', type=int, default=20000, help='frequency of saving the checkpoints')
        parser.add_argument('--freq_save_latest', type=int, default=10000, help='frequency of saving the latest model')
        parser.add_argument('--freq_smooth_loss', type=int, default=250, help='smoothing window for loss visualization')
        parser.add_argument('--freq_save_loss', type=int, default=2500, help='frequency of loss plot updates')
        parser.add_argument('--freq_fid', type=int, default=2500, help='frequency of saving the fid score (in training iterations)')
        parser.add_argument('--continue_train', action='store_true', help='resume previously interrupted training')
        parser.add_argument('--which_iter', type=str, default='latest', help='which epoch to load when continue_train')
        parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train')

    else:
        parser.add_argument('--results_dir', type=str, default='/misc/no_backups/d1502/medical_image_synthesis_3d'
                                                               '/results/', help='saves testing results here.')
        parser.add_argument('--ckpt_iter', type=str, default='best', help='which epoch to load to evaluate a model')
    return parser


def save_options(opt, parser):
    path_name = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(path_name, exist_ok=True)
    with open(path_name + '/opt.txt', 'wt') as opt_file:
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

    with open(path_name + '/opt.pkl', 'wb') as opt_file:
        pickle.dump(opt, opt_file)


def update_options_from_file(opt, parser):
    new_opt = load_options(opt)
    for k, v in sorted(vars(opt).items()):
        if hasattr(new_opt, k) and v != getattr(new_opt, k):
            new_val = getattr(new_opt, k)
            parser.set_defaults(**{k: new_val})
    return parser


def load_options(opt):
    file_name = os.path.join(opt.checkpoints_dir, opt.name, "opt.pkl")
    new_opt = pickle.load(open(file_name, 'rb'))
    return new_opt


def load_iter(opt):
    if opt.which_iter == "latest":
        with open(os.path.join(opt.checkpoints_dir, opt.name, "latest_iter.txt"), "r") as f:
            res = int(f.read())
            return res
    elif opt.which_iter == "best":
        with open(os.path.join(opt.checkpoints_dir, opt.name, "best_iter.txt"), "r") as f:
            res = int(f.read())
            return res
    else:
        return int(opt.which_iter)


def print_options(opt, parser):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
