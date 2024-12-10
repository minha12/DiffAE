from math import exp
import os
import shutil
from typing import List

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from pytorch_fid import fid_score
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.autonotebook import tqdm, trange

# from renderer import *
from sampling import render_condition, render_uncondition
from modules import BeatGANsAutoencModel

# from config import *
from general_config import TrainConfig, ModelType
from multiprocessing import get_context
from diffusion import Sampler

# from dist_utils import *
import lpips
from torch.utils.data import Dataset

# from dataset import SubsetDataset
from modules import Model


class SubsetDataset(Dataset):
    def __init__(self, dataset, size):
        assert len(dataset) >= size
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        assert index < self.size
        return self.dataset[index]


def make_subset_loader(
    conf: TrainConfig,
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    parallel: bool,
    drop_last=True,
):
    dataset = SubsetDataset(dataset, size=conf.eval_num_images)
    if parallel and distributed.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        # with sampler, use the sample instead of this option
        shuffle=False if sampler else shuffle,
        num_workers=conf.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        multiprocessing_context=get_context("fork"),
    )


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def barrier():
    if distributed.is_initialized():
        distributed.barrier()
    else:
        pass


def all_gather(data: List, src):
    if distributed.is_initialized():
        distributed.all_gather(data, src)
    else:
        data[0] = src


def get_world_size():
    if distributed.is_initialized():
        return distributed.get_world_size()
    else:
        return 1


def evaluate_lpips(
    sampler: Sampler,
    model: Model,
    conf: TrainConfig,
    device,
    val_data: Dataset,
    latent_sampler: Sampler = None,
    use_inverted_noise: bool = False,
):
    """
    compare the generated images from autoencoder on validation dataset

    Args:
        use_inversed_noise: the noise is also inverted from DDIM
    """
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    val_loader = make_subset_loader(
        conf,
        dataset=val_data,
        batch_size=conf.batch_size_eval,
        shuffle=False,
        parallel=True,
    )

    model.eval()
    with torch.no_grad():
        scores = {
            "lpips": [],
            "mse": [],
            "ssim": [],
            "psnr": [],
        }
        for batch in tqdm(val_loader, desc="lpips"):
            imgs = batch["img"].to(device)

            if use_inverted_noise:
                # inverse the noise
                # with condition from the encoder
                model_kwargs = {}
                if conf.model_type.has_autoenc():
                    with torch.no_grad():
                        model_kwargs = model.encode(imgs)
                x_T = sampler.ddim_reverse_sample_loop(
                    model=model, x=imgs, clip_denoised=True, model_kwargs=model_kwargs
                )
                x_T = x_T["sample"]
            else:
                x_T = torch.randn(
                    (len(imgs), 3, conf.img_size, conf.img_size), device=device
                )

            if conf.model_type == ModelType.ddpm:
                # the case where you want to calculate the inversion capability of the DDIM model
                assert use_inverted_noise
                pred_imgs = render_uncondition(
                    conf=conf,
                    model=model,
                    x_T=x_T,
                    sampler=sampler,
                    latent_sampler=latent_sampler,
                )
            else:
                pred_imgs = render_condition(
                    conf=conf,
                    model=model,
                    x_T=x_T,
                    x_start=imgs,
                    cond=None,
                    sampler=sampler,
                )
            # # returns {'cond', 'cond2'}
            # conds = model.encode(imgs)
            # pred_imgs = sampler.sample(model=model,
            #                            noise=x_T,
            #                            model_kwargs=conds)

            # (n, 1, 1, 1) => (n, )
            scores["lpips"].append(lpips_fn.forward(imgs, pred_imgs).view(-1))

            # need to normalize into [0, 1]
            norm_imgs = (imgs + 1) / 2
            norm_pred_imgs = (pred_imgs + 1) / 2
            # (n, )
            scores["ssim"].append(ssim(norm_imgs, norm_pred_imgs, size_average=False))
            # (n, )
            scores["mse"].append(
                (norm_imgs - norm_pred_imgs).pow(2).mean(dim=[1, 2, 3])
            )
            # (n, )
            scores["psnr"].append(psnr(norm_imgs, norm_pred_imgs))
        # (N, )
        for key in scores.keys():
            scores[key] = torch.cat(scores[key]).float()
    model.train()

    barrier()

    # support multi-gpu
    outs = {
        key: [
            torch.zeros(len(scores[key]), device=device)
            for i in range(get_world_size())
        ]
        for key in scores.keys()
    }
    for key in scores.keys():
        all_gather(outs[key], scores[key])

    # final scores
    for key in scores.keys():
        scores[key] = torch.cat(outs[key]).mean().item()

    # {'lpips', 'mse', 'ssim'}
    return scores


def psnr(img1, img2):
    """
    Args:
        img1: (n, c, h, w)
    """
    v_max = 1.0
    # (n,)
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    return 20 * torch.log10(v_max / torch.sqrt(mse))


def broadcast(data, src):
    if distributed.is_initialized():
        distributed.broadcast(data, src)
    else:
        pass


def get_rank():
    if distributed.is_initialized():
        return distributed.get_rank()
    else:
        return 0


def chunk_size(size, rank, world_size):
    extra = rank < size % world_size
    return size // world_size + extra


def evaluate_fid(
    sampler: Sampler,
    model: Model,
    conf: TrainConfig,
    device,
    train_data: Dataset,
    val_data: Dataset,
    latent_sampler: Sampler = None,
    conds_mean=None,
    conds_std=None,
    remove_cache: bool = True,
    clip_latent_noise: bool = False,
):
    assert conf.fid_cache is not None
    if get_rank() == 0:
        # no parallel
        # validation data for a comparing FID
        val_loader = make_subset_loader(
            conf,
            dataset=val_data,
            batch_size=conf.batch_size_eval,
            shuffle=False,
            parallel=False,
        )

        # put the val images to a directory
        cache_dir = f"{conf.fid_cache}_{conf.eval_num_images}"
        if (
            os.path.exists(cache_dir)
            and len(os.listdir(cache_dir)) < conf.eval_num_images
        ):
            shutil.rmtree(cache_dir)

        if not os.path.exists(cache_dir):
            # write files to the cache
            # the images are normalized, hence need to denormalize first
            loader_to_path(val_loader, cache_dir, denormalize=True)

        # create the generate dir
        if os.path.exists(conf.generate_dir):
            shutil.rmtree(conf.generate_dir)
        os.makedirs(conf.generate_dir)

    barrier()

    world_size = get_world_size()
    rank = get_rank()
    batch_size = chunk_size(conf.batch_size_eval, rank, world_size)

    def filename(idx):
        return world_size * idx + rank

    model.eval()
    with torch.no_grad():
        if conf.model_type.can_sample():
            eval_num_images = chunk_size(conf.eval_num_images, rank, world_size)
            desc = "generating images"
            for i in trange(0, eval_num_images, batch_size, desc=desc):
                batch_size = min(batch_size, eval_num_images - i)
                x_T = torch.randn(
                    (batch_size, 3, conf.img_size, conf.img_size), device=device
                )
                batch_images = render_uncondition(
                    conf=conf,
                    model=model,
                    x_T=x_T,
                    sampler=sampler,
                    latent_sampler=latent_sampler,
                    conds_mean=conds_mean,
                    conds_std=conds_std,
                ).cpu()

                batch_images = (batch_images + 1) / 2
                # keep the generated images
                for j in range(len(batch_images)):
                    img_name = filename(i + j)
                    torchvision.utils.save_image(
                        batch_images[j],
                        os.path.join(conf.generate_dir, f"{img_name}.png"),
                    )
        elif conf.model_type == ModelType.autoencoder:
            if conf.train_mode.is_latent_diffusion():
                # evaluate autoencoder + latent diffusion (doesn't give the images)
                model: BeatGANsAutoencModel
                eval_num_images = chunk_size(conf.eval_num_images, rank, world_size)
                desc = "generating images"
                for i in trange(0, eval_num_images, batch_size, desc=desc):
                    batch_size = min(batch_size, eval_num_images - i)
                    x_T = torch.randn(
                        (batch_size, 3, conf.img_size, conf.img_size), device=device
                    )
                    batch_images = render_uncondition(
                        conf=conf,
                        model=model,
                        x_T=x_T,
                        sampler=sampler,
                        latent_sampler=latent_sampler,
                        conds_mean=conds_mean,
                        conds_std=conds_std,
                        clip_latent_noise=clip_latent_noise,
                    ).cpu()
                    batch_images = (batch_images + 1) / 2
                    # keep the generated images
                    for j in range(len(batch_images)):
                        img_name = filename(i + j)
                        torchvision.utils.save_image(
                            batch_images[j],
                            os.path.join(conf.generate_dir, f"{img_name}.png"),
                        )
            else:
                # evaulate autoencoder (given the images)
                # to make the FID fair, autoencoder must not see the validation dataset
                # also shuffle to make it closer to unconditional generation
                train_loader = make_subset_loader(
                    conf,
                    dataset=train_data,
                    batch_size=batch_size,
                    shuffle=True,
                    parallel=True,
                )

                i = 0
                for batch in tqdm(train_loader, desc="generating images"):
                    imgs = batch["img"].to(device)
                    x_T = torch.randn(
                        (len(imgs), 3, conf.img_size, conf.img_size), device=device
                    )
                    batch_images = render_condition(
                        conf=conf,
                        model=model,
                        x_T=x_T,
                        x_start=imgs,
                        cond=None,
                        sampler=sampler,
                        latent_sampler=latent_sampler,
                    ).cpu()
                    # model: BeatGANsAutoencModel
                    # # returns {'cond', 'cond2'}
                    # conds = model.encode(imgs)
                    # batch_images = sampler.sample(model=model,
                    #                               noise=x_T,
                    #                               model_kwargs=conds).cpu()
                    # denormalize the images
                    batch_images = (batch_images + 1) / 2
                    # keep the generated images
                    for j in range(len(batch_images)):
                        img_name = filename(i + j)
                        torchvision.utils.save_image(
                            batch_images[j],
                            os.path.join(conf.generate_dir, f"{img_name}.png"),
                        )
                    i += len(imgs)
        else:
            raise NotImplementedError()
    model.train()

    barrier()

    if get_rank() == 0:
        fid = fid_score.calculate_fid_given_paths(
            [cache_dir, conf.generate_dir], batch_size, device=device, dims=2048
        )

        # remove the cache
        if remove_cache and os.path.exists(conf.generate_dir):
            shutil.rmtree(conf.generate_dir)

    barrier()

    if get_rank() == 0:
        # need to float it! unless the broadcasted value is wrong
        fid = torch.tensor(float(fid), device=device)
        broadcast(fid, 0)
    else:
        fid = torch.tensor(0.0, device=device)
        broadcast(fid, 0)
    fid = fid.item()
    print(f"fid ({get_rank()}):", fid)

    return fid


def loader_to_path(loader: DataLoader, path: str, denormalize: bool):
    # not process safe!

    if not os.path.exists(path):
        os.makedirs(path)

    # write the loader to files
    i = 0
    for batch in tqdm(loader, desc="copy images"):
        imgs = batch["img"]
        if denormalize:
            imgs = (imgs + 1) / 2
        for j in range(len(imgs)):
            torchvision.utils.save_image(imgs[j], os.path.join(path, f"{i+j}.png"))
        i += len(imgs)


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
