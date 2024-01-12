# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

import multiprocessing as mp
from pathlib import Path
import numpy as np

"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import os
import argparse
import torch as th
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util
from time import perf_counter
import cv2
import lpips
import pandas as pd

# Workaround
try:
    import ctypes

    libgcc_s = ctypes.CDLL("libgcc_s.so.1")
except:
    pass


from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402


def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def build_conf(
    exp, conf_name, total_it=20, n=1, jump_length=None, jump_n_sample=None, seed=0, parallel=True
):
    conf_path = f"experiments/{exp}/confs/{conf_name}.yml"
    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(conf_path))

    conf_arg['callback'] = parallel

    conf_arg["cond_y"] = 933  # 78 # 933
    output_folder = f"experiments/{exp}/outputs/{conf_name}/its_{total_it}_jl_{jump_length}_js_{jump_n_sample}"

    eval_name = conf_arg.get_default_eval_name()
    # name
    # seed
    # gt path
    conf_arg["data"]["eval"][eval_name][
        "gt_path"
    ] = f"experiments/{exp}/gts/{conf_name}/img"
    conf_arg["data"]["eval"][eval_name][
        "mask_path"
    ] = f"experiments/{exp}/gts/{conf_name}/mask"
    conf_arg["data"]["eval"][eval_name]["paths"]["srs"] = f"{output_folder}/inpainted"
    conf_arg["data"]["eval"][eval_name]["paths"]["lrs"] = f"{output_folder}/gt_masked"
    conf_arg["data"]["eval"][eval_name]["paths"]["gts"] = f"{output_folder}/gt"
    conf_arg["data"]["eval"][eval_name]["paths"][
        "gt_keep_masks"
    ] = f"{output_folder}/gt_keep_mask"
    conf_arg["log_dir"] = f"{output_folder}/logs/"
    # mask path
    # max_len
    # paths: srs, lrs, gts, gt_keep_masks
    conf_arg["data"]["eval"][eval_name]["max_len"] = n
    conf_arg["timestep_respacing"] = str(total_it)
    conf_arg["schedule_jump_params"]["t_T"] = total_it
    if jump_length is not None:
        conf_arg["schedule_jump_params"]["jump_length"] = jump_length
    if jump_n_sample is not None:
        conf_arg["schedule_jump_params"]["jump_n_sample"] = jump_n_sample
    # conf_arg['schedule_jump_params']['jump_length'] = 5
    # conf_arg['schedule_jump_params']['jump_n_sample'] = 3

    conf_arg["seed"] = seed

    conf_arg["reload"] = False
    # conf_arg['save_model'] = f'experiments/{exp}/outputs/{conf_name}/model.pkl'
    # conf_arg['save_idx'] = [14]
    # conf_arg['stop_it'] = [14]

    return conf_arg


def main(conf):
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    im = plt.imshow(
        np.random.rand(10, 10), animated=True
    )  # Initialize with a random image
    plt.axis("off")
    plt.title("Sampled image")
    plt.subplot(1, 2, 2)
    gr = plt.plot(np.random.rand(10), marker="x")
    scatter = plt.scatter(np.random.rand(10), np.random.rand(10), color='red', zorder=10)
    plt.xlabel("Repaint step")
    plt.ylabel("Diffusion time")
    plt.title("Resampling strategy")

    queue = mp.Queue()

    times = []

    def callback(args):
        while not queue.empty():
            data = queue.get_nowait()
            if isinstance(data, tuple):
                if isinstance(data[0], str):
                    if data[0] == "times":
                        times.extend(data[1])
                        data_arr = np.array(data[1])
                        x = np.arange(data_arr.shape[0])
                        gr[0].set_ydata(data_arr)
                        gr[0].set_xdata(x)
                        gr[0].axes.set_xlim(0, data_arr.shape[0])
                        gr[0].axes.set_ylim(np.min(data_arr), np.max(data_arr))
                        np.savetxt(conf["log_dir"] + "/times.txt", np.c_[x, data_arr], fmt="%f")
                    else:
                        plt.suptitle(data[1])
                        if data[1] == "Sampling complete":
                            ani.event_source.stop()
                            print("Done")
                else:
                    idx = data[0]
                    data = data[1]
                    data = (data - np.min(data)) / (np.max(data) - np.min(data))
                    if times is not None:
                        scatter.set_offsets(np.c_[idx, times[idx]])
                    im.set_array(data)  # Update the image
        return im, gr, scatter

  
    if conf['callback']:
        ani = animation.FuncAnimation(
            fig, callback, frames=range(1000), interval=1000, blit=False
        )


        p = mp.Process(target=sample_now, args=(conf, queue))
        p.start()

        plt.show()

        return p, ani
    else:
        print("Starting process...")
        p = mp.Process(target=sample_now, args=(conf, queue))
        p.start()

        return p, queue


import sys


def sample_now(conf, callback_code):
    os.makedirs(conf["log_dir"], exist_ok=True)
    sys.stdout = open(conf["log_dir"] + str(os.getpid()) + ".out", "w")
    sys.stderr = open(conf["log_dir"] + str(os.getpid()) + ".err", "w")

    th.random.manual_seed(conf["seed"])
    np.random.seed(conf["seed"])

    assert conf["schedule_jump_params"]["t_T"] == int(conf["timestep_respacing"]), (
        conf["schedule_jump_params"]["t_T"],
        conf["timestep_respacing"],
    )

    print("Start", conf["name"])
    callback_code.put(("msg", f"Start {conf['name']}..."))

    device = dist_util.dev(conf.get("device"))
    print("device:", device)
    conf_y = conf.get("cond_y")
    print("cond_y:", conf_y)
    callback_code.put(("msg", f"device: {device}..."))

    print("loading model...")
    loss_fn_alex = lpips.LPIPS(net="alex")  # best forward scores
    callback_code.put(("msg", f"loading model..."))
    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    print("loading state")
    callback_code.put(("msg", f"loading state..."))
    model.load_state_dict(
        dist_util.load_state_dict(
            os.path.expanduser(conf.model_path), map_location="cpu"
        )
    )
    model.to(device)

    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        callback_code.put(("msg", f"loading classifier..."))
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys())
        )
        print(select_args(conf, classifier_defaults().keys()))
        print(conf.classifier_path)
        classifier.load_state_dict(
            dist_util.load_state_dict(
                os.path.expanduser(conf.classifier_path), map_location="cpu"
            )
        )

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale

    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")
    all_images = []

    dset = "eval"

    eval_name = conf.get_default_eval_name()

    print("eval_name:", eval_name)
    print("loading dataloader...")
    callback_code.put(("msg", f"loading dataloader..."))
    dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    counter = 0
    count_max = conf["data"]["eval"][eval_name]["max_len"]
    times = []
    for batch in iter(dl):
        counter += 1
        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}

        model_kwargs["gt"] = batch["GT"]

        gt_keep_mask = batch.get("gt_keep_mask")
        if gt_keep_mask is not None:
            model_kwargs["gt_keep_mask"] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        # print('cond_y', conf.conf_y)

        # if 'cond_y' in conf:
        # conf.cond_y = conf['cond_y']
        classes = th.ones(batch_size, dtype=th.long, device=device) * conf_y
        model_kwargs["y"] = classes
        print(model_kwargs["y"])
        # if conf_y is not None:
        #     classes = th.ones(batch_size, dtype=th.long, device=device)
        #     model_kwargs["y"] = classes * conf_y
        # else:
        #     classes = th.randint(
        #         low=0, high=NUM_CLASSES, size=(batch_size,), device=device
        #     )
        #     model_kwargs["y"] = classes
        try:
            import json

            with open("inet_labels.json", "r") as f:
                class_names = json.load(f)
            print(classes)
            print("classes:", classes, class_names[str(classes[0].item())])
        except:
            print("Failed to load class names")

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )

        callback_code.put(("msg", f"Start sampling... {counter}/{count_max}"))
        time_begin = perf_counter()
        result = sample_fn(
            model_fn,
            (batch_size, 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf,
            callback=callback_code,
        )
        time_end = perf_counter()
        times.append(time_end - time_begin)
        srs = toU8(result["sample"])
        gts = toU8(result["gt"])
        lrs = toU8(
            result.get("gt") * model_kwargs.get("gt_keep_mask")
            + (-1)
            * th.ones_like(result.get("gt"))
            * (1 - model_kwargs.get("gt_keep_mask"))
        )

        gt_keep_masks = toU8((model_kwargs.get("gt_keep_mask") * 2 - 1))

        conf.eval_imswrite(
            srs=srs,
            gts=gts,
            lrs=lrs,
            gt_keep_masks=gt_keep_masks,
            img_names=batch["GT_name"],
            dset=dset,
            name=eval_name,
            verify_same=False,
        )

    result_dir = str(Path(conf["log_dir"]).parent) + "/results/"
    os.makedirs(result_dir, exist_ok=True)
    # with open(result_dir + conf['name'] + '.times', 'w') as f:
    #     np.savetxt(f, times, fmt='%f')

    # lpips score
    losses = []
    ssims = []
    mses = []
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error

    for img in sorted(os.listdir(conf["data"]["eval"][eval_name]["gt_path"]))[:count_max]:
        file_img0 = os.path.join(conf["data"]["eval"][eval_name]["paths"]["gts"], img)
        file_img1 = os.path.join(conf["data"]["eval"][eval_name]["paths"]["srs"], img)
        img0 = cv2.imread(file_img0, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        img1 = cv2.imread(file_img1, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

        print("LPIPS for", img)

        if img0.shape[0] > 64:
            # downsample to 64x64
            img0 = cv2.resize(img0 * 255, (64, 64), interpolation=cv2.INTER_AREA)
            cv2.imwrite(file_img0 + ".n.png", img0)
            img1 = cv2.resize(img1 * 255, (64, 64), interpolation=cv2.INTER_AREA)
            cv2.imwrite(file_img1 + ".dn.png", img1)
            print("resizing to 64x64")

        img0t = th.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).float()
        img1t = th.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()



        d = loss_fn_alex(img0t, img1t)
        losses.append(d.item())

        # ssim
        print("SSIM for", img0.shape, img1.shape)
        ssimdim1 = ssim(img0[:, :, 0], img1[:, :, 0], data_range=1.0)
        ssimdim2 = ssim(img0[:, :, 1], img1[:, :, 1], data_range=1.0)
        ssimdim3 = ssim(img0[:, :, 2], img1[:, :, 2], data_range=1.0)
        ssims.append((ssimdim1 + ssimdim2 + ssimdim3) / 3.0)

        mse = mean_squared_error(img0, img1)
        mses.append(mse)

        # ssim  = ssim(img0.squeeze().permute(1, 2, 0).numpy(), img1.squeeze().permute(1, 2, 0).numpy(), multichannel=True)
        # ssims.append(ssim)

        # mse = mean_squared_error(img0.squeeze().permute(1, 2, 0).numpy(), img1.squeeze().permute(1, 2, 0).numpy())
        # mses.append(mse)




    r_jump_length = [conf["schedule_jump_params"]["jump_length"]] * len(losses)
    r_jump_n_sample = [conf["schedule_jump_params"]["jump_n_sample"]] * len(losses)
    r_total_it = [conf["schedule_jump_params"]["t_T"]] * len(losses)
    r_seed = [conf["seed"]] * len(losses)
    r_model = [os.path.basename(conf["model_path"])] * len(losses)

    results = pd.DataFrame({"lpips": losses, "ssim": ssims, "mse": mses,
        "time": times, 
        "model_name": r_model, "jump_length": r_jump_length, "jump_n_sample": r_jump_n_sample, "total_it": r_total_it, "seed": r_seed})
    #"model_name":None, "jump_length":None, "jump_n_sample":None, "total_it":None; "seed": None})
    results = results.round(4)
    results.to_csv(result_dir + conf["name"] + ".csv")

    print("sampling complete")
    callback_code.put(("msg", f"Sampling complete"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path", type=str, required=False, default=None)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get("conf_path")))
    main(conf_arg)
