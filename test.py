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

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
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

def main(conf):    
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    im = plt.imshow(np.random.rand(10,10), animated=True)  # Initialize with a random image
    plt.axis('off')
    plt.title('Sampled image')
    plt.subplot(1, 2, 2)
    gr = plt.plot(np.random.rand(10), marker='x')
    scatter = plt.scatter(np.random.rand(10), np.random.rand(10))
    plt.xlabel('Repaint step')
    plt.ylabel('Diffusion time')
    plt.title("Resampling strategy")

    queue = mp.Queue()

    times = []

    def callback(args):
        while not queue.empty():
            data = queue.get_nowait()
            if isinstance(data, tuple):
                if isinstance(data[0], str):
                    if data[0] == 'times':                        
                        times.extend(data[1])
                        data_arr = np.array(data[1])
                        x = np.arange(data_arr.shape[0])
                        gr[0].set_ydata(data_arr)
                        gr[0].set_xdata(x)
                        gr[0].axes.set_xlim(0, data_arr.shape[0])
                        gr[0].axes.set_ylim(np.min(data_arr), np.max(data_arr))
                    else:
                        plt.suptitle(data[1])
                else:
                    idx = data[0]
                    data = data[1]
                    data = (data - np.min(data))/(np.max(data) - np.min(data))
                    if times is not None:
                        scatter.set_offsets(np.c_[idx, times[idx]])
                    im.set_array(data)  # Update the image                
        return im, gr, scatter

    ani = animation.FuncAnimation(fig, callback, frames=range(1000), interval=1000, blit=False)

    p = mp.Process(target=sample_now, args=(conf, queue))
    p.start()

    plt.show()
    return p, ani

def sample_now(conf, callback_code):    
    th.random.manual_seed(conf['seed'])
    np.random.seed(conf['seed'])

    assert conf['schedule_jump_params']['t_T'] == int(conf['timestep_respacing']), (conf['schedule_jump_params']['t_T'], conf['timestep_respacing'])

    print("Start", conf['name'])
    callback_code.put(('msg', f"Start {conf['name']}..."))

    device = dist_util.dev(conf.get('device'))
    print("device:", device)
    callback_code.put(('msg', f"device: {device}..."))

    print("loading model...")
    callback_code.put(('msg', f"loading model..."))
    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    print("loading state")
    callback_code.put(('msg', f"loading state..."))
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)

    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        callback_code.put(('msg', f"loading classifier..."))
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        print(select_args(conf, classifier_defaults().keys()))
        print(conf.classifier_path)
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
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

    dset = 'eval'

    eval_name = conf.get_default_eval_name()

    print("eval_name:", eval_name)
    print("loading dataloader...")
    callback_code.put(('msg', f"loading dataloader..."))
    dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    for batch in iter(dl):
        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}

        model_kwargs["gt"] = batch['GT']

        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            import json
            with open( 'inet_labels.json', 'r') as f:
                class_names = json.load(f)
            print("classes:", classes, class_names[str(classes[0].item())])
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )

        callback_code.put(('msg', f"Start sampling..."))
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
            callback=callback_code
        )
        srs = toU8(result['sample'])
        gts = toU8(result['gt'])
        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                   th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))

        gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

        conf.eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

    print("sampling complete")
    callback_code.put(('msg', f"Sampling complete"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    main(conf_arg)
