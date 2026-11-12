"""Checkpoint loading for the PreNet + guided residual diffusion model."""

import json
import os
from collections import OrderedDict

import torch

import model as Model


def read_config(config_path):
    """Read a JSON config, tolerating ``//`` line comments."""
    json_str = ""
    with open(config_path, "r") as f:
        for line in f:
            json_str += line.split("//")[0] + "\n"
    return json.loads(json_str, object_pairs_hook=OrderedDict)


def load_model_for_dose(config_path, checkpoint_prefix, device_id=0):
    """Instantiate the model described by ``config_path`` and load weights.

    ``checkpoint_prefix`` is the checkpoint stem without the network suffix,
    e.g. ``checkpoints/I500000_E93``.  The loader expects eight sibling files::

        {prefix}_PreNet_gen.pth
        {prefix}_DenoiseNet_gen.pth
        {prefix}_guide_3D_{1,2,3}_gen.pth
        {prefix}_guide_spectrum_{1,2,3}_gen.pth

    Returns:
        ``(diffusion_model, opt)``.
    """
    opt = read_config(config_path)

    if os.path.isabs(checkpoint_prefix):
        resume_state = checkpoint_prefix
        ckpt_dir = os.path.dirname(checkpoint_prefix)
    else:
        resume_state = os.path.join(os.getcwd(), checkpoint_prefix)
        ckpt_dir = os.path.dirname(resume_state) or "checkpoints"

    opt["path"]["checkpoint"] = ckpt_dir
    opt["path"]["resume_state"] = resume_state
    opt["phase"] = "val"
    opt["distributed"] = False

    if torch.cuda.is_available():
        opt["gpu_ids"] = [device_id]
        torch.cuda.set_device(device_id)
    else:
        opt["gpu_ids"] = []   # CPU fallback

    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(
        opt["model"]["beta_schedule"]["val"], schedule_phase="val")
    return diffusion, opt
