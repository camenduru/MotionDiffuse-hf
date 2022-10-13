import os
import torch
import argparse

import utils.paramUtil as paramUtil
from torch.utils.data import DataLoader
from utils.plot_script import *

from utils.utils import *
from utils.motion_process import recover_from_ric


def plot_t2m(opt, data, result_path, caption):
    joint = recover_from_ric(torch.from_numpy(data).float(), opt.joints_num).numpy()
    # joint = motion_temporal_filter(joint, sigma=1)
    plot_3d_motion(result_path, paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20)


def process(trainer, opt, device, mean, std, text, motion_length, result_path):

    result_dict = {}
    with torch.no_grad():
        if motion_length != -1:
            caption = [text]
            m_lens = torch.LongTensor([motion_length]).to(device)
            pred_motions = trainer.generate(caption, m_lens, opt.dim_pose)
            motion = pred_motions[0].cpu().numpy()
            motion = motion * std + mean
            title = text + " #%d" % motion.shape[0]
            plot_t2m(opt, motion, result_path, title)
