import os
import sys
import gradio as gr

os.makedirs("outputs", exist_ok=True) 
sys.path.insert(0, '.')


from utils.get_opt import get_opt
from os.path import join as pjoin
import numpy as np
from trainers import DDPMTrainer
from models import MotionTransformer

device = 'cpu'
opt = get_opt("checkpoints/t2m/t2m_motiondiffuse/opt.txt", device)
opt.do_denoise = True

assert opt.dataset_name == "t2m"
opt.data_root = './dataset/HumanML3D'
opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
opt.text_dir = pjoin(opt.data_root, 'texts')
opt.joints_num = 22
opt.dim_pose = 263

mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
std = np.load(pjoin(opt.meta_dir, 'std.npy'))


def build_models(opt):
    encoder = MotionTransformer(
        input_feats=opt.dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder


encoder = build_models(opt).to(device)
trainer = DDPMTrainer(opt, encoder)
trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))

trainer.eval_mode()
trainer.to(opt.device)

def generate(prompt, length):
    from tools.visualization import process
    result_path = "outputs/" + str(hash(prompt)) + ".mp4"
    process(trainer, opt, device, mean, std, prompt, int(length), result_path)
    return result_path

demo = gr.Interface(
    fn=generate,
    inputs=["text", gr.Slider(20, 196, value=60)],
    examples=[
        ["the man throws a punch with each hand.", 58],
        ["a person spins quickly and takes off running.", 29],
        ["a person quickly waves with their right hand", 46],
        ["a person performing a slight bow", 89],
    ],
    outputs="video",
    title="MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model",
    description="This is an interactive demo for MotionDiffuse. For more information, feel free to visit our project page(https://mingyuan-zhang.github.io/projects/MotionDiffuse.html).")

demo.launch(enable_queue=True)