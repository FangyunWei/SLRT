import bpy
import numpy as np
from math import radians
# from mathutils import Matrix
import pickle
import os, argparse
from cmd_parser import parse_config
from tqdm import tqdm
import random; random.seed(0)
import math


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args, remaining = parser.parse_known_args()
    args = parse_config(remaining)

    bpy.ops.preferences.addon_install(filepath = '../../pretrained_models/smplx_blender_addon_300_20220623.zip', overwrite = True)
    bpy.ops.preferences.addon_enable(module = 'smplx_blender_addon')
    bpy.ops.wm.save_userpref()
    bpy.ops.wm.open_mainfile(filepath="../../pretrained_models/smplx_ronglai.blend")
    
    path = os.path.abspath('../../pretrained_models/smplx_blender_addon/data')
    bpy.ops.file.find_missing_files(directory=path)

    bpy.data.scenes['Scene'].render.resolution_y = 512
    bpy.data.scenes['Scene'].render.resolution_x = 512
    # bpy.data.objects["Camera"].location[0] = -0.02
    bpy.data.objects["Camera"].location[1] = -0.85 #-0.725
    bpy.data.objects["Camera"].location[2] = 0.155
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'

    with open('data/phoenix_dev.pkl', 'rb') as f:
        video_ids = pickle.load(f)
    init_idx = args['init_idx']
    num_per_proc = args['num_per_proc']
    start_idx = init_idx
    end_idx = start_idx + num_per_proc
    video_ids = video_ids[start_idx:end_idx]

    for video_id in tqdm(video_ids):
        motion_path = os.path.join('./motions', video_id)
        if not os.path.exists(motion_path):
            continue
        motion_lst = os.listdir(motion_path)
        motion_lst.sort()
        
        img_dir = os.path.join('./images', video_id)
        os.makedirs(img_dir, exist_ok=True)

        for i in range(len(motion_lst)):
            fname = os.path.join(motion_path, motion_lst[i])
            bpy.ops.object.smplx_load_pose(filepath=fname)
            bpy.ops.render.render()
            bpy.data.images["Render Result"].save_render(img_dir+'/images{:04d}.png'.format(i))