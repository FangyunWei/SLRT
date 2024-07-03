# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
# Contact: Vassilis choutas, vassilis.choutas@tuebingen.mpg.de

import os
import os.path as osp

import pickle, gzip
import torch
import argparse

import pyrender
import trimesh
import numpy as np
from cmd_parser import parse_config
from human_body_prior.tools.model_loader import load_vposer
import smplx
import PIL.Image as pil_img
import cv2
from numpy import pi
import math
from sign_connector_train import MLP
import utils
from collections import defaultdict
import random; random.seed(0)
from tqdm import tqdm


class GuassianBlur():
    def __init__(self, r, sigma=1):

        self.r = r
        self.kernel = np.empty(2 * r + 1)
        total = 0
        for i in range(2 * r + 1):
            self.kernel[i] = np.exp(-((i - r) ** 2 ) / (2 * sigma ** 2)) / ((2 * pi)**1/2 * sigma ** 2)
            total += self.kernel[i]
        self.kernel /= total

    def guassian_blur(self, mesh, flag=0):

        b, l, k  = mesh.shape
        mesh_copy = np.zeros([b+2*self.r,l,k])
        mesh_result = np.zeros([b + 2 * self.r, l, k])
        mesh_copy[:self.r,:,:] = mesh[0,:,:]
        mesh_copy[self.r:b+self.r,:,:] =mesh
        mesh_copy[b+self.r:b+2*self.r,:,:]=mesh[-1,:,:]

        for i in range(k):
            for j in range(self.r,self.r+b):
                mesh_result[j,0,i] = np.sum(self.kernel * mesh_copy[j-self.r : j+self.r+1, 0, i]) # 卷积运算

        return mesh_result[self.r:self.r+b,:,:]


if __name__ == '__main__':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    parser = argparse.ArgumentParser()
    args, remaining = parser.parse_known_args()
    args = parse_config(remaining)
    dtype = torch.float32
    use_cuda = args.get('use_cuda', True)
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model_type = args.get('model_type', 'smplx')
    print('Model type:', model_type)
    print(args.get('model_folder'))

    mapping = utils.smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                                     use_face_contour=False, openpose_format='coco25')
    joint_mapper = utils.JointMapper(mapping)

    model_params = dict(model_path=args.get('model_folder'),
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype,
                        **args)

    #-------------------------------------------------------Sign Connector---------------------------------------
    joint_idx = np.array([3, 4, 6, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66], dtype=np.int32)
    model = smplx.create(**model_params)
    model = model.to(device=device)
    sign_connector = MLP(input_dim=len(joint_idx)*3*2+len(joint_idx))
    sign_connector.load_state_dict(torch.load('data/connector.pth'), strict=True)
    sign_connector.to(device)
    sign_connector.eval()

    batch_size = args.get('batch_size', 1)
    use_vposer = args.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    vposer_ckpt = args.get('vposer_ckpt', '')
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)
        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()

    #-----------------------------------------------------Prepare Dict---------------------------------------------
    with open('data/Phoenix-2014T_all.pkl', 'rb') as f:
        render_results_all = pickle.load(f)
    gloss2items_path = 'data/gloss2items.pkl'
    with open(gloss2items_path, 'rb') as f:
        gloss2items = pickle.load(f)
    t2g_results = {}
    with open('data/T2G_results.pkl', 'rb') as f:
        res = pickle.load(f)
        t2g_results.update(res)

    with open('data/phoenix_dev.pkl', 'rb') as f:
        video_ids = pickle.load(f)
    init_idx = args['init_idx']
    num_per_proc = args['num_per_proc']
    start_idx = init_idx
    end_idx = start_idx + num_per_proc
    video_ids = video_ids[start_idx:end_idx]

    for video_id in tqdm(video_ids):
        glosses_translated = t2g_results[video_id]['gls_hyp'].split()
        clips = []
        for gloss in glosses_translated:
            if gloss not in gloss2items:
                # print(gloss, 'not in dict')  #maybe due to smplified-to-traditional conversion. rare cases.
                continue
            clips.append(gloss2items[gloss][0][0])

        est_params_all = []
        inter_flag = []
        for id_idx in range(len(clips)):
            render_id = clips[id_idx]['video_file']
            render_results = render_results_all[render_id]

            for pkl_idx in range(clips[id_idx]['start'], clips[id_idx]['end']):
                data = render_results[pkl_idx]
                est_params = {}
                for key, val in data[0]['result'].items():
                    if key in ['body_pose', 'left_hand_pose', 'right_hand_pose']:
                        est_params[key] = data[1][key + '_rot']
                    else:
                        est_params[key] = val
                est_params_all.append(est_params)
                inter_flag.append(False)

            if id_idx != len(clips)-1:
                clip_pre, clip_nex = clips[id_idx], clips[id_idx+1]
                data_0, data_1 = render_results[clip_pre['end']-1], render_results_all[clip_nex['video_file']][clip_nex['start']]

                est_params_pre = {}
                est_params_nex = {}
                for key, val in data_1[0]['result'].items():
                    est_params_pre[key] = val
                    est_params_nex[key] = val

                for k in range(2):
                    est_params = {}
                    if use_vposer:
                        with torch.no_grad():
                            if k == 0:
                                pose_embedding[:] = torch.tensor(
                                    est_params_pre['body_pose'], device=device, dtype=dtype)
                            else:
                                pose_embedding[:] = torch.tensor(
                                    est_params_nex['body_pose'], device=device, dtype=dtype)

                    for key, val in data[0]['result'].items():
                        if key == 'body_pose' and use_vposer:
                            body_pose = vposer.decode(
                                pose_embedding, output_type='aa').view(1, -1)
                            if model_type == 'smpl':
                                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                                        dtype=body_pose.dtype,
                                                        device=body_pose.device)
                                body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                            est_params['body_pose'] = body_pose
                        # elif key == 'betas':
                        #     est_params[key] = torch.zeros([1, 10], dtype=dtype, device=device)
                        # elif key == 'global_orient':
                        #     est_params[key] = torch.zeros([1, 3], dtype=dtype, device=device)
                        else:
                            if k == 0:
                                est_params[key] = torch.tensor(est_params_pre[key], dtype=dtype, device=device)
                            else:
                                est_params[key] = torch.tensor(est_params_nex[key], dtype=dtype, device=device)
                    model_output = model(**est_params)
                    joints_location = model_output.joints
                    joints_idx = torch.tensor([3, 4, 6, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                                            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
                                            dtype=torch.int32).to(device=device)
                    joints_location = torch.index_select(joints_location, 1, joints_idx)
                    if k == 0:
                        joints_location_pre = joints_location
                    else:
                        joints_location_nex = joints_location
                joints_dis = torch.sqrt(((joints_location_pre-joints_location_nex)**2).sum(dim=-1))
                joints_location_pre = joints_location_pre.reshape([1,-1])
                joints_location_nex = joints_location_nex.reshape([1,-1])
                # print(joints_location_pre.shape, joints_location_nex.shape, joints_dis.shape)

                len_inter = sign_connector(torch.cat((joints_location_pre, joints_location_nex, joints_dis), 1))
                len_inter = max(round(len_inter.item()),1)
                # print(len_inter)

                weights = np.zeros(len_inter)
                interval = 1.0/(len_inter+1)
                for i in range(len_inter):
                    weights[i] = 1.0-(i+1)*interval
                for idx_w, weight in enumerate(weights):
                    est_params = {}
                    for key, val in data_0[0]['result'].items():
                        if key in ['body_pose', 'left_hand_pose', 'right_hand_pose']:
                            est_params[key] = weight*data_0[1][key + '_rot'] +(1-weight)*data_1[1][key + '_rot']
                        else:
                            est_params[key] = weight*data_0[0]['result'][key] + (1-weight)*data_1[0]['result'][key]
                    est_params_all.append(est_params)
                    inter_flag.append(True)

        for key, val in data[0]['result'].items():
            if key == 'camera_rotation':
                date_temp = np.zeros([len(est_params_all), 1, 9])
                for i in range(len(est_params_all)):
                    date_temp[i] = est_params_all[i][key].reshape(1, 9)
                GuassianBlur_ = GuassianBlur(1)
                out_smooth = GuassianBlur_.guassian_blur(date_temp, flag=0)
                for i in range(len(est_params_all)):
                    est_params_all[i][key] = out_smooth[i].reshape(1, 3, 3)
            elif key == 'betas':
                for i in range(len(est_params_all)):
                    est_params_all[i][key] = np.asarray([[0.421,-1.658,0.361,0.314,0.226,0.065,0.175,-0.150,-0.097,-0.191]])
            elif key == 'global_orient':
                for i in range(len(est_params_all)):
                    est_params_all[i][key] = np.asarray([[0,0,0]])
            else:
                date_temp = np.zeros([len(est_params_all), 1, est_params_all[0][key].shape[1]])
                for i in range(len(est_params_all)):
                    date_temp[i] = est_params_all[i][key]
                GuassianBlur_ = GuassianBlur(1)
                out_smooth = GuassianBlur_.guassian_blur(date_temp, flag=0)
                for i in range(len(est_params_all)):
                    est_params_all[i][key] = out_smooth[i]

        save_dir = os.path.join('./motions', video_id)
        os.makedirs(save_dir, exist_ok=True)

        for i in range(len(est_params_all)):
            est_params_all[i]['body_pose'][:, 0:15] = 0.
            est_params_all[i]['body_pose'][:, 18:24] = 0.
            est_params_all[i]['body_pose'][:, 27:33] = 0.
            fname = os.path.join(save_dir, str(i).zfill(3)+'.pkl')
            if inter_flag[i]:
                fname = os.path.join(save_dir, str(i).zfill(3)+'_inter.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(est_params_all[i], f)

