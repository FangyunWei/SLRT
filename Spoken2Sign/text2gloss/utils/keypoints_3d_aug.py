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

import os; os.environ['PYOPENGL_PLATFORM'] = 'egl'
import os.path as osp

import argparse
import pickle
import torch
import torch.nn as nn
import smplx

import configargparse, yaml
from human_body_prior.tools.model_loader import load_vposer

import pyrender
import numpy as np
from numpy import pi
import torch.nn.functional as F
from collections import defaultdict


class JointMapper(nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)


def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'

    '''
    if openpose_format.lower() == 'coco25':

        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format.lower() == 'cocowholebody':
        if model_type == 'smpl':
            return np.array([24,26,25,28,27,16,17,18,19,20,21,1,2,4,5,7,8,29,30,31,32,33,34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55,57,56,59,58,16,17,18,19,20,21,1,2,4,5,7,8,60,61,62,63,64,65], dtype=np.int32)
            mapping = [body_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127,
                                         dtype=np.int32)
                mapping += [face_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]


            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))
    

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
        mesh_copy[:self.r,:,:] = mesh[0,:,:]
        mesh_copy[self.r:b+self.r,:,:] =mesh
        mesh_copy[b+self.r:b+2*self.r,:,:]=mesh[-1,:,:]

        for i in range(k):
            for j in range(self.r,self.r+b):
                mesh_copy[j,0,i] = np.sum(self.kernel * mesh_copy[j-self.r : j+self.r+1, 0, i]) # 卷积运算

        return mesh_copy[self.r:self.r+b,:,:]


def parse_config():
    args_dict = {'data_folder': 'data', 'max_persons': 3, 'config': 'cfg_files/fit_smplx_phoenix_intp.yaml', 'loss_type': 'smplify', 'interactive': True, 'save_meshes': True, 'visualize': False, 'degrees': [0, 90, 180, 270], 'use_cuda': True, 'dataset': 'openpose', 'joints_to_ign': [1, 9, 12], 'output_folder': 'results/smplsignx_intp', 'img_folder': 'images_phoenix/', 'keyp_folder': 'keypoints', 'summary_folder': 'summaries', 'result_folder': 'results', 'mesh_folder': 'meshes', 'gender_lbl_type': 'none', 'gender': 'neutral', 'float_dtype': 'float32', 'model_type': 'smplx', 'camera_type': 'persp', 'optim_jaw': True, 'optim_hands': True, 'optim_expression': True, 'optim_shape': True, 'model_folder': '../../data/models', 'use_joints_conf': True, 'batch_size': 1, 'num_gaussians': 8, 'use_pca': True, 'num_pca_comps': 12, 'flat_hand_mean': False, 'body_prior_type': 'l2', 'left_hand_prior_type': 'l2', 'right_hand_prior_type': 'l2', 'jaw_prior_type': 'l2', 'use_vposer': True, 'vposer_ckpt': 'vposer', 'init_joints_idxs': [9, 12, 2, 5], 'body_tri_idxs': [(5, 12), (2, 9)], 'prior_folder': 'priors', 'focal_length': 5000.0, 'rho': 100.0, 'interpenetration': True, 'penalize_outside': True, 'data_weights': [1, 1, 1, 1, 1], 'body_pose_prior_weights': [404.0, 404.0, 57.4, 4.78, 4.78], 'shape_weights': [100.0, 50.0, 10.0, 5.0, 5.0], 'expr_weights': [100.0, 50.0, 10.0, 5.0, 5.0], 'face_joints_weights': [0.0, 0.0, 0.0, 0.0, 2.0], 'hand_joints_weights': [0.0, 0.0, 0.0, 0.1, 2.0], 'jaw_pose_prior_weights': ['4.04e03,4.04e04,4.04e04', '4.04e03,4.04e04,4.04e04', '574,5740,5740', '47.8,478,478', '47.8,478,478'], 'hand_pose_prior_weights': [404.0, 404.0, 57.4, 4.78, 4.78], 'coll_loss_weights': [0.0, 0.0, 0.0, 0.01, 1.0], 'depth_loss_weight': 100.0, 'df_cone_height': 0.0001, 'max_collisions': 128, 'point2plane': False, 'part_segm_fn': '', 'ign_part_pairs': ['9,16', '9,17', '6,16', '6,17', '1,2', '12,22'], 'use_hands': True, 'use_face': True, 'use_face_contour': False, 'side_view_thsh': 25, 'optim_type': 'lbfgsls', 'lr': 0.01, 'gtol': 1e-09, 'ftol': 1e-09, 'maxiters': 90, 'joints_to_fix': [-1], 'init_idx': 0, 'num_per_proc': 1, 'identifier': '', 'keypoints_folder': '../../data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody_iso.pkl', 'smooth_loss_weight': 8000.0, 'unseen_left_weight': 3000.0, 'unseen_right_weight': 3000.0, 'upstanding_loss_weight': 300000.0, 'data_subset': 'data/Phoenix-2014T.pkl', 'dataset_to_fitting': 'Phoenix-2014T', 'only_kps': 0}
    return args_dict


def create_smplx_model(cfg_file=''):
    args = parse_config()
    with open(cfg_file, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    args.update(cfg)

    dtype = torch.float32
    use_cuda = args.get('use_cuda', True)
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(os.environ['LOCAL_RANK']))
    else:
        device = torch.device('cpu')

    mapping = smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                                     use_face_contour=False, openpose_format='coco25')
    joint_mapper = JointMapper(mapping)
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

    model = smplx.create(**model_params)
    model = model.to(device=device)

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
    return {'model': model, 'vposer': vposer, 'pose_embedding': pose_embedding, 'args': args}


def kps_aug(model=None, vposer=None, pose_embedding=None, render_res=None, angle=0., camera_shift=0, args={}):
    device = torch.device('cuda:{}'.format(os.environ['LOCAL_RANK']))
    dtype = torch.float32
    use_vposer = args.get('use_vposer', True)
    model_type = 'smplx'
    data = render_res

    if args['dataset_to_fitting'] == 'Phoenix-2014T':
        H, W = 260, 210
        camera_transl = torch.tensor([0., -0.7, 24.], device=device)
    elif args['dataset_to_fitting'] == 'CSL-Daily':
        H, W = 512, 512
        camera_transl = torch.tensor([0., -0.7, 12.], device=device)
    elif args['dataset_to_fitting'] == 'WLASL':
        H, W = 256, 256
        camera_transl = torch.tensor([0., -0.7, 24.], device=device)
    elif args['dataset_to_fitting'] == 'MSASL':
        H, W = 256, 256
        camera_transl = torch.tensor([0., -0.7, 24.], device=device)
    else:
        raise ValueError
    camera_transl[2] -= camera_shift
    camera_transl[2] *= -1.

    est_params = {}
    for key, val in data[0][0]['result'].items():
        if key == 'camera_rotation':
            data_key = np.zeros([len(data),1,3,3])
            for idx, data_i in enumerate(data):
                data_key[idx] = data_i[0]['result'][key]
            est_params[key] = data_key
        else:
            data_key = np.zeros([len(data), 1, data[0][0]['result'][key].shape[1]])
            for idx, data_i in enumerate(data):
                data_key[idx] = data_i[0]['result'][key]
            est_params[key] = data_key

    for key, val in data[0][0]['result'].items():
        if key == 'camera_rotation':
            data_temp = est_params[key].reshape(-1,1,9)
            GuassianBlur_ = GuassianBlur(1)
            out_smooth = GuassianBlur_.guassian_blur(data_temp, flag=0)
            est_params[key] = out_smooth.reshape(-1,1,3,3)
        else:
            GuassianBlur_ = GuassianBlur(1)
            out_smooth = GuassianBlur_.guassian_blur(est_params[key], flag=0)
            est_params[key] = out_smooth

    batch_est_params = defaultdict(list)
    for idx, data in enumerate(render_res):
        if use_vposer:
            with torch.no_grad():
                pose_embedding[:] = torch.tensor(
                    est_params['body_pose'][idx], device=device, dtype=dtype)

        for key, val in data[0]['result'].items():
            if key == 'body_pose' and use_vposer:
                body_pose = vposer.decode(
                    pose_embedding, output_type='aa').view(1, -1)
                if model_type == 'smpl':
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                            dtype=body_pose.dtype,
                                            device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)

                batch_est_params['body_pose'].append(body_pose)
            elif key == 'betas':
                batch_est_params[key].append(torch.zeros([1, 10], dtype=dtype, device=device))
            else:
                batch_est_params[key].append(torch.tensor(est_params[key][idx], dtype=dtype, device=device))

    for k,v in batch_est_params.items():
        batch_est_params[k] = torch.cat(v, dim=0)
    rad_angle = pi*angle/180
    batch_est_params['global_orient'][..., -1] = batch_est_params['global_orient'][..., -1] + rad_angle
    model_output = model(**batch_est_params)

    camera_center = [1.0*W/2, 1.0*H/2]
    camera_pose = torch.eye(4, dtype=dtype, device=device)
    camera_pose[:3, 3] = camera_transl
    camera_pose[:3, :3] = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=dtype, device=device)

    focal_length = 5000
    camera = pyrender.camera.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=camera_center[0], cy=camera_center[1])

    joints_3D = model_output.joints
    # print(joints_3D.shape)
    num_kps = joints_3D.shape[1]
    proj_matrix = torch.tensor(camera.get_projection_matrix(W, H), dtype=dtype, device=device)
    joints_3D = F.pad(joints_3D.view(-1,3), [0,1], value=1).to(dtype)
    # joints_2D = proj_matrix @ torch.linalg.inv(camera_pose) @ joints_3D.T
    joints_2D = torch.matmul(torch.matmul(proj_matrix, torch.linalg.inv(camera_pose)), joints_3D.T)
    joints_2D = joints_2D/joints_2D[3]
    joints_2D[0] = W/2*joints_2D[0] + W/2
    joints_2D[1] = H - (H/2*joints_2D[1] + H/2)
    joints_2D = joints_2D[:2,:].T
    joints_2D = joints_2D.view(-1, num_kps, 2)
    # print(joints_2D.shape)

    mapping = np.array(
            [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11, 19, 20, 21, 22, 23, 24],
            dtype=np.int32)
    keypoints_coco = torch.zeros([joints_2D.shape[0], 133, 3], dtype=dtype, device=device)

    k = [_ for _ in range(mapping.shape[0])]
    keypoints_coco[:, k, :2] = joints_2D[:, mapping, :]
    keypoints_coco[:, k, -1] = 1
    keypoints_coco[:, 91:, :2] = joints_2D[:, 25:67, :]
    keypoints_coco[:, 91:, -1] = 1
    keypoints_coco[:, 40:91, :2] = joints_2D[:, 67:, :]
    keypoints_coco[:, 40:91, -1] = 1
    keypoints_coco = clean_nan_kps(keypoints_coco)
    return keypoints_coco

def clean_nan_kps(kps):
    idx = torch.isnan(kps.sum(dim=(1,2)))
    kps = kps[~idx]
    return kps

if __name__ == '__main__':
    kps_aug(render_res=None, angle=20.)