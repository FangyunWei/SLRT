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
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import os.path as osp

import time
import yaml
import torch
import pickle

import smplx

from utils import JointMapper, init_DDP, is_main_process, synchronize
from cmd_parser import parse_config
from data_parser import create_dataset
from fit_single_frame import fit_single_frame

from camera import create_camera
from prior import create_prior
import numpy as np

torch.backends.cudnn.enabled = False


if __name__ == "__main__":
    args = parse_config()
    local_rank, world_size, device = init_DDP()
    rank = torch.distributed.get_rank()

    output_folder = args.pop('output_folder')
    output_folder = osp.expandvars(output_folder)
    result_folder = args.pop('result_folder', 'results')
    result_folder = osp.join(output_folder, result_folder)
    mesh_folder = args.pop('mesh_folder', 'meshes')
    mesh_folder = osp.join(output_folder, mesh_folder)
    out_img_folder = osp.join(output_folder, 'images')
    if is_main_process():
        os.makedirs(output_folder, exist_ok=True)
        os.system("cp {} {}".format(args['config'], output_folder))
        os.makedirs(result_folder, exist_ok=True)
        os.makedirs(mesh_folder, exist_ok=True)
        os.makedirs(out_img_folder, exist_ok=True)

    float_dtype = args['float_dtype']
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float64
    else:
        print('Unknown float type {}, exiting!'.format(float_dtype))
        sys.exit(-1)

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    keypoints_folder = args['keypoints_folder']
    with open(keypoints_folder, 'rb') as f:
        keypoints_all = pickle.load(f)

    video_ids_folder = args['data_subset']
    with open(video_ids_folder, 'rb') as f:
        video_ids = pickle.load(f)

    # start_idx = args['start_idx']
    # end_idx = args['end_idx']
    init_idx = args['init_idx']
    num_per_proc = args['num_per_proc']
    dataset_to_fitting = args['dataset_to_fitting']

    start_idx = init_idx + rank*num_per_proc
    end_idx = start_idx + num_per_proc
    save_fname = osp.join(result_folder, '{}_{:06d}_{:06d}.pkl'.format(dataset_to_fitting, start_idx, end_idx))
    results_all = {}
    if osp.exists(save_fname):
        with open(save_fname, 'rb') as f:
            results_all = pickle.load(f)
    save_step = 1
    for i in range(start_idx, end_idx):
        if i >= len(video_ids):
            break
        # if osp.exists('smplx_debug/meshes/'+video_ids[i]+'/'+'camera.pkl'):
        #     continue
        if video_ids[i] in results_all:
            continue

        img_folder = args.pop('img_folder', 'images_phoenix/')
        img_folder_i =  img_folder + video_ids[i]
        dataset_obj = create_dataset(img_folder=img_folder_i, 
                                     video_id=video_ids[i], 
                                     video_len=keypoints_all[video_ids[i]].shape[0], 
                                     dataset_name=dataset_to_fitting,
                                     **args)
        start = time.time()

        input_gender = args.pop('gender', 'neutral')
        gender_lbl_type = args.pop('gender_lbl_type', 'none')
        max_persons = args.pop('max_persons', -1)

        float_dtype = args.get('float_dtype', 'float32')
        if float_dtype == 'float64':
            dtype = torch.float64
        elif float_dtype == 'float32':
            dtype = torch.float32
        else:
            raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

        joint_mapper = JointMapper(dataset_obj.get_model2data())

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

        male_model = smplx.create(gender='male', **model_params)
        # SMPL-H has no gender-neutral model
        if args.get('model_type') != 'smplh':
            neutral_model = smplx.create(gender='neutral', **model_params)
        female_model = smplx.create(gender='female', **model_params)

        # Create the camera object
        focal_length = args.get('focal_length')
        camera = create_camera(focal_length_x=focal_length,
                           focal_length_y=focal_length,
                           dtype=dtype,
                           **args)

        if hasattr(camera, 'rotation'):
            camera.rotation.requires_grad = False

        use_hands = args.get('use_hands', True)
        use_face = args.get('use_face', True)

        body_pose_prior = create_prior(
            prior_type=args.get('body_prior_type'),
            dtype=dtype,
            **args)

        jaw_prior, expr_prior = None, None
        if use_face:
            jaw_prior = create_prior(
                prior_type=args.get('jaw_prior_type'),
                dtype=dtype,
                **args)
            expr_prior = create_prior(
                prior_type=args.get('expr_prior_type', 'l2'),
                dtype=dtype, **args)

        left_hand_prior, right_hand_prior = None, None
        if use_hands:
            lhand_args = args.copy()
            lhand_args['num_gaussians'] = args.get('num_pca_comps')
            left_hand_prior = create_prior(
                prior_type=args.get('left_hand_prior_type'),
                dtype=dtype,
                use_left_hand=True,
                **lhand_args)

            rhand_args = args.copy()
            rhand_args['num_gaussians'] = args.get('num_pca_comps')
            right_hand_prior = create_prior(
                prior_type=args.get('right_hand_prior_type'),
                dtype=dtype,
                use_right_hand=True,
                **rhand_args)

        shape_prior = create_prior(
            prior_type=args.get('shape_prior_type', 'l2'),
            dtype=dtype, **args)

        angle_prior = create_prior(prior_type='angle', dtype=dtype)

        camera_transl = torch.zeros([3])
        camera_orient = torch.zeros([3])
        betas_fix = torch.zeros([10])
        joints_smooth = torch.zeros([1,21,3])

        if use_cuda and torch.cuda.is_available():
            camera_transl=camera_transl.to(device=device)
            camera_orient=camera_orient.to(device=device)
            betas_fix=betas_fix.to(device=device)
            joints_smooth=joints_smooth.to(device=device)

            camera = camera.to(device=device)
            female_model = female_model.to(device=device)
            male_model = male_model.to(device=device)
            if args.get('model_type') != 'smplh':
                neutral_model = neutral_model.to(device=device)
            body_pose_prior = body_pose_prior.to(device=device)
            angle_prior = angle_prior.to(device=device)
            shape_prior = shape_prior.to(device=device)
            if use_face:
                expr_prior = expr_prior.to(device=device)
                jaw_prior = jaw_prior.to(device=device)
            if use_hands:
                left_hand_prior = left_hand_prior.to(device=device)
                right_hand_prior = right_hand_prior.to(device=device)
        else:
            device = torch.device('cpu')

        # A weight for every joint of the model
        joint_weights = dataset_obj.get_joint_weights().to(device=device,
                                                       dtype=dtype)
        # Add a fake batch dimension for broadcasting
        joint_weights.unsqueeze_(dim=0)

        keypoints_coco = np.array(keypoints_all[video_ids[i]])
        keypoints_coco = torch.tensor(keypoints_coco)

        mapping = np.array([0,6,6,8,10,5,7,9,12,12,14,16,11,13,15,2,1,4,3,17,18,19,21,21,22],
                                dtype=np.int32)
        keypoints_openpose = torch.zeros([keypoints_coco.shape[0], 118, 3])
        for k, indice in enumerate(mapping):
            keypoints_openpose[:, k, :] = keypoints_coco[:, indice, :]
        keypoints_openpose[:, 1, 0] = (keypoints_coco[:, 5, 0] + keypoints_coco[:, 6, 0]) / 2
        keypoints_openpose[:, 8, 0] = (keypoints_coco[:, 12, 0] + keypoints_coco[:, 11, 0]) / 2
        keypoints_openpose[:, 25:67, :] = keypoints_coco[:, 91:, :]
        keypoints_openpose[:, 67:, :] = keypoints_coco[:, 40:91, :]
        keypoints_openpose=keypoints_openpose.to(device=device)
        if dataset_to_fitting == 'Phoenix-2014T':
            keypoints_openpose[:, :, 0] *= 1.24


        len_shoulder = keypoints_openpose[:, 5, 0] - keypoints_openpose[:, 2, 0]
        len_waist = len_shoulder / 1.7
        keypoints_openpose[:, 8, 0] = keypoints_openpose[:, 1, 0]
        keypoints_openpose[:, 8, 1] = keypoints_openpose[:, 1, 1] + 1.5 * len_shoulder
        keypoints_openpose[:, 9, 0] = keypoints_openpose[:, 8, 0] - 0.5 * len_waist
        keypoints_openpose[:, 9, 1] = keypoints_openpose[:, 8, 1]
        keypoints_openpose[:, 12, 0] = keypoints_openpose[:, 8, 0] + 0.5 * len_waist
        keypoints_openpose[:, 12, 1] = keypoints_openpose[:, 8, 1]
        keypoints_openpose[:, 10, 0] = keypoints_openpose[:, 9, 0]
        keypoints_openpose[:, 10, 1] = keypoints_openpose[:, 9, 1] + 2. * len_waist
        keypoints_openpose[:, 11, 0] = keypoints_openpose[:, 9, 0]
        keypoints_openpose[:, 11, 1] = keypoints_openpose[:, 9, 1] + 4. * len_waist
        keypoints_openpose[:, 13, 0] = keypoints_openpose[:, 12, 0]
        keypoints_openpose[:, 13, 1] = keypoints_openpose[:, 12, 1] + 2. * len_waist
        keypoints_openpose[:, 14, 0] = keypoints_openpose[:, 12, 0]
        keypoints_openpose[:, 14, 1] = keypoints_openpose[:, 12, 1] + 4. * len_waist
        keypoints_openpose[:, 8:15, 2] = 0.65

        curr_result_folder = osp.join(result_folder, video_ids[i])
        # if not osp.exists(curr_result_folder):
        #     os.makedirs(curr_result_folder)
        curr_mesh_folder = osp.join(result_folder, video_ids[i])
        # if not osp.exists(curr_mesh_folder):
        #     os.makedirs(curr_mesh_folder)
        # if not osp.exists('smplx_debug/images/' + video_ids[i]):
        #     os.makedirs('smplx_debug/images/' + video_ids[i])

        nan_flag = False
        cur_results = []
        for idx, data in enumerate(dataset_obj):
            # if idx == 12:
            #     break
            img = data['img']
            fn = data['fn']
            keypoints = keypoints_openpose[idx]
            keypoints.unsqueeze_(dim=0)

            if is_main_process():
                print('RANK: {}, Processing: {}, {}/{} videos, {}/{} frames'.format(rank, data['img_path'], i, end_idx, idx, len(dataset_obj)))

            for person_id in range(keypoints.shape[0]):

                if person_id >= max_persons and max_persons > 0:
                    continue

                curr_result_fn = osp.join(curr_result_folder,
                                      '{:03d}.pkl'.format(idx))
                curr_mesh_fn = osp.join(curr_mesh_folder,
                                    '{:03d}.obj'.format(idx))

                if gender_lbl_type != 'none':
                    if gender_lbl_type == 'pd' and 'gender_pd' in data:
                        gender = data['gender_pd'][person_id]
                    if gender_lbl_type == 'gt' and 'gender_gt' in data:
                        gender = data['gender_gt'][person_id]
                else:
                    gender = input_gender

                if gender == 'neutral':
                    body_model = neutral_model
                elif gender == 'female':
                    body_model = female_model
                elif gender == 'male':
                    body_model = male_model

                joints_smooth_re, camera_transl_re, camera_orient_re, betas_fix_re, nan_flag, results_frame =\
                    fit_single_frame(img, idx, keypoints,
                             body_model=body_model,
                             camera=camera,
                             joint_weights=joint_weights,
                             dtype=dtype,
                             output_folder=output_folder,
                             result_folder=curr_result_folder,
                             result_fn=curr_result_fn,
                             mesh_fn=curr_mesh_fn,
                             shape_prior=shape_prior,
                             expr_prior=expr_prior,
                             body_pose_prior=body_pose_prior,
                             left_hand_prior=left_hand_prior,
                             right_hand_prior=right_hand_prior,
                             jaw_prior=jaw_prior,
                             angle_prior=angle_prior,
                             joints_smooth=joints_smooth,
                             camera_transl=camera_transl,
                             camera_orient=camera_orient,
                             betas_fix=betas_fix,
                             **args)
                joints_smooth = joints_smooth_re.clone()
                camera_transl = camera_transl_re.clone()
                camera_orient = camera_orient_re.clone()
                betas_fix = betas_fix_re.clone()

                results_frame.append('{:03d}'.format(idx))
                results_frame.append(camera_transl.cpu())
                if nan_flag:
                    # break
                    continue
                cur_results.append(results_frame)
        
        results_all[video_ids[i]] = cur_results

        if (i-start_idx+1)%save_step==0:
            with open(save_fname, 'wb') as f:
                pickle.dump(results_all, f)

        elapsed = time.time() - start
        time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
        if is_main_process():
            print('Processing the data took: {}'.format(time_msg))
        # if nan_flag:
        #     continue
    
    with open(save_fname, 'wb') as f:
        pickle.dump(results_all, f)
    synchronize()
