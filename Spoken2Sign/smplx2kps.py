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
import pickle, json
import torch
import smplx

from cmd_parser import parse_config
from human_body_prior.tools.model_loader import load_vposer

from utils import JointMapper
import pyrender
import trimesh
import numpy as np
import PIL.Image as pil_img
import cv2
from numpy import pi
import math
import torch.nn.functional as F
from tqdm import tqdm
import utils
from plot_skeletons import draw_frame_2D_openpose


def imshow_keypoints(img,
                     pose_result,
                     skeleton=None,
                     kpt_score_thr=0.3,
                     pose_kpt_color=None,
                     pose_link_color=None,
                     radius=2,
                     thickness=1,
                     show_keypoint_weight=False):

    img_h, img_w, _ = img.shape

    skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                [8, 10], [1, 2], [0, 1], [0, 2],
                [1, 3], [2, 4], [3, 5], [4, 6], [15, 17], [15, 18],
                [15, 19], [16, 20], [16, 21], [16, 22], [91, 92],
                [92, 93], [93, 94], [94, 95], [91, 96], [96, 97],
                [97, 98], [98, 99], [91, 100], [100, 101], [101, 102],
                [102, 103], [91, 104], [104, 105], [105, 106],
                [106, 107], [91, 108], [108, 109], [109, 110],
                [110, 111], [112, 113], [113, 114], [114, 115],
                [115, 116], [112, 117], [117, 118], [118, 119],
                [119, 120], [112, 121], [121, 122], [122, 123],
                [123, 124], [112, 125], [125, 126], [126, 127],
                [127, 128], [112, 129], [129, 130], [130, 131],
                [131, 132]]

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255],
                        [255, 0, 0], [255, 255, 255]])

    pose_link_color = palette[[
                                  0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
                              ] + [16, 16, 16, 16, 16, 16] + [
                                  0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                                  16
                              ] + [
                                  0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                                  16
                              ]]

    pose_kpt_color = palette[
        [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0] +
        [0, 0, 0, 0, 0, 0] + [19] * (68 + 42)]


    for ii in range(1):

        kpts = np.array(pose_result, copy=False)

        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)

            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]

                if kpt_score < kpt_score_thr or pose_kpt_color[kid] is None:
                    # skip the point that should not be drawn
                    continue

                color = tuple(int(c) for c in pose_kpt_color[kid])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    cv2.circle(img_copy, (int(x_coord), int(y_coord)), radius,
                               color, -1)
                    transparency = max(0, min(1, kpt_score))
                    cv2.addWeighted(
                        img_copy,
                        transparency,
                        img,
                        1 - transparency,
                        0,
                        dst=img)
                else:
                    cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                               color, -1)

        # draw links
        if skeleton is not None and pose_link_color is not None:
            assert len(pose_link_color) == len(skeleton)

            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                        or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
                        or pos2[1] <= 0 or pos2[1] >= img_h
                        or kpts[sk[0], 2] < kpt_score_thr
                        or kpts[sk[1], 2] < kpt_score_thr
                        or pose_link_color[sk_id] is None):
                    # skip the link that should not be drawn
                    continue
                color = tuple(int(c) for c in pose_link_color[sk_id])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    X = (pos1[0], pos2[0])
                    Y = (pos1[1], pos2[1])
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    stickwidth = 2
                    polygon = cv2.ellipse2Poly(
                        (int(mX), int(mY)), (int(length / 2), int(stickwidth)),
                        int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(img_copy, polygon, color)
                    transparency = max(
                        0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
                    cv2.addWeighted(
                        img_copy,
                        transparency,
                        img,
                        1 - transparency,
                        0,
                        dst=img)
                else:
                    cv2.line(img, pos1, pos2, color, thickness=thickness)

    return img

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


if __name__ == '__main__':
    # os.environ['PYOPENGL_PLATFORM'] = 'egl'
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
                                     use_face_contour=args['use_face_contour'], openpose_format='coco25')
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

    # with open(args['data_subset'], 'rb') as f:
    #     video_ids = pickle.load(f, encoding='latin1')
    with open('../../data/phoenix_2014t/PT/name_lst.json', 'r') as f:
        video_ids = json.load(f)
        video_ids = [x[:-4] for x in video_ids]
    with open('../../data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody.pkl', 'rb') as f:
        all_kps = pickle.load(f)
    
    if 'unwanted' in args['data_subset']:
        res_file = '{}_unwanted_all.pkl'.format(args['dataset_to_fitting'])
    else:
        res_file = '{}_all.pkl'.format(args['dataset_to_fitting'])

    param_file = args['param_file']
    if param_file is None:
        param_file = '{}/{}'.format(args['output_folder'], res_file)

    with open(param_file, 'rb') as f:
    # with open('../../data/phoenix_syn/Phoenix-2014T_all.pkl', 'rb') as f:
        render_results_all = pickle.load(f)

    init_idx = args['init_idx']
    num_per_proc = args['num_per_proc']
    start_idx = init_idx
    end_idx = start_idx + num_per_proc
    video_ids = video_ids[start_idx:end_idx]
    render_results_all = {vid: render_results_all[vid] for vid in video_ids if vid in render_results_all}
    # render_results_all = {video_ids[0]: render_results_all}

    if args['dataset_to_fitting'] == 'Phoenix-2014T':
        H, W = 260, 210
        root_dir = '../../data/phoenix_syn'
        camera_transl = [0., -0.7, 20.]  #[0., -0.7, 24.]
    elif args['dataset_to_fitting'] == 'CSL-Daily':
        H, W = 512, 512
        root_dir = '../../data/csl-daily_syn'
        camera_transl = [0., -0.7, 12.]
    elif args['dataset_to_fitting'] == 'WLASL':
        H, W = 256, 256
        root_dir = '../../data/wlasl_syn'
        camera_transl = [0., -0.7, 24.]
    elif args['dataset_to_fitting'] == 'MSASL':
        H, W = 256, 256
        root_dir = '../../data/msasl_syn'
        camera_transl = [0., -0.7, 24.]
    else:
        raise ValueError
    camera_transl[2] *= -1.
    os.makedirs(root_dir, exist_ok=True)

    kps_3d, kps_2d = {}, {}
    kps_3d_path = os.path.join(root_dir, 'keypoints_3d_{}_{:06d}_{:06d}.pkl'.format(args['identifier'], start_idx, end_idx))
    kps_2d_path = os.path.join(root_dir, 'keypoints_2d_{}_{:06d}_{:06d}.pkl'.format(args['identifier'], start_idx, end_idx))
    if os.path.exists(kps_3d_path):
        with open(kps_3d_path, 'rb') as f:
            kps_3d = pickle.load(f)
        with open(kps_2d_path, 'rb') as f:
            kps_2d = pickle.load(f)
        assert len(kps_3d) == len(kps_2d) or len(kps_3d) == len(kps_2d)+2

    tot_step, save_step = 0, 200
    if bool(args['only_kps']):
        save_step = 100
    for video_id in tqdm(video_ids, desc='Processing Video'):
        if video_id in kps_3d or video_id not in render_results_all:
            continue
        data = render_res = render_results_all[video_id]
        if len(data) <= 0:
            continue
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

        kps_2d_vid, kps_3d_vid = [], []
        for idx, data in enumerate(render_res):
            num = data[-2]
            if use_vposer:
                with torch.no_grad():
                    pose_embedding[:] = torch.tensor(
                        est_params['body_pose'][idx], device=device, dtype=dtype)

            est_params_i = {}
            for key, val in data[0]['result'].items():
                if key == 'body_pose' and use_vposer:
                    body_pose = vposer.decode(
                        pose_embedding, output_type='aa').view(1, -1)
                    if model_type == 'smpl':
                        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                                dtype=body_pose.dtype,
                                                device=body_pose.device)
                        body_pose = torch.cat([body_pose, wrist_pose], dim=1)

                    est_params_i['body_pose'] = body_pose
                elif key == 'betas':
                    est_params_i[key] = torch.zeros([1, 10], dtype=dtype, device=device)
                else:
                    est_params_i[key] = torch.tensor(est_params[key][idx], dtype=dtype, device=device)

            rad_angle = pi*args['aug_angle']/180
            est_params_i['global_orient'][..., -1] = est_params_i['global_orient'][..., -1] + rad_angle
            model_output = model(**est_params_i)

            if not bool(args['only_kps']):
                vertices = model_output.vertices.detach().cpu().numpy().squeeze()
                # print(vertices.shape)

                out_mesh = trimesh.Trimesh(vertices, model.faces, process=False)
                # material = trimesh.visual.texture.SimpleMaterial(image=tex_img)
                # texture = trimesh.visual.texture.TextureVisuals(
                #         uv=uv,
                #         image=tex_img,
                #         material=material)
                # out_mesh.visual = texture

                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0,
                    alphaMode='OPAQUE',
                    baseColorFactor=(1.0, 1.0, 0.9, 1.0)
                    # baseColorFactor=(0.3, 0.3, 0.3, 1.0)
                    )
                # tex = pyrender.Texture(source=tex_img, source_channels='RGB')
                # material = pyrender.material.MetallicRoughnessMaterial(baseColorTexture=tex)
                mesh = pyrender.Mesh.from_trimesh(
                    out_mesh,
                    material=material)

                scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                    ambient_light=(0.3, 0.3, 0.3))
                light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=5e2)
                scene.add(mesh, 'mesh')
                pose = np.eye(4)
                pose = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
                scene.add(light, pose=pose)

            camera_center = [1.0*W/2, 1.0*H/2]
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_transl
            camera_pose[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]

            focal_length = 5000
            camera = pyrender.camera.IntrinsicsCamera(
                fx=focal_length, fy=focal_length,
                cx=camera_center[0], cy=camera_center[1])
                
            joints_3D = model_output.joints
            proj_matrix = camera.get_projection_matrix(W, H)
            joints_3D = F.pad(joints_3D.squeeze(0), [0,1], value=1).detach().cpu().numpy()
            joints_2D = proj_matrix @ np.linalg.inv(camera_pose) @ joints_3D.T
            joints_2D = joints_2D/joints_2D[3]
            joints_2D[0] = W/2*joints_2D[0] + W/2
            joints_2D[1] = H - (H/2*joints_2D[1] + H/2)
            joints_2D = joints_2D[:2,:].T

            mapping = np.array(
                [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11, 19, 20, 21, 22, 23, 24],
                dtype=np.int32)

            keypoints_coco = np.zeros([133, 3], dtype=np.float16)
            keypoints_coco_3d = np.zeros([133, 3], dtype=np.float32)
            for k, indice in enumerate(mapping):
                keypoints_coco[k, :2] = joints_2D[indice, :]
                keypoints_coco[k, -1] = 1
                keypoints_coco_3d[k, :] = joints_3D[indice, :3]
            keypoints_coco[91:, :2] = joints_2D[25:67, :]
            keypoints_coco[91:, -1] = 1
            keypoints_coco_3d[91:, :] = joints_3D[25:67, :3]
            keypoints_coco[40:91, :2] = joints_2D[67:118, :]
            keypoints_coco[40:91, -1] = 1
            keypoints_coco_3d[40:91, :] = joints_3D[67:118, :3]
            keypoints_coco[23:40, :2] = joints_2D[118:, :]
            keypoints_coco[23:40, -1] = 1
            keypoints_coco_3d[23:40, :] = joints_3D[118:, :3]
            keypoints_coco[11:23, :] = 0.

            save_dir = os.path.join(f'{root_dir}/kp_images_eccv', video_id)
            os.makedirs(save_dir, exist_ok=True)
            frame = np.zeros((650, 525, 3), dtype=np.uint8)
            frame = draw_frame_2D_openpose(frame, keypoints_coco, H=H, W=W)
            cv2.imwrite('{}/{:03d}.png'.format(save_dir, idx), frame)

            # print(joints_2D)
            kps_2d_vid.append(keypoints_coco)
            kps_3d_vid.append(keypoints_coco_3d)
            # joints_2D = camera(joints_3D)
            # joints_2D = joints_2D.detach().cpu().numpy()

            if not bool(args['only_kps']):
                scene.add(camera, pose=camera_pose)
                registered_keys = dict()
                viewer = pyrender.Viewer(scene, use_raymond_lighting=True,
                                        viewport_size=(W, H),
                                        cull_faces=False,
                                        run_in_thread=True,
                                        registered_keys=registered_keys)

                light_nodes = viewer._create_raymond_lights()
                for node in light_nodes:
                    scene.add_node(node)
                r = None
                r = pyrender.OffscreenRenderer(viewport_width=W,
                                            viewport_height=H,
                                            point_size=1.0)
                color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)

                color = pil_img.fromarray(color)
                img_dir = os.path.join(root_dir, 'images_{}'.format(args['identifier']), video_id)
                os.makedirs(img_dir, exist_ok=True)
                color.save(os.path.join(img_dir, '{}.png'.format(num)))
                viewer.on_close()
            # print(idx)

        kps_3d[video_id] = {'keypoints_3d': np.stack(kps_3d_vid, axis=0)}
        kps_2d[video_id] = {'keypoints': np.stack(kps_2d_vid, axis=0)}

        tot_step += 1
        if tot_step % save_step == 0:
            with open(kps_3d_path, 'wb') as f:
                pickle.dump(kps_3d, f)
            with open(kps_2d_path, 'wb') as f:
                pickle.dump(kps_2d, f)
    
    if 'proj_matrix' not in kps_3d:
        kps_3d['proj_matrix'] = proj_matrix
        kps_3d['camera_pose'] = camera_pose
    with open(kps_3d_path, 'wb') as f:
        pickle.dump(kps_3d, f)
    with open(kps_2d_path, 'wb') as f:
        pickle.dump(kps_2d, f)