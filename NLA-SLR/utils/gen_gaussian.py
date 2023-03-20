import torch


def gen_a_limb_heatmap(arr, starts, ends, sigma=1.0):
    # https://github.com/kennymckormick/pyskl/blob/main/pyskl/datasets/pipelines/heatmap_related.py
    """Generate pseudo heatmap for one limb in one frame.
    Args:
        arr: The array to store the generated heatmaps. Shape: img_h * img_w.
        starts: The coordinates of one keypoint in the corresponding limbs. Shape: M * 2.
        ends: The coordinates of the other keypoint in the corresponding limbs. Shape: M * 2.
    Returns:
        The generated pseudo heatmap.
    """

    img_h, img_w = arr.shape

    for start, end in zip(starts, ends):
        min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
        min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

        min_x = max(int(min_x - 3 * sigma), 0)
        max_x = min(int(max_x + 3 * sigma) + 1, img_w)
        if max_x <= min_x:
            max_x = min_x + 1
        min_y = max(int(min_y - 3 * sigma), 0)
        max_y = min(int(max_y + 3 * sigma) + 1, img_h)
        if max_y <= min_y:
            max_y = min_y + 1

        # min_x = min_y= 0; max_x = max_y = 112
        x = torch.arange(min_x, max_x, 1).float().to(start.device)
        y = torch.arange(min_y, max_y, 1).float().to(start.device)

        if not (len(x) and len(y)):
            continue
        
        y = y[:, None]
        x_0 = torch.zeros_like(x)
        y_0 = torch.zeros_like(y)

        # distance to start keypoints
        d2_start = ((x - start[0])**2 + (y - start[1])**2)

        # distance to end keypoints
        d2_end = ((x - end[0])**2 + (y - end[1])**2)

        # the distance between start and end keypoints.
        d2_ab = ((start[0] - end[0])**2 + (start[1] - end[1])**2)
        d2_ab = max(d2_ab, 1.0)

        coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

        a_dominate = (coeff <= 0).float()
        b_dominate = (coeff >= 1).float()
        seg_dominate = 1 - a_dominate - b_dominate

        position = torch.stack([x + y_0, y + x_0], dim=-1)
        projection = start + torch.stack([coeff, coeff], dim=-1) * (end - start)
        d2_line = position - projection
        d2_line = d2_line[:, :, 0]**2 + d2_line[:, :, 1]**2
        d2_seg = a_dominate * d2_start + b_dominate * d2_end + seg_dominate * d2_line

        patch = torch.exp(-d2_seg / 2. / sigma**2)

        arr[min_y:max_y, min_x:max_x] = torch.maximum(arr[min_y:max_y, min_x:max_x], patch)


def gen_gaussian_hmap_op(coords, raw_size=(260,210), map_size=None, sigma=1, threshold=0, **kwargs):
    # openpose version
    # pose [T,18,3]; face [T,70,3]; hand_0(left) [T,21,3]; hand_1(right) [T,21,3]
    # gamma: hyper-param, control the width of gaussian, larger gamma, SMALLER gaussian
    # flags: use pose or face or hands or some of them

    #coords T, C, 3

    T, hmap_num = coords.shape[:2] 
    raw_h, raw_w = raw_size #260,210
    if map_size==None:
        map_h, map_w = raw_h, raw_w
        factor_h, factor_w = 1, 1
    else:
        map_h, map_w = map_size
        factor_h, factor_w = map_h/raw_h, map_w/raw_w
    # generate 2d coords
    # NOTE: openpose generate opencv-style coordinates!
    coords_y =  coords[..., 1]*factor_h
    coords_x = coords[..., 0]*factor_w
    confs = coords[..., 2] #T, C
    
    # if not limb:
    y, x = torch.meshgrid(torch.arange(map_h), torch.arange(map_w))
    coords = torch.stack([coords_y, coords_x], dim=0)  
    grid = torch.stack([y,x], dim=0).to(coords.device)  #[2,H,W]
    grid = grid.unsqueeze(0).unsqueeze(0).expand(hmap_num,T,-1,-1,-1)  #[C,T,2,H,W]
    coords = coords.unsqueeze(0).unsqueeze(0).expand(map_h, map_w,-1,-1,-1).permute(4,3,2,0,1)  #[C,T,2,H,W]
    hmap = torch.exp(-((grid-coords)**2).sum(dim=2) / (2*sigma**2))  #[C,T,H,W]
    hmap = hmap.permute(1,0,2,3)  #[T,C,H,W]
    if threshold > 0:
        confs = confs.unsqueeze(-1).unsqueeze(-1) #T,C,1,1
        confs = torch.where(confs>threshold, confs, torch.zeros_like(confs))
        hmap = hmap*confs
    
    center = kwargs.pop('center', None)
    if center is not None:
        # generate shifted heatmaps
        rela_hmap_lst = []
        for cen in center:
            if cen == 'nose':
                cen_y = coords_y[..., 52]
                cen_x = coords_x[..., 52]
            elif cen == 'shoulder_mid':
                right_y, right_x = coords_y[..., 57], coords_x[..., 57]
                left_y, left_x = coords_y[..., 58], coords_x[..., 58]
                cen_y, cen_x = (right_y+left_y)/2., (right_x+left_x)/2.
            c_y = (coords_y - cen_y.unsqueeze(1) + map_h) / 2.
            c_x = (coords_x - cen_x.unsqueeze(1) + map_w) / 2.
            coords = torch.stack([c_y, c_x], dim=0)
            coords = coords.unsqueeze(0).unsqueeze(0).expand(map_h, map_w,-1,-1,-1).permute(4,3,2,0,1)
            rela_hmap = torch.exp(-((grid-coords)**2).sum(dim=2) / (2*sigma**2))
            rela_hmap = rela_hmap.permute(1,0,2,3)  #[T,C,H,W]
            rela_hmap_lst.append(rela_hmap)
            # print(cen)
        
        if kwargs.pop('rela_comb', False):
            rela_hmap = torch.cat(rela_hmap_lst, dim=1)
            hmap = torch.cat([hmap, rela_hmap], dim=1)
            hmap = hmap.view(T, len(center)+1, hmap_num, map_h, map_w).permute(0,2,1,3,4).reshape(T, hmap_num*(len(center)+1), map_h, map_w)
            # print('rela_comb')
        else:
            hmap = rela_hmap_lst[0]
    
    # else:
    #     hmap = []
    #     for i in range(hmap_num):
    #         c_y = coords_y[:, i]
    #         c_x = coords_x[:, i]
    #         start_y, start_x = c_y[:-1], c_x[:-1]
    #         end_y, end_x = c_y[1:], c_x[1:]
    #         starts = torch.stack([start_x, start_y], dim=-1)
    #         ends = torch.stack([end_x, end_y], dim=-1)

    #         arr = torch.zeros((map_h, map_w)).to(coords_y.device)
    #         gen_a_limb_heatmap(arr, starts, ends, sigma)
    #         hmap.append(arr)
    #     hmap = torch.stack(hmap, dim=0)

    temp_merge = kwargs.pop('temp_merge', False)
    if temp_merge: # and not limb:
        hmap = hmap.amax(dim=0)  #[C,H,W]

    return hmap
