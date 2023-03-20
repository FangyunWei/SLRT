
   
import torch

def gen_gaussian_hmap_op(coords, raw_size=(260,210), map_size=None, sigma=1, confidence=False, threshold=0, **kwargs):
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



    y, x = torch.meshgrid(torch.arange(map_h), torch.arange(map_w))
    coords = torch.stack([coords_y, coords_x], dim=0)  
    grid = torch.stack([y,x], dim=0).to(coords.device)  #[2,H,W]
    grid = grid.unsqueeze(0).unsqueeze(0).expand(hmap_num,T,-1,-1,-1)  #[C,T,2,H,W]
    coords = coords.unsqueeze(0).unsqueeze(0).expand(map_h, map_w,-1,-1,-1).permute(4,3,2,0,1)  #[C,T,2,H,W]
    hmap = torch.exp(-((grid-coords)**2).sum(dim=2) / (2*sigma**2))  #[C,T,H,W]
    hmap = hmap.permute(1,0,2,3)  #[T,C,H,W]
    if confidence:
        confs = confs.unsqueeze(-1).unsqueeze(-1) #T,C,1,1
        confs = torch.where(confs>threshold, confs, torch.zeros_like(confs))
        hmap = hmap*confs

    return hmap
