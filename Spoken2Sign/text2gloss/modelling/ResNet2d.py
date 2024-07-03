import math
import torch, torchvision
import torch.nn as nn
from utils.misc import get_logger, neq_load_customized
LAYER2NUM = {'layer0':3, 'layer1':4,'layer2':5,'layer3':6,'layer4':7}
class ResNet2d_backbone(torch.nn.Module):
    def __init__(self, pretrained_path,  
            frames_per_batch,
            use_layer=4, freeze_layer=3, out_channel=2048, all_frozen=False):
        super().__init__()
        self.logger = get_logger()
        self.frames_per_batch = frames_per_batch
        res50_model = torchvision.models.resnet50(pretrained=False)
        if pretrained_path!='scratch':
            state_dict = torch.load(pretrained_path)
            res50_model.load_state_dict(state_dict)
            self.logger.info('Load resnet50 from {}'.format(pretrained_path))
        else:
            self.logger.info('Train resnet50 from scratch')
        assert use_layer in [3,4], use_layer
        assert use_layer>=freeze_layer, (use_layer, freeze_layer)

        self.output_dim = 2048 if use_layer==4 else 1024
        self.backbone = nn.Sequential(
            *list(res50_model.children())[:LAYER2NUM[f'layer{freeze_layer}']+1])
        self.res_finetune = nn.Sequential(
            *list(res50_model.children())[LAYER2NUM[f'layer{freeze_layer}']+1:LAYER2NUM[f'layer{use_layer}']+1])
        self.all_frozen = all_frozen
        self.set_frozen_layers()

    def set_train(self):
        self.train()
        for m in getattr(self.backbone,'frozen_modules',[]):
            m.eval()

    def get_frozen_layers(self):
        if self.all_frozen:
            return [self.backbone, self.res_finetune]
        else:
            return self.backbone

    def set_frozen_layers(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.eval()

        if self.all_frozen:
            for name, param in self.res_finetune.named_parameters():
                param.requires_grad = False
            self.res_finetune.eval()            

    def forward(self, sgn_videos, sgn_lengths=None):
        output_dict = {}
        x = torch.transpose(sgn_videos,1,2)#(B, C, T_in, H, W)
        batch_size, num_steps, c, h, w = x.shape 
        if self.frames_per_batch!=-1:
            num_blocks = int(math.ceil(float(num_steps)/self.frames_per_batch))
            backbone_out = []
            for i in range(num_blocks):
                curr_idx = i * self.frames_per_batch
                cur_steps = min(num_steps-curr_idx, self.frames_per_batch)
                curr_data = x[:, curr_idx:curr_idx+cur_steps]
                curr_data = curr_data.contiguous().view(-1, c, h, w)
                self.backbone.eval()
                with torch.no_grad():
                    curr_emb = self.backbone(curr_data)
                if self.all_frozen:
                    self.res_finetune.eval()
                    with torch.no_grad():
                        curr_emb = self.res_finetune(curr_emb)
                else:
                    curr_emb = self.res_finetune(curr_emb)
                _, out_c, out_h, out_w = curr_emb.size()
                curr_emb = curr_emb.contiguous().view(batch_size, cur_steps, out_c, out_h, out_w)
                backbone_out.append(curr_emb) 
            x = torch.cat(backbone_out, dim=1)   #B, T, D, H, W   
        else:
            curr_data = x.contiguous().view(-1, c, h, w)
            self.backbone.eval()
            with torch.no_grad():
                curr_emb = self.backbone(curr_data)
            if self.all_frozen:
                self.res_finetune.eval()
                with torch.no_grad():
                    curr_emb = self.res_finetune(curr_emb)
            else:
                curr_emb = self.res_finetune(curr_emb)
            _, out_c, out_h, out_w = curr_emb.size()
            x = curr_emb.contiguous().view(batch_size, -1, out_c, out_h, out_w)

        #spatial pooling B*view,  T, D, H, W -> B*view, T, D
        x = torch.mean(x, dim=[3,4])  #B, T, D
        B, T_out, _ = x.shape
        valid_len_out = sgn_lengths #B, (undownsampled)
        sgn_mask = torch.zeros([B,1,T_out], dtype=torch.bool, device=x.device)
        for bi in range(B):
            sgn_mask[bi, :, :valid_len_out[bi]] = True        
        return {'sgn_feature': x, 'sgn_mask':sgn_mask, 'valid_len_out':valid_len_out}