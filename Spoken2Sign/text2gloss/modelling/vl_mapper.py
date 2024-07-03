import torch, os
from utils.misc import freeze_params, get_logger
from modelling.SpatialEmbeddings import MaskedNorm
class VLMapper(torch.nn.Module):
    def __init__(self, cfg, in_features, out_features,
        gloss_id2str=None,
        gls2embed=None,
        freeze=False) -> None:
        super().__init__()
        logger = get_logger()
        self.type = cfg.get('type','projection')
        if self.type == 'projection':
            self.hidden_size = out_features
            self.mapping = torch.nn.Sequential(
                torch.nn.Linear(in_features=in_features, out_features=self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=self.hidden_size, out_features=out_features)
            )
        elif self.type == 'embedding':
            self.mapping = torch.nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=False)
            #initialize
            assert in_features==len(gloss_id2str), (in_features, gloss_id2str)
            with torch.no_grad():
                for i,s in gloss_id2str.items():
                    if s in gls2embed:
                        self.mapping.weight[:, i] = gls2embed[s]
                    else:
                        logger.info('{} not in gls2embed, set fc to zero'.format(s))
                        self.mapping.weight[:, i] = 0
            if cfg['freeze']:
                logger.info('Freeze parameters in VLMapper ')
                freeze_params(self.mapping)
        elif self.type == 'projection_bn':
            hidden_sizes = cfg.get('hidden_sizes',[])
            assert len(hidden_sizes)>=1, hidden_sizes
            assert out_features==hidden_sizes[-1], (out_features, hidden_sizes[-1])
            #first layer with bn
            self.fc1 = torch.nn.Linear(in_features=in_features, out_features=hidden_sizes[0])
            self.bn = MaskedNorm('sync_batch', num_features=hidden_sizes[0])
            if len(hidden_sizes)>1:
                self.mapping = []
                for i in range(1,len(hidden_sizes)):
                    self.mapping += [
                        torch.nn.ReLU(),
                        torch.nn.Linear(in_features=hidden_sizes[i-1], out_features=hidden_sizes[i])]
                self.mapping = torch.nn.Sequential(*self.mapping)
            else:
                self.mapping = torch.nn.Identity()
        else:
            raise ValueError
        
        if 'pretrained_path' in cfg and os.path.isfile(cfg['pretrained_path']):
            logger.info('Load VL mapper from {}'.format(cfg['pretrained_path']))
            load_dict = torch.load(cfg['pretrained_path'],map_location='cpu')['model_state']
            new_dict = {}
            for k,v in load_dict.items():
                if 'preceding_layer' in k:
                    new_k = k.replace('preceding_layer.layers', 'mapping')
                    new_dict[new_k] = v          
            self.load_state_dict(new_dict)
    
    def forward(self, visual_outputs, lengths=None):
        if self.type=='projection':
            output = self.mapping(visual_outputs['gloss_feature'])
        elif self.type=='embedding':
            output = self.mapping(visual_outputs['gloss_feature']) #-> gloss_feature
        elif self.type=='projection_bn':
            x = visual_outputs['gloss_feature']
            mask = torch.zeros([x.shape[0], torch.max(lengths)],dtype=torch.long, device=x.device)
            for i, len_ in enumerate(lengths):
                mask[i, :len_] = 1
            x = self.fc1(x)
            x = self.bn(x, mask)
            output = self.mapping(x)
        return output



    