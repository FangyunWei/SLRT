import torch, os
from utils.misc import freeze_params, get_logger

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
    
    def forward(self, visual_outputs, lengths=None):
        if self.type=='projection':
            output = self.mapping(visual_outputs['gloss_feature'])
        elif self.type=='embedding':
            output = self.mapping(visual_outputs['gloss_feature'])
        else:
            raise ValueError
        return output



    