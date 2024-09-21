from modelling.recognition import RecognitionNetwork
from modelling.translation import TranslationNetwork
from modelling.denoiser import BagDenoiser
from utils.misc import get_logger
import torch


class SignLanguageModel(torch.nn.Module):
    def __init__(self, cfg, cls_num, word_emb_tab=None) -> None:
        super().__init__()
        self.logger = get_logger()
        self.task = cfg.get('task', 'ISLR')
        self.device = cfg['device']
        model_cfg = cfg['model']
        self.frozen_modules = []

        if self.task == 'ISLR':
            self.recognition_network = RecognitionNetwork(
                cfg=model_cfg['RecognitionNetwork'],
                cls_num=cls_num,
                transform_cfg=cfg['data']['transform_cfg'],
                input_streams=cfg['data'].get('input_streams','rgb'),
                input_frames=cfg['data'].get('num_output_frames', 64),
                word_emb_tab=word_emb_tab)
            
            if self.recognition_network.visual_backbone != None:
                self.frozen_modules.extend(self.recognition_network.visual_backbone.get_frozen_layers())
            if self.recognition_network.visual_backbone_keypoint != None:
                self.frozen_modules.extend(self.recognition_network.visual_backbone_keypoint.get_frozen_layers())
            if model_cfg['RecognitionNetwork'].get('only_tune_new_layer', False):
                self.frozen_modules.extend(
                    [self.recognition_network.visual_backbone_twostream.rgb_stream,
                    self.recognition_network.visual_backbone_twostream.pose_stream,
                    self.recognition_network.visual_head,
                    self.recognition_network.visual_head_keypoint])
                for name, params in self.recognition_network.named_parameters():
                    if not 'unified_logits_fc' in name and not 'lateral' in name.lower():
                        params.requires_grad = False
        
        elif self.task == 'G2G':
            input_type = model_cfg['TranslationNetwork'].get('input_type', 'gloss')
            self.input_type = input_type
            self.translation_network = TranslationNetwork(
                input_type=input_type, 
                cfg=model_cfg['TranslationNetwork'],
                task=self.task)
            self.tokenizer = self.translation_network.tokenizer
            if input_type != 'gloss':
                if input_type == 'prob':
                    in_features = cls_num
                elif input_type == 'feature':
                    in_features = 1024*2
                self.mapper = torch.nn.Sequential(
                    torch.nn.Linear(in_features=in_features, out_features=self.translation_network.input_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(in_features=self.translation_network.input_dim, out_features=self.translation_network.input_dim)
                )

        elif self.task == 'bag_denoise':
            self.denoiser = BagDenoiser(cfg=model_cfg['BagDenoiser'],
                                        cls_num=cls_num)

        else:
            raise ValueError


    def forward(self, is_train, labels, sgn_videos, sgn_keypoints, epoch, **kwargs):
        if self.task == 'ISLR':
            model_outputs = self.recognition_network(is_train, labels, sgn_videos, sgn_keypoints, epoch, **kwargs)
        elif self.task == 'G2G':
            translation_inputs = kwargs.pop('translation_inputs', {})
            if self.input_type in ['feature', 'prob']:
                # print(translation_inputs['input_feature'].shape)
                mapped_fea = self.mapper(translation_inputs['input_feature'])
                translation_inputs['input_feature'] = mapped_fea
            model_outputs = self.translation_network(**translation_inputs)
            model_outputs['total_loss'] = model_outputs['translation_loss']
        elif self.task == 'bag_denoise':
            denoise_inputs = kwargs.pop('denoise_inputs', {})
            model_outputs = self.denoiser(**denoise_inputs)
            model_outputs['total_loss'] = model_outputs['denoise_loss']
        return model_outputs
    

    def generate_txt(self, transformer_inputs=None, generate_cfg={}, **kwargs):          
        model_outputs = self.translation_network.generate(**transformer_inputs, **generate_cfg)  
        return model_outputs
    

    def predict_gloss_from_logits(self, gloss_logits, k=10):
        if self.task == 'ISLR':
            return self.recognition_network.decode(gloss_logits=gloss_logits, k=k)
        elif self.task == 'bag_denoise':
            return self.denoiser.decode(logits=gloss_logits, k=k)


    def set_train(self):
        self.train()
        for m in self.frozen_modules:
            m.eval()


    def set_eval(self):
        self.eval()


def build_model(cfg, cls_num, **kwargs):
    model = SignLanguageModel(cfg, cls_num, **kwargs)
    return model.to(cfg['device'])