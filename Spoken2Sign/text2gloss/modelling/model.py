from modelling.SpatialEmbeddings import SpatialEmbeddings
from modelling.recognition import RecognitionNetwork
from utils.misc import DATASETS, get_logger
from modelling.translation import TranslationNetwork
from modelling.translation_ensemble import TranslationNetwork_Ensemble
from modelling.vl_mapper import VLMapper
from modelling.Tokenizer import GlossTokenizer_S2G
from modelling.S3D import S3D_backbone
import torch
class SignLanguageModel(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.logger = get_logger()
        self.task, self.device = cfg['task'], cfg['device']
        model_cfg = cfg['model']
        self.frozen_modules = []
        if self.task=='S2G':
            self.text_tokenizer = None
            self.recognition_network = RecognitionNetwork(
                cfg=model_cfg['RecognitionNetwork'],
                input_type = 'video',
                transform_cfg={k:v['transform_cfg'] for k,v in cfg['data'].items() if k in DATASETS},
                input_streams = cfg['data'].get('input_streams',['rgb']))
            self.gloss_tokenizer = self.recognition_network.gloss_tokenizer
            #in S2G, some modules in S3D backbone are frozen
            if self.recognition_network.visual_backbone!=None:
                self.frozen_modules.extend(self.recognition_network.visual_backbone.get_frozen_layers())
            if self.recognition_network.visual_backbone_keypoint!=None:
                self.frozen_modules.extend(self.recognition_network.visual_backbone_keypoint.get_frozen_layers())
            if model_cfg['RecognitionNetwork'].get('only_tune_new_layer', False)==True:
                self.frozen_modules.extend(
                    [self.recognition_network.visual_backbone_twostream.rgb_stream,
                    self.recognition_network.visual_backbone_twostream.pose_stream,
                    self.recognition_network.visual_head,
                    self.recognition_network.visual_head_keypoint])
                for name, params in self.recognition_network.named_parameters():
                    if not 'unified_logits_fc' in name and not 'lateral' in name.lower():
                        params.requires_grad = False
            adapter_setting = model_cfg['RecognitionNetwork']['visual_head'].get('adapter', None)
            if adapter_setting is not None and 'freeze_backbone' in adapter_setting:
                self.frozen_modules.append(self.recognition_network.visual_backbone_twostream)
                self.logger.info('freeze two stream backbone')
                if 'freeze_backbone_aux_head' in adapter_setting:
                    self.frozen_modules.extend(self.recognition_network.visual_head_remain)
                    self.frozen_modules.extend(self.recognition_network.visual_head_keypoint_remain)
                    self.logger.info('freeze aux heads')

        if self.task in ['G2T', 'T2G']:
            input_type = model_cfg['TranslationNetwork'].get('input_type', 'gloss')
            self.translation_network = TranslationNetwork(
                input_type=input_type, cfg=model_cfg['TranslationNetwork'],
                task=self.task)
            self.text_tokenizer = self.translation_network.text_tokenizer
            self.gloss_tokenizer = self.translation_network.gloss_tokenizer #G2T

        if self.task=='S2T':
            self.recognition_weight = model_cfg.get('recognition_weight',1)
            self.translation_weight = model_cfg.get('translation_weight',1)
            self.recognition_network = RecognitionNetwork(
                cfg=model_cfg['RecognitionNetwork'],
                input_type = 'feature',
                input_streams = cfg['data'].get('input_streams',['rgb']),
                transform_cfg={k:v['transform_cfg'] for k,v in cfg['data'].items() if k in DATASETS})
            if model_cfg['RecognitionNetwork'].get('freeze', False)==True:
                self.logger.info('freeze recognition_network')
                self.frozen_modules.append(self.recognition_network)
                for param in self.recognition_network.parameters():
                    #print(param)
                    param.requires_grad = False
                self.recognition_network.eval()

            input_type = model_cfg['TranslationNetwork'].pop('input_type','feature')
            self.translation_network = TranslationNetwork(
                input_type=input_type, 
                cfg=model_cfg['TranslationNetwork'], 
                task=self.task)
            self.text_tokenizer = self.translation_network.text_tokenizer
            self.gloss_tokenizer = self.recognition_network.gloss_tokenizer #!!recognition_network.gloss_tokenizer!!
            #check
            if 'gloss+feature ' in input_type:
                assert self.translation_network.gloss_tokenizer.gloss2id==self.recognition_network.gloss_tokenizer.gloss2id
            if model_cfg['VLMapper'].get('type','projection') == 'projection':
                if 'in_features' in model_cfg['VLMapper']:
                    in_features = model_cfg['VLMapper'].pop('in_features')
                else:
                    in_features = model_cfg['RecognitionNetwork']['visual_head']['hidden_size']
            else:
                in_features = len(self.gloss_tokenizer)
            self.vl_mapper = VLMapper(
                cfg=model_cfg['VLMapper'],
                in_features = in_features,
                out_features = self.translation_network.input_dim,
                gloss_id2str=self.gloss_tokenizer.id2gloss,
                gls2embed=getattr(self.translation_network, 'gls2embed', None), 
            )

            only_fine_tune_lm_head = model_cfg['TranslationNetwork'].get('only_fine_tune_lm_head', False)
            if only_fine_tune_lm_head:
                for name, module in self.translation_network.model.named_children():
                    if only_fine_tune_lm_head == True:
                        if 'lm_head' not in name:
                            self.frozen_modules.append(module)
                    elif only_fine_tune_lm_head == 'all':
                        self.frozen_modules.append(module)

        if self.task=='S2T_glsfree':
            self.input_data = cfg['data']['input_data']
            self.translation_network = TranslationNetwork(
                input_type='feature', 
                cfg=model_cfg['TranslationNetwork'], task=self.task)
            if self.input_data=='feature':
                in_features = cfg['data'].get('in_feature', 832)
                self.vl_mapper = VLMapper(
                    cfg=model_cfg['VLMapper'], #
                    in_features=in_features,
                    out_features=self.translation_network.input_dim,
                )
            elif self.input_data=='video':
                self.feature_extractor = S3D_backbone(**cfg['model']['s3d'])
                #output {'sgn_feature':sgn, 'sgn_mask':sgn_mask, 'valid_len_out': valid_len_out}
                in_features = self.feature_extractor.out_features
                self.sgn_embed = SpatialEmbeddings(
                    embedding_dim=self.translation_network.input_dim, 
                    input_size=in_features, norm_type='sync_batch', activation_type='relu')
            else:
                raise ValueError
            self.text_tokenizer = self.translation_network.text_tokenizer
            self.gloss_tokenizer = None

        if self.task=='S2T_Ensemble':
            self.recognition_weight = 0
            self.translation_network = TranslationNetwork_Ensemble(
                cfg=model_cfg['TranslationNetwork_Ensemble']) #to-do TranslationN
            self.text_tokenizer = self.translation_network.text_tokenizer
            self.gloss_tokenizer = None


    def forward(self, is_train, translation_inputs={}, recognition_inputs={}, **kwargs):
        if self.task=='S2G':
            model_outputs = self.recognition_network(is_train=is_train, **recognition_inputs)
            model_outputs['total_loss'] = model_outputs['recognition_loss']            
        elif self.task in ['G2T', 'T2G']:
            model_outputs = self.translation_network(**translation_inputs)
            model_outputs['total_loss'] = model_outputs['translation_loss']
        elif self.task=='S2T':
            recognition_outputs = self.recognition_network(is_train=is_train, **recognition_inputs)
            #'input_lengths': 
            # 'recognition_loss'
            # 'gloss_feature'
            mapped_feature = self.vl_mapper(visual_outputs=recognition_outputs)
            translation_inputs = {
                **translation_inputs,
                'input_feature':mapped_feature, 
                'input_lengths':recognition_outputs['input_lengths']} 

            if self.translation_network.input_type == 'pred_gloss+feature':
                gloss_logits = recognition_outputs['ensemble_last_gloss_logits']
                with torch.no_grad():
                    ctc_decode_output = self.predict_gloss_from_logits(gloss_logits=gloss_logits, 
                        beam_size=1, input_lengths=recognition_outputs['input_lengths'],
                        datasetname=kwargs['datasetname'])
                    max_length = max([len(o) for o in ctc_decode_output])
                    gloss_ids, gloss_lengths = [], []
                    for ii, ids in enumerate(ctc_decode_output):
                        gloss_ids.append(ids+[0]*(max_length-len(ids)))
                        gloss_lengths.append(len(ids))
                    gloss_ids = torch.tensor(gloss_ids, dtype=torch.long, device=gloss_logits.device)
                    gloss_lengths =  torch.tensor(gloss_lengths, dtype=torch.long, device=gloss_logits.device)
                translation_inputs['gloss_ids'] = gloss_ids
                translation_inputs['gloss_lengths'] = gloss_lengths  
            translation_outputs = self.translation_network(**translation_inputs)
            model_outputs = {**translation_outputs, **recognition_outputs}
            model_outputs['transformer_inputs'] = model_outputs['transformer_inputs'] #for latter use of decoding
            model_outputs['total_loss'] = \
                model_outputs['recognition_loss']*self.recognition_weight + \
                model_outputs['translation_loss']*self.translation_weight 
        
        elif self.task=='S2T_glsfree':
            model_outputs = {}
            if self.input_data=='feature':
                input_feature = recognition_inputs['sgn_features']
                input_lengths = recognition_inputs['sgn_lengths']
                mapped_feature = self.vl_mapper(
                    visual_outputs={'gloss_feature':input_feature},
                    lengths = input_lengths)
            elif self.input_data=='video':
                model_outputs = self.feature_extractor(**recognition_inputs)
                input_feature = model_outputs['sgn_feature']
                input_lengths = model_outputs['valid_len_out']
                mapped_feature = self.sgn_embed(x=input_feature, lengths=input_lengths)         
            translation_inputs = {
                **translation_inputs,
                'input_feature': mapped_feature, #
                'input_lengths': input_lengths}
            translation_outputs = self.translation_network(**translation_inputs)
            model_outputs = {**model_outputs,**translation_outputs}
            model_outputs['transformer_inputs'] = model_outputs['transformer_inputs']
            model_outputs['total_loss'] = model_outputs['translation_loss']
        
        elif self.task=='S2T_Ensemble':
            assert 'inputs_embeds_list' in translation_inputs and 'attention_mask_list' in translation_inputs
            assert len(translation_inputs['inputs_embeds_list'])==len(self.translation_network.model.model_list)
            model_outputs = self.translation_network(**translation_inputs)
        return model_outputs

    
    def generate_txt(self, transformer_inputs=None, generate_cfg={}, **kwargs):          
        #transformer_inputs the same with forward
        #some keys are not used (labels, decoder_input_ids) this will be automatically ignored in generate
        #'inputs_embeds' 'attention_mask'!
        model_outputs = self.translation_network.generate(**transformer_inputs, **generate_cfg)  
        return model_outputs
    
    def predict_gloss_from_logits(self, gloss_logits, beam_size, input_lengths, datasetname, lm=None, alpha=0):
        return self.recognition_network.decode(
            gloss_logits=gloss_logits,
            beam_size=beam_size,
            input_lengths=input_lengths,
            datasetname=datasetname,
            lm=lm,
            alpha=alpha)

    def set_train(self):
        self.train()
        for m in self.frozen_modules:
            # m.eval()
            # print('freeze ', m)
            m.requires_grad_(False)

    def set_eval(self):
        self.eval()

def build_model(cfg):
    model = SignLanguageModel(cfg)
    return model.to(cfg['device'])