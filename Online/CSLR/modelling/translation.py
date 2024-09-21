import torch
from transformers import MBartForConditionalGeneration, MBartConfig
from utils.misc import freeze_params, get_logger
from utils.loss import XentLoss
from modelling.tokenizer import GlossTokenizer_G2G
import pickle, math

class TranslationNetwork(torch.nn.Module):
    def __init__(self, input_type, cfg, task) -> None:
        super().__init__()
        self.logger = get_logger()
        self.task = task
        self.input_type = input_type
        assert self.input_type in ['gloss', 'feature', 'prob']
        self.tokenizer = GlossTokenizer_G2G(tokenizer_cfg=cfg['Tokenizer'])

        if 'pretrained_model_name_or_path' in cfg:
            self.logger.info('Initialize translation network from {}'.format(cfg['pretrained_model_name_or_path']))
            self.model = MBartForConditionalGeneration.from_pretrained(
                cfg['pretrained_model_name_or_path'],
                vocab_size=len(self.tokenizer.gloss2id),
                ignore_mismatched_sizes=True
            )
        else:
            self.logger.info('Train mBart from scratch!')
            config = MBartConfig()
            config.vocab_size = len(self.tokenizer.gloss2id)
            self.model = MBartForConditionalGeneration(config)

        self.translation_loss_fun = XentLoss(
            pad_index=self.tokenizer.pad_index, 
            smoothing=cfg['label_smoothing'])
        self.input_dim = self.model.config.d_model
        self.input_embed_scale = cfg.get('input_embed_scale', math.sqrt(self.model.config.d_model))

        if self.task == 'G2G':# and 'pretrained_model_name_or_path' in cfg:
            self.gloss_embedding = self.build_gloss_embedding(**cfg['GlossEmbedding'])
            #debug
            self.gls_eos = cfg.get('gls_eos', 'gls') #gls or txt
        else:
            raise ValueError

        if cfg.get('from_scratch', False):
            self.model.init_weights()
            self.logger.info('Train Translation Network from scratch!')
        if cfg.get('freeze_txt_embed', False):
            freeze_params(self.model.model.shared)
            self.logger.info('Set txt embedding frozen')

        if 'load_ckpt' in cfg:
            self.load_from_pretrained_ckpt(cfg['load_ckpt'])


    def load_from_pretrained_ckpt(self, pretrained_ckpt):
        logger = get_logger()
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')['model_state']
        load_dict = {}
        for k,v in checkpoint.items():
            if 'translation_network' in k:
                load_dict[k.replace('translation_network.','')] = v
        self.load_state_dict(load_dict)
        logger.info('Load Translation network from pretrained ckpt {}'.format(pretrained_ckpt))


    def build_gloss_embedding(self, gloss2embed_file, from_scratch=False, freeze=False):
        gloss_embedding = torch.nn.Embedding(
                num_embeddings=len(self.tokenizer.id2gloss),
                embedding_dim=self.model.config.d_model,
                padding_idx=self.tokenizer.pad_index) 
        self.logger.info('gloss2embed_file '+ gloss2embed_file)
        if from_scratch:
            self.logger.info('Train Gloss Embedding from scratch')
            assert freeze==False
        else:
            gls2embed = torch.load(gloss2embed_file)
            self.gls2embed = gls2embed
            self.logger.info('Initialize gloss embedding from {}'.format(gloss2embed_file))
            with torch.no_grad():
                for id_, gls in self.tokenizer.id2gloss.items():
                    if gls in gls2embed:
                        assert gls in gls2embed, gls
                        gloss_embedding.weight[id_,:] = gls2embed[gls]
                    else:
                        self.logger.info('{} not in gls2embed train from scratch'.format(gls))

        if freeze:
            freeze_params(gloss_embedding)  
            self.logger.info('Set gloss embedding frozen')
        return gloss_embedding
    

    def prepare_gloss_inputs(self, input_ids):
        input_emb = self.gloss_embedding(input_ids)*self.input_embed_scale
        return input_emb


    def prepare_feature_inputs(self, input_feature, input_lengths, gloss_embedding=None, gloss_lengths=None):
        suffix_embedding = [self.gloss_embedding.weight[self.tokenizer.eos_index,:]]

        if self.task == 'G2G' and self.gloss_embedding:
            src_lang_code_embedding = self.gloss_embedding.weight[self.tokenizer.convert_tokens_to_ids(self.tokenizer.src_lang),:] #to-debug
            suffix_embedding.append(src_lang_code_embedding)
        suffix_len = len(suffix_embedding)
        suffix_embedding = torch.stack(suffix_embedding, dim=0)

        max_length = torch.max(input_lengths) + suffix_len
        inputs_embeds = []
        attention_mask = torch.zeros([input_feature.shape[0], max_length], dtype=torch.long, device=input_feature.device)

        for ii, feature in enumerate(input_feature):
            valid_len = input_lengths[ii] 
            valid_feature = feature[:valid_len,:] #t,D
            if suffix_embedding != None:
                feature_w_suffix = torch.cat([valid_feature, suffix_embedding], dim=0) # t+2, D
            else:
                feature_w_suffix = valid_feature
            if feature_w_suffix.shape[0]<max_length:
                pad_len = max_length-feature_w_suffix.shape[0]
                padding = torch.zeros([pad_len, feature_w_suffix.shape[1]], 
                    dtype=feature_w_suffix.dtype, device=feature_w_suffix.device)
                padded_feature_w_suffix = torch.cat([feature_w_suffix, padding], dim=0) #t+2+pl,D
                inputs_embeds.append(padded_feature_w_suffix)
            else:
                inputs_embeds.append(feature_w_suffix)
            attention_mask[ii, :valid_len+suffix_len] = 1
        transformer_inputs = {
            'inputs_embeds': torch.stack(inputs_embeds, dim=0)*self.input_embed_scale, #B,T,D
            'attention_mask': attention_mask #attention_mask
        }
        return transformer_inputs


    def forward(self, **kwargs):
        if self.input_type == 'gloss':
            input_ids = kwargs.pop('input_ids')
            kwargs['inputs_embeds'] = self.prepare_gloss_inputs(input_ids)
        elif self.input_type in ['feature', 'prob']:
            input_feature = kwargs.pop('input_feature')
            input_lengths = kwargs.pop('input_lengths')
            #quick fix
            kwargs.pop('gloss_ids', None)
            kwargs.pop('gloss_lengths', None)
            new_kwargs = self.prepare_feature_inputs(input_feature, input_lengths)
            kwargs = {**kwargs, **new_kwargs}
        else:
            raise ValueError
        
        output_dict = self.model(**kwargs, return_dict=True)
        log_prob = torch.nn.functional.log_softmax(output_dict['logits'], dim=-1)  # B, T, L
        batch_loss_sum = self.translation_loss_fun(log_probs=log_prob,targets=kwargs['labels'])
        output_dict['translation_loss'] = batch_loss_sum/log_prob.shape[0]
        output_dict['transformer_inputs'] = kwargs #for later use (decoding)
        return output_dict


    def generate(self, input_ids=None, attention_mask=None, #decoder_input_ids,
                inputs_embeds=None, input_lengths=None,
                num_beams=4, max_length=100, length_penalty=1, **kwargs):
        assert attention_mask != None
        batch_size = attention_mask.shape[0]
        decoder_input_ids = torch.ones([batch_size,1], dtype=torch.long, device=attention_mask.device) * self.tokenizer.sos_index
        assert inputs_embeds != None and attention_mask != None
        output_dict = self.model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, #same with forward 
            decoder_input_ids=decoder_input_ids,
            num_beams=num_beams, length_penalty=length_penalty, max_length=max_length, 
            return_dict_in_generate=True)
        output_dict['decoded_sequences'] = self.tokenizer.batch_decode(output_dict['sequences'])
        return output_dict
