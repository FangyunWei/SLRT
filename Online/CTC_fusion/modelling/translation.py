import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig
from utils.misc import freeze_params, get_logger
from utils.loss import XentLoss
from .Tokenizer import GlossTokenizer_G2T, TextTokenizer
import pickle, math

class TranslationNetwork(torch.nn.Module):
    def __init__(self, input_type, cfg, task) -> None:
        super().__init__()
        self.logger = get_logger()
        self.task = task
        self.input_type = input_type #gloss or feature
        assert self.input_type in ['gloss','feature','gt_gloss+feature','pred_gloss+feature']
        self.text_tokenizer = TextTokenizer(tokenizer_cfg=cfg['TextTokenizer'])

        if 'pretrained_model_name_or_path' in cfg:
            self.logger.info('Initialize translation network from {}'.format(cfg['pretrained_model_name_or_path']))
            self.model = MBartForConditionalGeneration.from_pretrained(
                cfg['pretrained_model_name_or_path'],
                **cfg.get('overwrite_cfg', {}) 
            )
        elif 'model_config' in cfg:
            self.logger.info('Train translation network from scratch using config={}'.format(cfg['model_config']))
            config = MBartConfig.from_pretrained(cfg['model_config'])
            for k,v in cfg.get('overwrite_cfg', {}).items():
                setattr(config, k, v)
                self.logger.info('Overwrite {}={}'.format(k,v))
            if cfg['TextTokenizer'].get('level','sentencepiece') == 'word':
                setattr(config, 'vocab_size', len(self.text_tokenizer.id2token))
                self.logger.info('Vocab_size {}'.format(config.vocab_size))
            self.model = MBartForConditionalGeneration(config=config)

            if 'pretrained_pe' in cfg:
                pe = torch.load(cfg['pretrained_pe']['pe_file'], map_location='cpu')
                self.logger.info('Load pretrained positional embedding from ', cfg['pretrained_pe']['pe_file'])
                with torch.no_grad():
                    self.model.model.encoder.embed_positions.weight = torch.nn.parameter.Parameter(pe['model.encoder.embed_positions.weight'])
                    self.model.model.decoder.embed_positions.weight = torch.nn.parameter.Parameter(pe['model.decoder.embed_positions.weight'])
                if cfg['pretrained_pe']['freeze']:
                    self.logger.info('Set positional embedding frozen')
                    freeze_params(self.model.model.encoder.embed_positions)
                    freeze_params(self.model.model.decoder.embed_positions)
                else:
                    self.logger.info('Set positional embedding trainable')
        else:
            raise ValueError

        self.translation_loss_fun = XentLoss(
            pad_index=self.text_tokenizer.pad_index, 
            smoothing=cfg['label_smoothing'])
        self.input_dim = self.model.config.d_model
        self.input_embed_scale = cfg.get('input_embed_scale', math.sqrt(self.model.config.d_model))

        if self.task in ['S2T', 'G2T'] and 'pretrained_model_name_or_path' in cfg:
            #in both S2T or G2T, we need gloss_tokenizer and gloss_embedding
            self.gloss_tokenizer = GlossTokenizer_G2T(tokenizer_cfg=cfg['GlossTokenizer'])
            self.gloss_embedding = self.build_gloss_embedding(**cfg['GlossEmbedding'])
            #debug
            self.gls_eos = cfg.get('gls_eos', 'gls') #gls or txt
        elif self.task == 'S2T_glsfree': 
            self.gls_eos = None
            self.gloss_tokenizer, self.gloss_embedding = None, None         
        elif 'pretrained_model_name_or_path' not in cfg:
            self.gls_eos = 'txt'
            self.gloss_tokenizer, self.gloss_embedding = None, None             
        else:
            raise ValueError

        if cfg.get('from_scratch',False)==True:
            self.model.init_weights()
            self.logger.info('Train Translation Network from scratch!')
        if cfg.get('freeze_txt_embed', False)==True:
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
                num_embeddings=len(self.gloss_tokenizer.id2gloss),
                embedding_dim=self.model.config.d_model,
                padding_idx=self.gloss_tokenizer.gloss2id['<pad>']) 
        self.logger.info('gloss2embed_file '+ gloss2embed_file)
        if from_scratch:
            self.logger.info('Train Gloss Embedding from scratch')
            assert freeze==False
        else:
            gls2embed = torch.load(gloss2embed_file)
            self.gls2embed = gls2embed
            self.logger.info('Initialize gloss embedding from {}'.format(gloss2embed_file))
            with torch.no_grad():
                for id_, gls in self.gloss_tokenizer.id2gloss.items():
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
        if self.task == 'S2T_glsfree':
            suffix_len = 0
            suffix_embedding = None
        else:
            if self.gls_eos=='gls':
                suffix_embedding = [self.gloss_embedding.weight[self.gloss_tokenizer.convert_tokens_to_ids('</s>'),:]]
            else:
                suffix_embedding = [self.model.model.shared.weight[self.text_tokenizer.eos_index,:]]
            if self.task in ['S2T', 'G2T'] and self.gloss_embedding:
                if self.gls_eos == 'gls':
                    src_lang_code_embedding = self.gloss_embedding.weight[ \
                        self.gloss_tokenizer.convert_tokens_to_ids(self.gloss_tokenizer.src_lang),:] #to-debug
                else:
                    src_lang_id = self.text_tokenizer.pruneids[30]#self.text_tokenizer.pruneids[self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)]
                    assert src_lang_id==31
                    src_lang_code_embedding = self.model.model.shared.weight[src_lang_id,:]
                suffix_embedding.append(src_lang_code_embedding)
            suffix_len = len(suffix_embedding)
            suffix_embedding = torch.stack(suffix_embedding, dim=0)

        if 'gloss+feature' in self.input_type:
            input_lengths = input_lengths+gloss_lengths #prepend gloss
        max_length = torch.max(input_lengths)+suffix_len
        inputs_embeds = []
        attention_mask = torch.zeros([input_feature.shape[0], max_length], dtype=torch.long, device=input_feature.device)

        for ii, feature in enumerate(input_feature):
            valid_len = input_lengths[ii] #(gloss included for input_type=gloss_feature)
            if 'gloss+feature' in self.input_type:
                valid_feature = torch.cat(
                    [gloss_embedding[ii, :gloss_lengths[ii],:], feature[:valid_len-gloss_lengths[ii],:]],
                    dim=0)
            else:
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
            'attention_mask': attention_mask#attention_mask
        }
        return transformer_inputs

    def forward(self,**kwargs):
        if self.input_type=='gloss':
            input_ids = kwargs.pop('input_ids')
            kwargs['inputs_embeds'] = self.prepare_gloss_inputs(input_ids)
        elif self.input_type=='feature':
            input_feature = kwargs.pop('input_feature')
            input_lengths = kwargs.pop('input_lengths')
            #quick fix
            kwargs.pop('gloss_ids', None)
            kwargs.pop('gloss_lengths', None)
            new_kwargs = self.prepare_feature_inputs(input_feature, input_lengths)
            kwargs = {**kwargs, **new_kwargs}
        elif self.input_type in ['gt_gloss+feature','pred_gloss+feature']:
            gloss_embedding = self.gloss_embedding(kwargs.pop('gloss_ids')) #do not multiply scale here!
            new_kwargs = self.prepare_feature_inputs(
                kwargs.pop('input_feature'), kwargs.pop('input_lengths'),
                gloss_embedding=gloss_embedding, gloss_lengths=kwargs.pop('gloss_lengths'))
            kwargs = {**kwargs, **new_kwargs}
        else:
            raise ValueError
        output_dict = self.model(**kwargs, return_dict=True)
        #print(output_dict.keys()) loss, logits, past_key_values, encoder_last_hidden_state
        log_prob = torch.nn.functional.log_softmax(output_dict['logits'], dim=-1)  # B, T, L
        batch_loss_sum = self.translation_loss_fun(log_probs=log_prob,targets=kwargs['labels'])
        output_dict['translation_loss'] = batch_loss_sum/log_prob.shape[0]

        output_dict['transformer_inputs'] = kwargs #for later use (decoding)
        return output_dict

    def generate(self, 
        input_ids=None, attention_mask=None, #decoder_input_ids,
        inputs_embeds=None, input_lengths=None,
        num_beams=4, max_length=100, length_penalty=1, **kwargs):
        assert attention_mask!=None
        batch_size = attention_mask.shape[0]
        decoder_input_ids = torch.ones([batch_size,1],dtype=torch.long, device=attention_mask.device)*self.text_tokenizer.sos_index
        # if self.input_type=='gloss':
        #     inputs_embeds = self.prepare_gloss_inputs(input_ids)
        # else:
        #     assert inputs_embeds!=None and attention_mask!=None
        assert inputs_embeds!=None and attention_mask!=None
        output_dict = self.model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, #same with forward 
            decoder_input_ids=decoder_input_ids,
            num_beams=num_beams, length_penalty=length_penalty, max_length=max_length, 
            return_dict_in_generate=True)
        output_dict['decoded_sequences'] = self.text_tokenizer.batch_decode(output_dict['sequences'])
        return output_dict



