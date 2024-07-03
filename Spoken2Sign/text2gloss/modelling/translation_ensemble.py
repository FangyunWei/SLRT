import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.file_utils import ModelOutput
from utils.misc import freeze_params, get_logger
from utils.loss import XentLoss
from .Tokenizer import GlossTokenizer_G2T, TextTokenizer
import pickle, math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

def load_from_pretrained_ckpt(model, pretrained_ckpt):
    checkpoint = torch.load(pretrained_ckpt, map_location='cpu')['model_state']
    load_dict = {}
    for k,v in checkpoint.items():
        if 'translation_network' in k and not 'gloss_embedding' in k:
            load_dict[k.replace('translation_network.model.','')] = v
    model.load_state_dict(load_dict)
    return

class MBart_Ensemble(PreTrainedModel):
    def __init__(self, cfg):
        config = MBartConfig.from_pretrained(cfg['pretrained_model_name_or_path'])
        super().__init__(config)       
        self.logger = get_logger()
        self.model_list  = []
        for ii,ckpt_path in enumerate(cfg['load_ckpts']):
            model = MBartForConditionalGeneration.from_pretrained(
                cfg['pretrained_model_name_or_path'],
                **cfg.get('overwrite_cfg', {}))
            load_from_pretrained_ckpt(model, ckpt_path)
            self.logger.info(f'Load Translation Network {ii} from {ckpt_path}')
            self.model_list.append(model)
        self.model_list = torch.nn.ModuleList(self.model_list)
    

    # #to overwrite (you may need to import some module from mBART)
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs #inputs_embeds
    ) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs: #actually a list
            encoder_outputs_list = []
            for model_id, model in enumerate(self.model_list):
                # retrieve encoder hidden states
                encoder = model.get_encoder()
                encoder_kwargs = {
                    argument: value
                    for argument, value in model_kwargs.items()
                    if (not (argument.startswith("decoder_") or argument.startswith("cross_attn")) and '_list' not in argument)
                }
                #inputs_embeds
                #attention_mask
                encoder_kwargs['inputs_embeds'] = model_kwargs['inputs_embeds_list'][model_id]
                encoder_kwargs['attention_mask'] = model_kwargs['attention_mask_list'][model_id]
                encoder_outputs: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)
                encoder_outputs_list.append(encoder_outputs)
            model_kwargs['encoder_outputs_list'] = encoder_outputs_list
            model_kwargs['encoder_outputs'] = encoder_outputs_list[0]
        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            raise ValueError
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            for model_id, am in enumerate(model_kwargs["attention_mask_list"]):
                model_kwargs["attention_mask_list"][model_id]=am.index_select(0, expanded_return_idx)
            model_kwargs['attention_mask'] = model_kwargs["attention_mask_list"][0]
        else:
            raise ValueError

        if is_encoder_decoder:
            assert encoder_outputs is not None
            model_kwargs["encoder_outputs"] = []
            for model_id, eo in enumerate(model_kwargs["encoder_outputs_list"]):
                model_kwargs["encoder_outputs_list"][model_id]["last_hidden_state"] = eo.last_hidden_state.index_select(
                            0, expanded_return_idx.to(eo.last_hidden_state.device)
                        )
            model_kwargs["encoder_outputs"] = model_kwargs["encoder_outputs_list"][0]
        else:
            raise ValueError
        return input_ids, model_kwargs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        inputs_embeds_list,
        encoder_outputs_list, 
        attention_mask_list,
        past=None,   
        use_cache=None,
        # past=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        **kwargs
    ):  
        # cut decoder_input_ids if past is used
        if past is not None: #past=None (1st forwarding) past=tuple (afterwards)
            assert type(past)==list
            decoder_input_ids = decoder_input_ids[:, -1:]
        model_kwargs_list = []
        for model_id in range(len(inputs_embeds_list)):
            model_kwargs_list.append({
                "input_ids":None, #neither input_ids nor inputs_embeds is required here (we have encoder_outputs)
                "encoder_outputs": encoder_outputs_list[model_id],
                "past_key_values": past[model_id] if past!=None else None,
                "decoder_input_ids": decoder_input_ids,
                "attention_mask": attention_mask_list[model_id],
                "head_mask": head_mask,
                "decoder_head_mask": decoder_head_mask,
                "cross_attn_head_mask": cross_attn_head_mask,
                "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)                
            })
        return {'model_kwargs_list': model_kwargs_list}

    def forward(self, model_kwargs_list, **other_kwargs):
        #print('we arrive here!')
        #print(model_kwargs_list[0]['encoder_outputs'])
        #please check the model inputs before going on 
        outputs_list = []
        for model_id, model_kwargs in enumerate(model_kwargs_list):
            # print(model_kwargs['attention_mask'].shape)
            # print(model_kwargs['decoder_input_ids'].shape)
            # print(model_kwargs['encoder_outputs'].last_hidden_state.shape)
            # input()
            outputs = self.model_list[model_id](**model_kwargs, **other_kwargs)
            #logits', 'past_key_values', 'encoder_last_hidden_state
            outputs_list.append(outputs)
        #outputs.logits
        #outputs.logits = sum([o.logits for o in outputs_list]) #plus][]
        outputs.logits = sum([o.logits.softmax(dim=-1) for o in outputs_list]).log() #plus][]
        outputs['past_key_values_list'] = [o.past_key_values for o in outputs_list]
        outputs['encoder_last_hidden_state_list'] = [o.encoder_last_hidden_state for o in outputs_list]
        return outputs

    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past
        if "past_key_values_list" in outputs:
            model_kwargs["past"] = outputs.past_key_values_list
        else:
            raise ValueError
        # elif "mems" in outputs:
        #     model_kwargs["past"] = outputs.mems
        # elif "past_buckets_states" in outputs:
        #     model_kwargs["past"] = outputs.past_buckets_states
        # else:
        #     model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            raise ValueError
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            raise ValueError
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return model_kwargs

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past_list = []
        for model_id, past_ in enumerate(past):
            reordered_past = ()
            for layer_past in past_:
                # cached cross_attention states don't have to be reordered -> they are always the same
                reordered_past += (
                    tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
                )
            reordered_past_list.append(reordered_past)
        return reordered_past_list

class TranslationNetwork_Ensemble(PreTrainedModel):
    def __init__(self, cfg) -> None:
        config = MBartConfig.from_pretrained(cfg['pretrained_model_name_or_path'])
        super().__init__(config)
        self.logger = get_logger()   
        self.text_tokenizer = TextTokenizer(tokenizer_cfg=cfg['TextTokenizer'])
        self.logger.info('Initialize translation network from {}'.format(cfg['pretrained_model_name_or_path']))
        self.model = MBart_Ensemble(cfg)
        self.translation_loss_fun = XentLoss(
            pad_index=self.text_tokenizer.pad_index, 
            smoothing=cfg['label_smoothing'])

    def forward(self, inputs_embeds_list, attention_mask_list, **kwargs):
        output_dict = {}
        for model_id, model in enumerate(self.model.model_list):
            output_dict_ = model(
                inputs_embeds=inputs_embeds_list[model_id].to(kwargs['labels'].device),
                attention_mask=attention_mask_list[model_id].to(kwargs['labels'].device),
                **kwargs,
                return_dict=True
            )
            log_prob = torch.nn.functional.log_softmax(output_dict_['logits'], dim=-1)  # B, T, L
            batch_loss_sum = self.translation_loss_fun(log_probs=log_prob,targets=kwargs['labels'])
            output_dict[f'model_{model_id}_translation_loss'] = batch_loss_sum/log_prob.shape[0]
        output_dict['transformer_inputs'] = {
            'inputs_embeds_list':inputs_embeds_list,
            'attention_mask_list':attention_mask_list,
            **kwargs}
        return output_dict
    
    def generate(self, 
        inputs_embeds_list, attention_mask_list,
        num_beams, length_penalty, max_length, **kwargs):
        batch_size = attention_mask_list[0].shape[0]
        decoder_input_ids = torch.ones([batch_size,1],dtype=torch.long, device=attention_mask_list[0].device)*self.text_tokenizer.sos_index
        output_dict = self.model.generate(
            inputs_embeds=inputs_embeds_list[0], #place holder
            attention_mask=attention_mask_list[0], #place holder
            decoder_input_ids=decoder_input_ids,
            inputs_embeds_list=inputs_embeds_list,
            attention_mask_list=attention_mask_list,
            num_beams=num_beams, length_penalty=length_penalty, max_length=max_length, 
            return_dict_in_generate=True)
        output_dict['decoded_sequences'] = self.text_tokenizer.batch_decode(output_dict['sequences'])  
        return output_dict
    
