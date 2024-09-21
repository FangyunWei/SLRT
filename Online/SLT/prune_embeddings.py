import torch, os, gzip, pickle, json, numpy as np
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
import transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer


tokenizer = MBartTokenizer.from_pretrained('../../pretrained_models/mBart')
tokenizer.src_lang = 'zh_CN'
model = MBartForConditionalGeneration.from_pretrained('../../pretrained_models/mBart')
print('Vocab size = ', tokenizer.vocab_size)
full_embedding_weight = model.model.shared.weight
full_final_logits_bias = model.final_logits_bias
print(full_embedding_weight.shape, full_final_logits_bias.shape)
with open(os.path.join('../../pretrained_models/mBart/config.json'),'r') as f:
    config_json = json.load(f)

new_embedding_weight_list = []
new_final_logits_bias_list = []
map_ids = {}
with open(os.path.join('../../pretrained_models/mBart/tokenizer.json'),'r') as f:
    tokenizer_json = json.load(f)
print('Special tokens #', len(tokenizer_json['added_tokens']))
for new_id, added_token in enumerate(tokenizer_json['added_tokens']):
    id_ = added_token['id']
    new_embedding_weight_list.append(full_embedding_weight[id_,:])
    new_final_logits_bias_list.append(full_final_logits_bias[:,id_])
    map_ids[id_] = new_id
    print('{} {}->{}'.format(added_token['content'], id_, new_id))


def gather_vocab(filename_format, tokenizer):
    text_ids = defaultdict(int)
    glosses = defaultdict(int)
    for split in ['train','dev','test']:
        with open(filename_format.format(split),'rb') as f:
            data = pickle.load(f)
        for d in data:
            input_ids = tokenizer(d['text'])['input_ids'][:-2]
            # print(tokenizer(d['text'])['input_ids'][-1])
            for id_ in input_ids:
                text_ids[id_] += 1
            for gls in d['gloss'].lower().split():
                input_ids = tokenizer(gls)['input_ids'][:-2]
                for id_ in input_ids:
                    text_ids[id_] += 1
                    glosses[gls] += 1
    print(os.path.dirname(filename_format), '#subunits=',len(text_ids), ' #gloss=',len(glosses))
    return dict(text_ids), dict(glosses)
text2fre_zh, gloss2fre_zh = gather_vocab('../../data/tvb/v5.6_{}_sim.pkl', tokenizer)


def add_subunit(subunits, embedding_list, logits_list, map_ids):
    offset = len(map_ids)
    assert len(map_ids) == len(embedding_list)
    print('Length of embedding list ', len(embedding_list),end='->')
    for ii, sid in enumerate(subunits):
        if sid in map_ids:
            print(sid, 'already exists in embedding (a special token)')
            continue
        map_ids[sid] = len(embedding_list) #ii + offset 
        embedding_list.append(full_embedding_weight[sid,:])
        logits_list.append(full_final_logits_bias[:,sid])
    print(len(embedding_list))
    assert len(map_ids)==len(embedding_list), (len(map_ids),len(embedding_list))
    return embedding_list, logits_list, map_ids
new_embedding_weight_list_zh, new_final_logits_bias_list_zh, map_ids_zh = add_subunit(
            text2fre_zh, 
            new_embedding_weight_list[:], 
            new_final_logits_bias_list[:], deepcopy(map_ids))


def save_new_model(src_dir, tgt_dir, new_logits_list, new_embeddings_list, map_ids):
    os.makedirs(tgt_dir, exist_ok=True)
    #1. cp tokenizer, sentencepiece
    os.system('cp {} {}'.format(os.path.join(src_dir,'sentencepiece*'), tgt_dir))
    os.system('cp {} {}'.format(os.path.join(src_dir,'tokenizer.json'), tgt_dir))
    #2. model_state_dict
    new_state_dict = deepcopy(model.state_dict())
    new_state_dict['final_logits_bias'] = torch.cat(new_logits_list, dim=0).unsqueeze(0)
    print('final_logits_bias shape=', new_state_dict['final_logits_bias'].shape)
    new_state_dict['model.shared.weight'] = torch.stack(new_embeddings_list, dim=0)
    print('new_embeddings shape=', new_state_dict['model.shared.weight'].shape)
    new_state_dict['model.encoder.embed_tokens.weight'] = new_state_dict['model.shared.weight'] #model.encoder.embed_tokens.weight
    new_state_dict['model.decoder.embed_tokens.weight'] = new_state_dict['model.shared.weight']
    new_state_dict['lm_head.weight'] = new_state_dict['model.shared.weight']
    torch.save(new_state_dict, os.path.join(tgt_dir, 'pytorch_model.bin'))
    #3. config
    new_config_json = deepcopy(config_json)
    new_config_json['vocab_size'] = new_state_dict['model.shared.weight'].shape[0]
    print('new vocab size=', new_config_json['vocab_size'])
    with open(os.path.join(tgt_dir,'config.json'),'w') as f:
        json.dump(new_config_json, f)
    #4.map_ids:
    assert len(map_ids) == new_config_json['vocab_size']
    with open(os.path.join(tgt_dir,'map_ids.pkl'),'wb') as f:
        pickle.dump(map_ids, f)
save_new_model(
    '../../pretrained_models/mBart', '../../pretrained_models/mBart_tvb',
    new_logits_list=new_final_logits_bias_list_zh, 
    new_embeddings_list=new_embedding_weight_list_zh, 
    map_ids=map_ids_zh)


def create_gloss_embedding(glosses):
    gls2emb = {}
    #special tokens!
    #</s><lang><unk><mask>
    for t in ['<s>', '<pad>', '</s>', '<unk>','<mask>','zh_CN','de_DE']:
        emb_id = tokenizer.convert_tokens_to_ids(t)
        print('Special token {} {}'.format(emb_id, t))
        gls2emb[t] = full_embedding_weight[emb_id,:]
    gls2emb['zh_CSL'] = gls2emb['zh_CN']
    gls2emb['de_DGS'] = gls2emb['de_DE']
    #gls
    for gls in glosses:
        gls = gls.lower()
        gls_ids = tokenizer(gls)['input_ids'][:-2] # remove</s> <lang>
        emb = []
        for i in gls_ids:
            emb.append(full_embedding_weight[i,:])
        emb = torch.mean(torch.stack(emb, dim=0), dim=0)
        gls2emb[gls] = emb
    print(len(glosses), len(gls2emb))
    return gls2emb
gls2emb_zh =  create_gloss_embedding(gloss2fre_zh)
torch.save(gls2emb_zh, os.path.join('../../pretrained_models/mBart_tvb/gloss_embeddings.bin'))


def save_gloss_index(gls2emb, output_dir):
    gls2id = {}
    for id_, gls in enumerate(gls2emb):
        gls2id[gls] = id_
    with open(os.path.join(output_dir,'gloss2ids.pkl'),'wb') as f:
        pickle.dump(gls2id, f)
    # print(len(gls2id))
    # print(gls2id)
save_gloss_index(gls2emb_zh, '../../pretrained_models/mBart_tvb')


# tokenizer = MBartTokenizer.from_pretrained('../../pretrained_models/mBart_tvb')
# print(tokenizer('这层'))