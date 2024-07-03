import pickle, torch, gzip, os
from copy import deepcopy
from transformers import MBartForConditionalGeneration, MBartTokenizer
# from unidecode import unidecode


if __name__ == '__main__':
    # state_dict = torch.load('../../pretrained_models/mBart_de/pytorch_model.bin')
    # weight = state_dict['lm_head.weight']
    # torch.save(weight, '../../pretrained_models/mBart_de_t2g/text_embeddings.bin')

    # gloss_emb = torch.load('../../pretrained_models/mBart_de/gloss_embeddings.bin')
    # with open('../../pretrained_models/mBart_de/gloss2ids.pkl', 'rb') as f:
    #     gloss2ids = pickle.load(f)
    #     print(len(gloss2ids))
    # id2gloss = {v:k for k,v in gloss2ids.items()}
    # replace_weight = []
    # for id in range(len(id2gloss)):
    #     gloss = id2gloss[id]
    #     replace_weight.append(gloss_emb[gloss])
    # replace_weight = torch.stack(replace_weight, dim=0)
    # print(replace_weight.shape)

    # new_state_dict = deepcopy(state_dict)
    # new_state_dict['model.shared.weight'] = replace_weight
    # new_state_dict['model.encoder.embed_tokens.weight'] = replace_weight
    # new_state_dict['model.decoder.embed_tokens.weight'] = replace_weight
    # new_state_dict['lm_head.weight'] = replace_weight
    # new_state_dict['final_logits_bias'] = torch.zeros(1, len(gloss2ids))
    # torch.save(new_state_dict, '../../pretrained_models/mBart_de_t2g/pytorch_model.bin')


    # with open('../../data/phoenix_2014t/gloss2ids.pkl', 'rb') as f:
    #     gloss2ids_raw = pickle.load(f)
    #     id2gloss_raw = {v:k for k,v in gloss2ids_raw.items()}
    # with open('../../pretrained_models/mBart_de_t2g/gloss2ids.pkl', 'rb') as f:
    #     gloss2ids = pickle.load(f)
    #     id2gloss = {v:k for k,v in gloss2ids.items()}
    # with open('../../pretrained_models/mBart_de_t2g/map_ids.pkl', 'rb') as f:
    #     map_ids = pickle.load(f)
    # with gzip.open('../../data/phoenix_2014t/phoenix14t.train', 'rb') as f:
    #     train = pickle.load(f)
    # with gzip.open('../../data/phoenix_2014t/phoenix14t.dev', 'rb') as f:
    #     dev = pickle.load(f)
    # with gzip.open('../../data/phoenix_2014t/phoenix14t.test', 'rb') as f:
    #     test = pickle.load(f)
    #     # print(test)
    # tokenizer = MBartTokenizer.from_pretrained('../../pretrained_models/mBart_de', src_lang='de_DE')

    # print('für-alle')
    # special_char_map = {ord('ä'):'ae', ord('ü'):'ue', ord('ö'):'oe', ord('ß'):'ss'}
    # a='für-alle'
    # a.translate(special_char_map)
    
    # # gloss = list(gloss2ids.keys())
    # # st_gloss = []
    # # for g in gloss:
    # #     st_gloss.append(unidecode(g))
    
    # glosses = [id2gloss[i] for i in range(len(id2gloss))]
    # glosses = [g.translate(special_char_map) for g in glosses]

    # for item in train:
    #     for g in item['gloss'].split():
    #         if g.lower() not in glosses:
    #             print(g.lower())

    # g1 = []
    # for gloss, id in gloss2ids_raw.items():
    #     if gloss not in glosses:
    #         g1.append((gloss, id))

    # g2 = []
    # for gloss, id in gloss2ids.items():
    #     if gloss.translate(special_char_map) not in gloss2ids_raw:
    #         g2.append((gloss, id))
    # print(g1)
    # print(g2)

    # for g in gloss2ids:
    #     if unidecode(g) == 'massig':
    #         print(g)

    # with open('gloss_comp.txt', 'w') as f:
    #     f.write('idx raw mbart\n')
    #     for i in range(len(id2gloss)):
    #         f.write('{} {} {}\n'.format(i, id2gloss_raw[i], id2gloss[i]))
    #     for j in range(i+1, len(id2gloss_raw)):
    #         f.write('{} {} XXX\n'.format(j, id2gloss_raw[j]))

    
    # count = tot = 0
    # for item in test:
    #     for g in item['gloss'].split():
    #         tot += 1
    #         if g.lower() not in gloss2ids:
    #             count += 1
    #             print(g.lower())
    #             input_ids = tokenizer(g.lower())
    #             print(input_ids)
    # print(count, tot)

    # gloss_raw = set(list(gloss2ids_raw.keys()))
    # gloss = set(list(gloss2ids.keys()))
    # print(set(gloss) - set(gloss).intersection(set(gloss_raw)))
    # print(gloss2ids['andere-möglichkeit'])

    #------------------------------------------make new gloss embedding-------------------------------------
    # state_dict = torch.load('../../pretrained_models/mBart_de/pytorch_model_t2g_old.bin')
    # weight = state_dict['model.shared.weight']
    # replace_weight = []
    # for i in range(len(id2gloss_raw)):
    #     gloss_raw = id2gloss_raw[i]
    #     try:
    #         idx = glosses.index(gloss_raw)
    #         replace_weight.append(weight[idx])
    #     except:
    #         gls_mapping = {'poss-sein': 'sein', 'poss-euch': 'euch', 'sechszehn': 'sechzehn', 'haben2': 'haben', 'poss-bei-uns': 'bei-uns', 
    #                        'nocheinmal': 'noch-einmal', 'miteilen': 'mitteilen', 'erleichert': 'erleichtert', 'poss-mein': 'mein', 's0nne': 'sonne',
    #                        'vorderscheibe': 'vordere-scheibe', 'unwahrscheinlich': 'un-wahrscheinlich', 'tiefdruckzone': 'tief-druck-zone',
    #                        'rausfallen': 'raus-fallen', 'aufeinandertreffen': 'aufeinander-treffen'}
    #         assert gloss_raw in gls_mapping
    #         mapped_gls = gls_mapping[gloss_raw]
    #         print(mapped_gls)
    #         idx = glosses.index(mapped_gls)
    #         replace_weight.append(weight[idx])
    
    # replace_weight = torch.stack(replace_weight, dim=0)
    # print(replace_weight.shape)

    # new_state_dict = deepcopy(state_dict)
    # new_state_dict['model.shared.weight'] = replace_weight
    # new_state_dict['model.encoder.embed_tokens.weight'] = replace_weight
    # new_state_dict['model.decoder.embed_tokens.weight'] = replace_weight
    # new_state_dict['lm_head.weight'] = replace_weight
    # new_state_dict['final_logits_bias'] = torch.zeros(1, len(gloss2ids_raw))
    # torch.save(new_state_dict, '../../pretrained_models/mBart_de_t2g/pytorch_model.bin')

    #------------------------------------------lm_head and bias--------------------------------------------
    # state_dict = torch.load('../../pretrained_models/mBart/pytorch_model.bin', map_location='cpu')
    # full_embedding_weight = state_dict['model.shared.weight']
    # gls_mapping = {'poss-sein': 'sein', 'poss-euch': 'euch', 'sechszehn': 'sechzehn', 'haben2': 'haben', 'poss-bei-uns': 'bei-uns', 
    #                        'nocheinmal': 'noch-einmal', 'miteilen': 'mitteilen', 'erleichert': 'erleichtert', 'poss-mein': 'mein', 's0nne': 'sonne',
    #                        'vorderscheibe': 'vordere-scheibe', 'unwahrscheinlich': 'un-wahrscheinlich', 'tiefdruckzone': 'tief-druck-zone',
    #                        'rausfallen': 'raus-fallen', 'aufeinandertreffen': 'aufeinander-treffen'}
    # remain = ['poss-euch', 'haben2', 'miteilen', 's0nne']
    # replace_weight = []
    # new_map_ids = {}
    # offset = 0
    # for i in range(len(id2gloss_raw)):
    #     gloss_raw = id2gloss_raw[i]
    #     try:
    #         idx = glosses.index(gloss_raw)
    #     except:
    #         assert gloss_raw in gls_mapping
    #         mapped_gls = gls_mapping[gloss_raw]
    #         print(mapped_gls)
    #         idx = glosses.index(mapped_gls)
        
    #     text_gls = id2gloss[idx]
    #     print(gloss_raw, text_gls)
    #     if text_gls in ['<s>', '<pad>', '</s>', '<unk>', '<mask>','zh_CN','de_DE']:
    #         emb_id = tokenizer.convert_tokens_to_ids(text_gls)
    #         print('Special token {} {}'.format(emb_id, text_gls))
    #         replace_weight.append(full_embedding_weight[emb_id, :])
    #         new_map_ids[emb_id] = offset
    #         offset += 1
    #     elif text_gls in ['zh_CSL', 'de_DGS']:
    #         if text_gls == 'zh_CSL':
    #             emb_id = tokenizer.convert_tokens_to_ids('zh_CN')
    #         else:
    #             emb_id = tokenizer.convert_tokens_to_ids('de_DE')
    #         replace_weight.append(full_embedding_weight[emb_id, :])
    #         offset += 1
    #     else:
    #         input_ids = tokenizer(text_gls)['input_ids'][:-2]
    #         for id_ in input_ids:
    #             if id_ in new_map_ids:
    #                 continue
    #             new_map_ids[id_] = offset
    #             replace_weight.append(full_embedding_weight[id_, :])
    #             offset += 1
    
    # for text_gls in remain:
    #     input_ids = tokenizer(text_gls)['input_ids'][:-2]
    #     for id_ in input_ids:
    #         if id_ in new_map_ids:
    #             continue
    #         new_map_ids[id_] = offset
    #         replace_weight.append(full_embedding_weight[id_, :])
    #         offset += 1
    
    # replace_weight = torch.stack(replace_weight, dim=0).cpu()
    # print(len(new_map_ids), offset, replace_weight.shape)
    # new_state_dict = deepcopy(state_dict)
    # new_state_dict['model.shared.weight'] = replace_weight
    # new_state_dict['model.encoder.embed_tokens.weight'] = replace_weight
    # new_state_dict['model.decoder.embed_tokens.weight'] = replace_weight
    # new_state_dict['lm_head.weight'] = replace_weight
    # new_state_dict['final_logits_bias'] = torch.zeros(1, replace_weight.shape[0])
    # torch.save(new_state_dict, '../../pretrained_models/mBart_de_t2g/pytorch_model.bin')
    
    # gls_mapping_clean = {id2gloss[i]: glosses[i] for i in range(len(gloss2ids))}
    # inv_gls_mapping = {v:k for k,v in gls_mapping.items() if k not in remain}
    # for gls in inv_gls_mapping:
    #     gls_mapping_clean[gls] = inv_gls_mapping[gls]
    # for r in remain:
    #     gls_mapping_clean[r] = r
    # # print(gls_mapping_clean)
    # # with open('../../pretrained_models/mBart_de_t2g/gloss_mapping.pkl', 'wb') as f:
    # #     pickle.dump(gls_mapping_clean, f)
    # # with open('../../pretrained_models/mBart_de_t2g/gloss_map_ids.pkl', 'wb') as f:
    # #     pickle.dump(new_map_ids, f)

    # for item in dev:
    #     for g in item['gloss'].split():
    #         if g.lower() not in gls_mapping_clean.values():
    #             print(g)


    # gls2emb = {}
    # #special tokens!
    # #</s><lang><unk><mask>
    # for t in ['<s>', '<pad>', '</s>', '<unk>','<mask>','zh_CN','de_DE']:
    #     emb_id = tokenizer.convert_tokens_to_ids(t)
    #     print('Special token {} {}'.format(emb_id, t))
    #     gls2emb[t] = full_embedding_weight[emb_id, :]
    # gls2emb['zh_CSL'] = gls2emb['zh_CN']
    # gls2emb['de_DGS'] = gls2emb['de_DE']

    # #gls
    # for gls in glosses:
    #     gls = gls.lower()
    #     gls_ids = tokenizer(gls)['input_ids'][:-2] # remove</s> <lang>
    #     emb = []
    #     for i in gls_ids:
    #         emb.append(full_embedding_weight[i,:])
    #     gls2emb[gls] = emb
    # print(len(glosses), len(gls2emb))
    
    #---------------------------------------------------------------CSL-Daily-------------------------------------------------------
    # state_dict = torch.load('../../pretrained_models/mBart_zh/pytorch_model.bin')
    # weight = state_dict['lm_head.weight']
    # torch.save(weight, '../../pretrained_models/mBart_zh_t2g/text_embeddings.bin')

    # gloss_emb = torch.load('../../pretrained_models/mBart_zh/gloss_embeddings.bin')
    # with open('../../pretrained_models/mBart_zh/gloss2ids.pkl', 'rb') as f:
    #     gloss2ids = pickle.load(f)
    #     print(len(gloss2ids))
    # id2gloss = {v:k for k,v in gloss2ids.items()}
    # replace_weight = []
    # for id in range(len(id2gloss)):
    #     gloss = id2gloss[id]
    #     replace_weight.append(gloss_emb[gloss])
    # replace_weight = torch.stack(replace_weight, dim=0)
    # print(replace_weight.shape)

    # new_state_dict = deepcopy(state_dict)
    # new_state_dict['model.shared.weight'] = replace_weight
    # new_state_dict['model.encoder.embed_tokens.weight'] = replace_weight
    # new_state_dict['model.decoder.embed_tokens.weight'] = replace_weight
    # new_state_dict['lm_head.weight'] = replace_weight
    # new_state_dict['final_logits_bias'] = torch.zeros(1, len(gloss2ids))
    # torch.save(new_state_dict, '../../pretrained_models/mBart_zh_t2g/pytorch_model.bin')

    #----------------------------------------------------CSL-Daily update gloss2ids------------------------------------------
    # gls_emb = torch.load('../../pretrained_models/mBart_zh_t2g/gloss_embeddings.bin')
    # gloss2ids = {}
    # idx = 0
    # replace_weight = []
    # for k,v in gls_emb.items():
    #     gloss2ids[k] = idx
    #     replace_weight.append(v)
    #     idx += 1
    # replace_weight = torch.stack(replace_weight, dim=0)
    # print(replace_weight.shape)
    # print(len(gloss2ids))
    # with open('../../pretrained_models/mBart_zh_t2g/gloss2ids.pkl', 'wb') as f:
    #     pickle.dump(gloss2ids, f)

    # state_dict = torch.load('../../pretrained_models/mBart_zh_t2g/pytorch_model.bin')
    # new_state_dict = deepcopy(state_dict)
    # new_state_dict['model.shared.weight'] = replace_weight
    # new_state_dict['model.encoder.embed_tokens.weight'] = replace_weight
    # new_state_dict['model.decoder.embed_tokens.weight'] = replace_weight
    # new_state_dict['lm_head.weight'] = replace_weight
    # new_state_dict['final_logits_bias'] = torch.zeros(1, len(gloss2ids))
    # torch.save(new_state_dict, '../../pretrained_models/mBart_zh_t2g/pytorch_model.bin')


    #---------------------------------------------------------------TVB-------------------------------------------------------
    state_dict = torch.load('../../pretrained_models/mBart_tvb/pytorch_model.bin')
    weight = state_dict['lm_head.weight']
    torch.save(weight, '../../pretrained_models/mBart_tvb_t2g/text_embeddings.bin')

    gloss_emb = torch.load('../../pretrained_models/mBart_tvb/gloss_embeddings.bin')
    with open('../../pretrained_models/mBart_tvb/gloss2ids.pkl', 'rb') as f:
        gloss2ids = pickle.load(f)
        print(len(gloss2ids))
    id2gloss = {v:k for k,v in gloss2ids.items()}
    replace_weight = []
    for id in range(len(id2gloss)):
        gloss = id2gloss[id]
        replace_weight.append(gloss_emb[gloss])
    replace_weight = torch.stack(replace_weight, dim=0)
    print(replace_weight.shape)

    new_state_dict = deepcopy(state_dict)
    new_state_dict['model.shared.weight'] = replace_weight
    new_state_dict['model.encoder.embed_tokens.weight'] = replace_weight
    new_state_dict['model.decoder.embed_tokens.weight'] = replace_weight
    new_state_dict['lm_head.weight'] = replace_weight
    new_state_dict['final_logits_bias'] = torch.zeros(1, len(gloss2ids))
    torch.save(new_state_dict, '../../pretrained_models/mBart_tvb_t2g/pytorch_model.bin')

