from turtle import forward
import warnings, wandb
import pickle
from collections import defaultdict
from modelling.model import build_model
from utils.optimizer import build_optimizer, build_scheduler
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import os, sys
import shutil
import time
import queue
sys.path.append(os.getcwd())#slt dir
import torch
from torch.nn.parallel import DistributedDataParallel as DDP, distributed
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from utils.misc import (
    get_logger,
    load_config,
    log_cfg,
    load_checkpoint,
    make_logger, make_writer,
    set_seed,
    symlink_update,
    is_main_process, init_DDP, move_to_device,
    neq_load_customized,
    synchronize,
)
from utils.metrics import compute_accuracy, wer_list
from dataset.Dataloader import build_dataloader
from dataset.Dataset import build_dataset
from utils.progressbar import ProgressBar
from copy import deepcopy
from ctcdecode import CTCBeamDecoder
from itertools import groupby
from utils.phoenix_cleanup import clean_phoenix_2014, clean_phoenix_2014_trans


def get_entropy(p):
    logp = p.log()
    return -(p*logp).sum(dim=-1)

def map_phoenix_gls(g_lower):#lower->upper
    if 'neg-' in g_lower[:4]:
        g_upper = 'neg-'+g_lower[4:].upper()
    elif 'poss-' in g_lower:
        g_upper = 'poss-'+g_lower[5:].upper()
    elif 'negalp-' in g_lower:
        g_upper = 'negalp-'+g_lower[7:].upper()
    else:
        g_upper = g_lower.upper()
    return g_upper

def index2token(index, vocab):
    token = [map_phoenix_gls(vocab[i]) for i in index]
    return token


def evaluation(model, val_dataloader, cfg, 
        tb_writer=None, wandb_run=None,
        epoch=None, global_step=None,
        generate_cfg={}, save_dir=None, return_prob=False, return_others=False):  #to-do output_dir
    logger = get_logger()
    logger.info(generate_cfg)
    print()
    vocab = val_dataloader.dataset.vocab
    split = val_dataloader.dataset.split
    cls_num = len(vocab)

    word_emb_tab = []
    if val_dataloader.dataset.word_emb_tab is not None:
        for w in vocab:
            word_emb_tab.append(torch.from_numpy(val_dataloader.dataset.word_emb_tab[w]))
        word_emb_tab = torch.stack(word_emb_tab, dim=0).float().to(cfg['device'])
    else:
        word_emb_tab = None

    if is_main_process() and os.environ.get('enable_pbar', '1') == '1':
        pbar = ProgressBar(n_total=len(val_dataloader), desc=val_dataloader.dataset.split.upper())
    else:
        pbar = None
    if epoch != None:
        logger.info('------------------Evaluation epoch={} {} examples #={}---------------------'.format(epoch, val_dataloader.dataset.split, len(val_dataloader.dataset)))
    elif global_step != None:
        logger.info('------------------Evaluation global step={} {} examples #={}------------------'.format(global_step, val_dataloader.dataset.split, len(val_dataloader.dataset)))
    
    model.eval()
    val_stat = defaultdict(float)
    results = defaultdict(dict)
    name_prob = {}
    contras_setting = cfg['model']['RecognitionNetwork']['visual_head']['contras_setting']
    if contras_setting is not None and 'only' in contras_setting:
        pred_src = 'word_emb_att_scores'
        if 'l1' in contras_setting or 'l2' in contras_setting:
            pred_src = 'fea_vect'
        if 'margin' in contras_setting:
            pred_src = 'gloss_logits'
    else:
        pred_src = 'gloss_logits'
    if cfg['model']['RecognitionNetwork']['visual_head']['variant'] in ['arcface', 'cosface']:
        pred_src = 'gloss_raw_logits'

    dataset_name = cfg['data']['dataset_name']
    if dataset_name in ['phoenix']:
        cls_num = len(vocab)
        if '<blank>' in vocab:
            ctc_decoder_vocab = [chr(x) for x in range(20000, 20000+cls_num)]
            blank_id = vocab.index('<blank>')
        else:
            ctc_decoder_vocab = [chr(x) for x in range(20000, 20000+cls_num+1)]
            blank_id = cls_num
            vocab.append('<blank>')
        ctc_decoder = CTCBeamDecoder(ctc_decoder_vocab,
                                    beam_width=cfg['testing']['cfg']['recognition']['beam_size'],
                                    blank_id=blank_id,
                                    num_processes=5,
                                    log_probs_input=False
                                    )
    
    with torch.no_grad():
        logits_name_lst = []
        for step, batch in enumerate(val_dataloader):
            #forward -- loss
            batch = move_to_device(batch, cfg['device'])

            forward_output = model(is_train=False, labels=batch['labels'], sgn_videos=batch['sgn_videos'], sgn_keypoints=batch['sgn_keypoints'], epoch=epoch)

            if dataset_name in ['phoenix']:
                name = batch['names'][0]
                gls_ref = batch['gls_ref'][0]
                results[name]['gls_ref'] = gls_ref
                rgb_gls_logits = forward_output['rgb_temp_gloss_logits']
                keypoint_gls_logits = forward_output['keypoint_temp_gloss_logits']
                fuse_gls_logits = forward_output['fuse_temp_gloss_logits']
                final_gls_prob = (rgb_gls_logits.softmax(-1)+keypoint_gls_logits.softmax(-1)+fuse_gls_logits.softmax(-1)) / 3
                print(final_gls_prob.shape)

                T = final_gls_prob.shape[1]
                len_video = torch.LongTensor([T]).to(final_gls_prob.device)
                
                pred_seq, beam_scores, _, out_seq_len = ctc_decoder.decode(final_gls_prob, len_video)
                hyp = [x[0] for x in groupby(pred_seq[0][0][:out_seq_len[0][0]].tolist())]
                gls_hyp = index2token(hyp, vocab)
                # print(hyp)
                # print(gls_hyp)
                results[name]['beam_search_gls_hyp'] = ' '.join(gls_hyp)
                continue

            if is_main_process():
                for k,v in forward_output.items():
                    if '_loss' in k:
                        val_stat[k] += v.item()
                    # elif '_weight' in k:
                    #     val_stat[k] += v.mean().item()

            #rgb/keypoint/fuse/ensemble_last_logits
            for k, gls_logits in forward_output.items():
                if pred_src not in k or gls_logits == None:
                    continue
                
                logits_name = k.replace(pred_src,'')
                if 'word_fused' in logits_name or 'xmodal_fused' in logits_name:
                    continue
                if logits_name not in logits_name_lst:
                    logits_name_lst.append(logits_name)

                # if pred_src == 'word_emb_att_scores':
                #     gls_logits = gls_logits.softmax(dim=-1).mean(dim=1) if gls_logits.ndim > 2 else gls_logits.softmax(dim=-1)

                decode_output = model.predict_gloss_from_logits(gloss_logits=gls_logits, k=10)
                if return_prob and (contras_setting is None or 'only' not in contras_setting):
                    gls_prob = forward_output[f'{logits_name}gloss_logits'].softmax(dim=-1)
                if return_prob:
                    if (len(cfg['data']['input_streams']) == 1 and logits_name == '') or \
                        (len(cfg['data']['input_streams']) > 1 and logits_name in ['ensemble_last_', 'fuse_', 'ensemble_all_']):
                        for i in range(gls_prob.shape[0]):
                            name = logits_name + batch['names'][i]
                            name_prob[name] = gls_prob[i]

                if (contras_setting is None or 'only' not in contras_setting) and return_prob and \
                    (contras_setting is not None and ('dual' not in contras_setting or 'word_fused' not in logits_name)):
                    gls_prob = torch.sort(gls_prob, dim=-1, descending=True)[0]
                    gls_prob = gls_prob[..., :10]  #[B,10]
                if contras_setting is not None and 'dual' in contras_setting and 'word_fused' in logits_name:
                    if len(cfg['data']['input_streams']) == 1:
                        gls_prob = forward_output['word_fused_gloss_probabilities']
                        if 'multi_label' not in contras_setting:
                            B,K,N = gls_prob.shape
                            topk_idx = forward_output['topk_idx'].view(B,-1)

                for i in range(decode_output.shape[0]):
                    name = batch['names'][i]
                    if contras_setting is not None and 'dual' in contras_setting and 'word_fused' in logits_name:
                        if len(cfg['data']['input_streams']) == 1:
                            if 'multi_label' not in contras_setting:
                                topk = topk_idx[i]  #[K]
                                word_cond_prob = gls_prob[i]  #[K,N]
                                hyp, prior, cond, cond_overall = [], [], [], []
                                for j in range(K):
                                    cond.append(word_cond_prob[j, topk[j]].item())
                                    cond_overall.append(word_cond_prob[j, topk])
                                    prior.append(forward_output['gloss_probabilities'][i, topk[j]].item())
                                prior, cond = torch.tensor(prior), torch.tensor(cond)
                                prior, cond = F.normalize(prior, p=1.0, dim=0), F.normalize(cond, p=1.0, dim=0)
                                idx = torch.argsort(cond, dim=-1, descending=True)
                                results[name]['word_fused_hyp'] = [topk[d.item()].item() for d in idx]

                                cond_overall = torch.stack(cond_overall, dim=0).to(prior.device)
                                cond_overall = F.normalize(cond_overall, p=1.0, dim=1)
                                margin_w = torch.matmul(prior, cond_overall)
                                idx = torch.argsort(margin_w, dim=-1, descending=True)
                                results[name]['margin_hyp'] = [topk[d.item()].item() for d in idx]
                                logits_name_lst.append('margin_')

                                prior_u = (1.0/K) * torch.ones(K)
                                margin_u = torch.matmul(prior_u, cond_overall)
                                idx = torch.argsort(margin_u, dim=-1, descending=True)
                                results[name]['margin_uniform_hyp'] = [topk[d.item()].item() for d in idx]
                                logits_name_lst.append('margin_uniform_')

                                idx = torch.argsort(prior+margin_w, dim=-1, descending=True)
                                results[name]['joint_margin_hyp'] = [topk[d.item()].item() for d in idx]
                                logits_name_lst.append('joint_margin_')

                                idx = torch.argsort(prior+margin_u, dim=-1, descending=True)
                                results[name]['joint_margin_uniform_hyp'] = [topk[d.item()].item() for d in idx]
                                logits_name_lst.append('joint_margin_uniform_')

                                # hyp = [1]
                                # results[name][f'{logits_name}hyp'] = hyp
                            else:
                                idx = torch.argsort(gls_prob, dim=-1, descending=True)[i,:10]
                                results[name]['word_fused_hyp'] = [d.item() for d in idx]

                        else:
                            hyp = [1]
                            results[name][f'{logits_name}hyp'] = hyp

                    else:
                        hyp = [d.item() for d in decode_output[i]]
                        results[name][f'{logits_name}hyp'] = hyp

                    if (contras_setting is None or 'only' not in contras_setting) and return_prob and \
                        (contras_setting is not None and ('dual' not in contras_setting or 'word_fused' not in logits_name)):
                        prob = [d.item() for d in gls_prob[i]]
                        results[name][f'{logits_name}prob'] = prob

                    ref = batch['labels'][i].item()
                    results[name]['ref'] = ref

                    if contras_setting is not None and 'contras' in contras_setting and 'ensemble' not in logits_name and \
                        'word_fused' not in logits_name and forward_output[f'{logits_name}word_emb_att_scores'] is not None:
                        s = forward_output[f'{logits_name}word_emb_att_scores']
                        word_emb_att_scores = s.softmax(dim=-1).mean(dim=1)[i] if s.ndim>2 else s.softmax(dim=-1)[i]
                        top_scores = word_emb_att_scores[decode_output[i]]
                        max_idx = torch.argmax(word_emb_att_scores)
                        max_score = torch.amax(word_emb_att_scores)
                        top_scores = top_scores.tolist()
                        top_scores.extend([max_idx.item(), max_score.item()])
                        # print(top_scores)
                        results[name][f'{logits_name}word_emb_att_scores'] = top_scores

            if pbar:
                pbar(step)
        print()
    
    if dataset_name in ['phoenix']:
        gls_hyp = []
        gls_ref = []
        clean_func = clean_phoenix_2014_trans
        for name, res in results.items():
            gls_hyp.append(clean_func(res['beam_search_gls_hyp']))
            gls_ref.append(clean_func(res['gls_ref']))
            # print(res['gls_ref'], clean_func(res['gls_ref']))
        # print(gls_ref, res['gls_ref'])
        wer_results = wer_list(references=gls_ref, hypotheses=gls_hyp)
        logger.info('WER: {:.2f}, DEL: {:.2f}, INS: {:.2f}, SUB: {:.2f}'\
                    .format(wer_results['wer'], wer_results['del'], wer_results['ins'], wer_results['sub']))
        return None, None, None, None, None

    #logging and tb_writer
    if is_main_process():
        for k, v in val_stat.items():
            if '_loss' in k:
                logger.info('{} Average:{:.2f}'.format(k, v/len(val_dataloader)))
            if wandb_run:
                wandb.log({f'eval/{k}': v/len(val_dataloader)})
    
    print('compute acc...')
    per_ins_stat_dict, per_cls_stat_dict = compute_accuracy(results, logits_name_lst, cls_num, cfg['device'])
    others = {}
    if return_others:
        #compute accuracy for other variants
        if cfg['data']['dataset_name'] == 'WLASL_2000':
            other_vocab = [1000, 300, 100]
        elif cfg['data']['dataset_name'] == 'MSASL_1000':
            other_vocab = [500, 200, 100]
        else:
            other_vocab = []
        for o in other_vocab:
            others[str(o)] = {}
            name_lst = val_dataloader.dataset.other_vars[str(o)][split]
            effective_label_idx = val_dataloader.dataset.other_vars[str(o)]['vocab_idx']
            temp_ins_stat_dict, temp_cls_stat_dict = compute_accuracy(results, logits_name_lst, cls_num, cfg['device'], name_lst,
                                                        effective_label_idx)
            others[str(o)]['per_ins_stat'] = deepcopy(temp_ins_stat_dict)
            others[str(o)]['per_cls_stat'] = deepcopy(temp_cls_stat_dict)

    #save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'results_{}.pkl'.format(cfg['rank'])), 'wb') as f:
            pickle.dump(results, f)
        with open(os.path.join(save_dir, 'per_cls_stat_dict_{}.pkl'.format(cfg['rank'])), 'wb') as f:
            pickle.dump(per_cls_stat_dict, f)

    if return_prob:
        with open(os.path.join(save_dir, 'name_prob.pkl'), 'wb') as f:
            pickle.dump(name_prob, f)
    logger.info('-------------------------Evaluation Finished-------------------------'.format(global_step, len(val_dataloader.dataset)))
    return per_ins_stat_dict, per_cls_stat_dict, results, name_prob, others


def sync_results(per_ins_stat_dict, per_cls_stat_dict, save_dir=None, wandb_run=None, sync=True):
    logger = get_logger()

    if sync:
        for d in [per_ins_stat_dict, per_cls_stat_dict]:
            for k,v in d.items():
                synchronize()
                torch.distributed.all_reduce(v)

    evaluation_results = {}
    if is_main_process():
        for k, per_ins_stat in per_ins_stat_dict.items():
            correct, correct_5, correct_10, num_samples = per_ins_stat
            logger.info('#samples: {}'.format(num_samples))
            evaluation_results[f'{k}per_ins_top_1'] = (correct / num_samples).item()
            logger.info('-------------------------{}Per-instance ACC Top-1: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_ins_top_1']))
            evaluation_results[f'{k}per_ins_top_5'] = (correct_5 / num_samples).item()
            logger.info('-------------------------{}Per-instance ACC Top-5: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_ins_top_5']))
            evaluation_results[f'{k}per_ins_top_10'] = (correct_10 / num_samples).item()
            logger.info('-------------------------{}Per-instance ACC Top-10: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_ins_top_10']))

        # one class missing in the test set of WLASL_2000
        for k, per_cls_stat in per_cls_stat_dict.items():
            top1_t, top1_f, top5_t, top5_f, top10_t, top10_f = per_cls_stat
            evaluation_results[f'{k}per_cls_top_1'] = np.nanmean((top1_t / (top1_t+top1_f)).cpu().numpy())
            logger.info('-------------------------{}Per-class ACC Top-1: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_cls_top_1']))
            evaluation_results[f'{k}per_cls_top_5'] = np.nanmean((top5_t / (top5_t+top5_f)).cpu().numpy())
            logger.info('-------------------------{}Per-class ACC Top-5: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_cls_top_5']))
            evaluation_results[f'{k}per_cls_top_10'] = np.nanmean((top10_t / (top10_t+top10_f)).cpu().numpy())
            logger.info('-------------------------{}Per-class ACC Top-10: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_cls_top_10']))

        if wandb_run:
            for k, v in evaluation_results.items():
                wandb.log({f'eval/{k}': 100*v})

        if save_dir:
            with open(os.path.join(save_dir, 'evaluation_results.pkl'), 'wb') as f:
                pickle.dump(evaluation_results, f)
    if sync:
        synchronize()
    return evaluation_results


def eval_denoise(model, val_dataloader, cfg, 
        tb_writer=None, wandb_run=None,
        epoch=None, global_step=None,
        generate_cfg={}, save_dir=None, return_prob=False, return_others=False):
    
    logger = get_logger()
    tot = cor = 0
    for step, batch in enumerate(val_dataloader):
        #forward -- loss
        batch = move_to_device(batch, cfg['device'])
        forward_output = model(is_train=False, labels=batch['labels'], sgn_videos=batch['sgn_videos'], sgn_keypoints=batch['sgn_keypoints'], epoch=epoch,
                               denoise_inputs=batch.get('denoise_inputs', {}))
        logits = forward_output['logits']
        decode_results = model.predict_gloss_from_logits(logits, k=10)
        decode_results = decode_results[:,0]
        labels = batch['denoise_inputs']['labels']
        tot += labels.shape[0]
        cor += (decode_results==labels).sum().item()

    evaluation_results = {'per_ins_top_1': cor/tot}
    print()
    logger.info('-----------------------per-ins accuracy: {:.2f}------------------------'.format(100*cor/tot))
    return evaluation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SLT baseline Testing")
    parser.add_argument("--config", default="configs/default.yaml", type=str, help="Training configuration file (yaml).")
    parser.add_argument("--save_subdir", default='prediction', type=str)
    parser.add_argument('--ckpt_name', default='best.ckpt', type=str)
    parser.add_argument('--eval_setting', default='origin', type=str)
    # parser.add_argument('--split', default='test', type=str)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg['local_rank'], cfg['world_size'], cfg['device'] = init_DDP()
    cfg['rank'] = torch.distributed.get_rank()
    set_seed(seed=cfg["training"].get("random_seed", 42))
    model_dir = cfg['training']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    global logger
    logger = make_logger(model_dir=model_dir, log_file='prediction_{}_{}.log'.format(args.eval_setting, cfg['rank']))

    dataset = build_dataset(cfg['data'], 'train')
    vocab = dataset.vocab
    cls_num = len(vocab)
    word_emb_tab = []
    if dataset.word_emb_tab is not None:
        for w in vocab:
            word_emb_tab.append(torch.from_numpy(dataset.word_emb_tab[w]))
        word_emb_tab = torch.stack(word_emb_tab, dim=0).float().to(cfg['device'])
    else:
        word_emb_tab = None
    del vocab; del dataset
    model = build_model(cfg, cls_num, word_emb_tab=word_emb_tab)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    #load model
    load_model_path = os.path.join(model_dir,'ckpts',args.ckpt_name)
    if os.path.isfile(load_model_path):
        state_dict = torch.load(load_model_path, map_location='cuda')
        neq_load_customized(model, state_dict['model_state'], verbose=True)
        epoch, global_step = state_dict.get('epoch',0), state_dict.get('global_step',0)
        logger.info('Load model ckpt from '+load_model_path)
    else:
        logger.info(f'{load_model_path} does not exist')
        epoch, global_step = 0, 0
    
    model = DDP(model, 
            device_ids=[cfg['local_rank']], 
            output_device=cfg['local_rank'],
            find_unused_parameters=True)

    for split in ['dev', 'test']:
        logger.info('Evaluate on {} set'.format(split))
        if args.eval_setting == 'origin':
            dataloader, sampler = build_dataloader(cfg, split, is_train=False, val_distributed=True)
            per_ins_stat, per_cls_stat, _, _, others = evaluation(model=model.module, val_dataloader=dataloader, cfg=cfg, 
                    epoch=epoch, global_step=global_step, 
                    generate_cfg=cfg['testing']['cfg'],
                    save_dir=os.path.join(model_dir,args.save_subdir,split), 
                    return_prob=False, return_others=False)
            
            sync_results(per_ins_stat, per_cls_stat)
            # if cfg['data']['dataset_name'] == 'WLASL_2000':
            #     other_vocab = [1000, 300, 100]
            # elif cfg['data']['dataset_name'] == 'MSASL_1000':
            #     other_vocab = [500, 200, 100]
            # else:
            #     other_vocab = []
            # for o in other_vocab:
            #     logger.info('-----------------------Variant: {:d}-------------------------'.format(o))
            #     sync_results(others[str(o)]['per_ins_stat'], others[str(o)]['per_cls_stat'])
        
        elif args.eval_setting in ['3x', '5x', 'model_ens', 'central_random_1', 'central_random_2', '5x_random_1', '5x_random_2',
                                    '3x_pad', '3x_left_mid', '3x_left_mid_pad']:
            if args.eval_setting in ['3x', '3x_pad', '3x_left_mid', '3x_left_mid_pad', '5x']:
                if args.eval_setting == '3x':
                    test_p = ['start', 'end', 'central']
                    test_m = ['pad', 'pad', 'pad']
                elif args.eval_setting == '3x_pad':
                    test_p = ['start', 'end', 'central']
                    test_m = ['start_pad', 'end_pad', 'pad']
                elif args.eval_setting == '3x_left_mid':
                    test_p = ['left_mid', 'right_mid', 'central']
                    test_m = ['pad', 'pad', 'pad']
                elif args.eval_setting == '3x_left_mid_pad':
                    test_p = ['left_mid', 'right_mid', 'central']
                    test_m = ['left_mid_pad', 'right_mid_pad', 'pad']
                else:
                    test_p = ['left_mid', 'right_mid', 'start', 'end', 'central']
                    test_m = ['left_mid_pad', 'right_mid_pad', 'start_pad', 'end_pad', 'pad']
                    # test_m = ['pad', 'pad', 'pad', 'pad', 'pad']
                    # test_p = ['start']
                    # test_m = ['pad']
                all_prob = {}
                for t_p, t_m in zip(test_p, test_m):
                    logger.info('----------------------------------crop position: {}----------------------------'.format(t_p))
                    new_cfg = deepcopy(cfg)
                    new_cfg['data']['transform_cfg']['index_setting'][2] = t_p
                    new_cfg['data']['transform_cfg']['index_setting'][3] = t_m
                    dataloader, sampler = build_dataloader(new_cfg, split, is_train=False, val_distributed=False)
                    per_ins_stat, per_cls_stat, results, name_prob, _ = evaluation(model=model.module, val_dataloader=dataloader, cfg=new_cfg, 
                                            epoch=epoch, global_step=global_step, 
                                            generate_cfg=cfg['testing']['cfg'],
                                            save_dir=os.path.join(model_dir,args.save_subdir,split), return_prob=True)
                    all_prob[t_p] = name_prob
                with open(os.path.join(model_dir,args.save_subdir,split,'prob_5x.pkl'), 'wb') as f:
                    pickle.dump(all_prob, f)

            elif args.eval_setting == 'model_ens':
                all_prob = {}
                # with open('./results_debug/two_lbsm0.2_wordembsim_dual_top2k_ema_fc2_dec_mixup_0.75_0.8/prediction/test/prob_5x.pkl', 'rb') as f:
                #     all_prob['m1'] = pickle.load(f)
                # with open('./results_debug/two_lbsm0.2_wordembsim_dual_top2k_ema_fc2_dec_mixup_0.75_0.8/prediction/test/results_0.pkl', 'rb') as f:
                #     results = pickle.load(f)
                # with open('./results_debug/two_best_frame32_train/prediction/test/prob_5x.pkl', 'rb') as f:
                #     all_prob['m2'] = pickle.load(f)
                new_cfg = deepcopy(cfg)
                dataloader, sampler = build_dataloader(new_cfg, split, is_train=False, val_distributed=False)
                per_ins_stat, per_cls_stat, results, name_prob, _ = evaluation(model=model.module, val_dataloader=dataloader, cfg=new_cfg, 
                                        epoch=epoch, global_step=global_step, 
                                        generate_cfg=cfg['testing']['cfg'],
                                        save_dir=os.path.join(model_dir,args.save_subdir,split), return_prob=True)
                all_prob['start'] = name_prob

            elif args.eval_setting in ['central_random_1', 'central_random_2', '5x_random_1', '5x_random_2']:
                new_cfg = deepcopy(cfg)
                new_cfg['data']['transform_cfg']['from64'] = 'random'

                if 'central' in args.eval_setting:
                    test_p = ['central']
                    test_m = ['pad']
                else:
                    test_p = ['left_mid', 'right_mid', 'start', 'end', 'central']
                    test_m = ['pad', 'pad', 'pad', 'pad', 'pad']
                
                all_prob = {}
                for t_p, t_m in zip(test_p, test_m):
                    logger.info('----------------------------------crop position: {}----------------------------'.format(t_p))
                    temp_cfg = deepcopy(new_cfg)
                    temp_cfg['data']['transform_cfg']['index_setting'][2] = t_p
                    temp_cfg['data']['transform_cfg']['index_setting'][3] = t_m
                    dataloader, sampler = build_dataloader(temp_cfg, split, is_train=False, val_distributed=False)

                    times = int(args.eval_setting.split('_')[-1])
                    for i in range(times):
                        per_ins_stat, per_cls_stat, results, name_prob, _ = evaluation(model=model.module, val_dataloader=dataloader, cfg=temp_cfg, 
                                                epoch=epoch, global_step=global_step, 
                                                generate_cfg=cfg['testing']['cfg'],
                                                save_dir=os.path.join(model_dir,args.save_subdir,split), return_prob=True)
                        all_prob[t_p+'_'+str(i)] = name_prob
            
            if len(cfg['data']['input_streams']) == 1:
                if type(cfg['data']['num_output_frames']) == int:
                    logits_name_lst = ['']
                else:
                    logits_name = ['frame_ensemble_', '32_', '64_']
            elif len(cfg['data']['input_streams']) == 4:
                logits_name_lst = ['ensemble_last_', 'ensemble_all_']
            else:
                logits_name_lst = ['ensemble_last_', 'fuse_']
            
            evaluation_results = compute_accuracy(results, logits_name_lst, cls_num, cfg['device'], 
                all_prob=all_prob, eval_setting=args.eval_setting)
            for logits_name in logits_name_lst:
                logger.info('-------------------------{}Per-instance ACC Top-1: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_ins_top_1']))
                logger.info('-------------------------{}Per-instance ACC Top-5: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_ins_top_5']))
                logger.info('-------------------------{}Per-instance ACC Top-10: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_ins_top_10']))

                # one class missing in the test set of WLASL_2000
                logger.info('-------------------------{}Per-class ACC Top-1: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_cls_top_1']))
                logger.info('-------------------------{}Per-class ACC Top-5: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_cls_top_5']))
                logger.info('-------------------------{}Per-class ACC Top-10: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_cls_top_10']))
                logger.info('-------------------------Evaluation Finished-------------------------')
            
            # if cfg['data']['dataset_name'] == 'WLASL_2000':
            #     other_vocab = [1000, 300, 100]
            # elif cfg['data']['dataset_name'] == 'MSASL_1000':
            #     other_vocab = [500, 200, 100]
            # else:
            #     other_vocab = []
            # for o in other_vocab:
            #     name_lst = dataloader.dataset.other_vars[str(o)][split]
            #     effective_label_idx = dataloader.dataset.other_vars[str(o)]['vocab_idx']
            #     evaluation_results = compute_accuracy(results, logits_name_lst, cls_num, cfg['device'], name_lst, 
            #         effective_label_idx, all_prob, args.eval_setting)
            #     logger.info('-------------------------Variant: {:d}-------------------------'.format(o))
            #     for logits_name in logits_name_lst:
            #         logger.info('-------------------------{}Per-instance ACC Top-1: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_ins_top_1']))
            #         logger.info('-------------------------{}Per-instance ACC Top-5: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_ins_top_5']))
            #         logger.info('-------------------------{}Per-instance ACC Top-10: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_ins_top_10']))

            #         # one class missing in the test set of WLASL_2000
            #         logger.info('-------------------------{}Per-class ACC Top-1: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_cls_top_1']))
            #         logger.info('-------------------------{}Per-class ACC Top-5: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_cls_top_5']))
            #         logger.info('-------------------------{}Per-class ACC Top-10: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_cls_top_10']))
            #         logger.info('-------------------------Evaluation Finished-------------------------')
