from logging import Logger
import warnings, wandb
import pickle
from google.protobuf.reflection import ParseMessage
from collections import defaultdict
from modelling.model import build_model
from modelling.recognition import ctc_decode_func
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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP, distributed
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from utils.misc import (
    DATASETS,
    get_logger,
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger, make_writer,
    set_seed,
    symlink_update,
    is_main_process, init_DDP, move_to_device,
    neq_load_customized
)
from dataset.Dataloader import build_dataloader
from utils.progressbar import ProgressBar
from utils.metrics import bleu, rouge, wer_list, wer_list_per_sen
from utils.phoenix_cleanup import clean_phoenix_2014_trans, clean_phoenix_2014
from copy import deepcopy
from itertools import groupby
from opencc import OpenCC


def clean_tvb(s):
    op = []
    for t in s.split():
        if '<' in t and '>' in t:
            continue
        op.append(t)
    return ' '.join(op)


def phoenix_gls_mapping(pred, fname):
    with open(fname, 'rb') as f:
        gls_mapping = pickle.load(f)
    op = []
    for p in pred:
        if p not in gls_mapping:
            # print(p)
            continue
        op.append(gls_mapping[p])
    return op


def evaluation(model, val_dataloader, cfg, 
        tb_writer=None, wandb_run=None,
        epoch=None, global_step=None,
        generate_cfg={}, save_dir=None,
        do_translation=True, do_recognition=True, external_logits=None, save_logits=False):  #to-do output_dir
    logger = get_logger()
    logger.info(generate_cfg)
    print()
    if external_logits is not None:
        with open(external_logits, 'rb') as f:
            external_logits = pickle.load(f)
    if os.environ.get('enable_pbar', '1')=='1':
        pbar = ProgressBar(n_total=len(val_dataloader), desc='Validation')
    else:
        pbar = None
    if epoch!=None:
        logger.info('Evaluation epoch={} validation examples #={}'.format(epoch, len(val_dataloader.dataset)))
    elif global_step!=None:
        logger.info('Evaluation global step={} validation examples #={}'.format(global_step, len(val_dataloader.dataset)))
    model.eval()
    total_val_loss = defaultdict(int)
    dataset2results = defaultdict(lambda: defaultdict(dict))
    cc = OpenCC('s2t')
    logits_dict = {}
    tot_time = 0
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            #forward -- loss
            batch = move_to_device(batch, cfg['device'])
            # print(batch['gloss'])
            datasetname = batch['datasetname']
            st_time = time.time()
            forward_output = model(is_train=False, **batch)
            time_cost = time.time() - st_time
            tot_time += time_cost
            for k,v in forward_output.items():
                if '_loss' in k:
                    total_val_loss[k] += v.item()
            if do_recognition: #wer
                #rgb/keypoint/fuse/ensemble_last_logits
                for k, gls_logits in forward_output.items():
                    if not 'gloss_logits' in k or gls_logits==None:
                        continue

                    logits_name = k.replace('gloss_logits','')
                    if logits_name in ['rgb_','keypoint_','fuse_','ensemble_last_','ensemble_early_','']:
                        if logits_name=='ensemble_early_':
                            input_lengths = forward_output['aux_lengths']['rgb'][-1]
                        else:
                            input_lengths = forward_output['input_lengths']
                        ctc_decode_output = model.predict_gloss_from_logits(
                            gloss_logits=gls_logits, 
                            beam_size=generate_cfg['recognition']['beam_size'], 
                            input_lengths=input_lengths,
                            datasetname=datasetname,
                            lm=cfg.get('lm', None),
                            alpha=cfg.get('alpha', 0.0)
                        )
                        if save_logits and logits_name in ['ensemble_last_']:
                            logits_dict[batch['name'][0]] = gls_logits.detach().cpu()
                        batch_pred_gls = model.gloss_tokenizer.convert_ids_to_tokens(ctc_decode_output, datasetname)       
                        # print(batch_pred_gls, batch['gloss'])
                        for name, gls_hyp, gls_ref in zip(batch['name'], batch_pred_gls, batch['gloss']):
                            dataset2results[datasetname][name][f'{logits_name}gls_hyp'] = \
                                ' '.join(gls_hyp).upper() if model.gloss_tokenizer.lower_case \
                                    else ' '.join(gls_hyp)
                            dataset2results[datasetname][name]['gls_ref'] = gls_ref.upper() if model.gloss_tokenizer.lower_case \
                                    else gls_ref
                    else:
                        raise ValueError


            if do_translation or cfg['task'] == 'T2G':
                generate_output = model.generate_txt(
                    transformer_inputs=forward_output['transformer_inputs'],
                    generate_cfg=generate_cfg['translation'])
                #decoded_sequences
                for name, hyp, txt_ref, gls_ref in zip(batch['name'], generate_output['decoded_sequences'], batch['text'], batch['gloss']):
                    if cfg['task'] == 'T2G':
                        clean_hyp = [g for g in hyp.split(' ') if g not in model.gloss_tokenizer.special_tokens]
                        clean_hyp = ' '.join(clean_hyp)
                        dataset2results[datasetname][name]['gls_hyp'] = cc.convert(clean_hyp).upper() if model.gloss_tokenizer.lower_case else hyp
                        dataset2results[datasetname][name]['gls_ref'] = cc.convert(gls_ref).upper() if model.gloss_tokenizer.lower_case else gls_ref
                        
                    else:
                        dataset2results[datasetname][name]['txt_hyp'], dataset2results[datasetname][name]['txt_ref'] = hyp, txt_ref
                        # print('hyp: ', hyp, 'ref: ', txt_ref)

            #misc
            if pbar:
                pbar(step)
        print()
    logger.info('#samples: {}, average time cost per video: {}s'.format(len(val_dataloader), tot_time/len(val_dataloader)))
    #logging and tb_writer
    for k, v in total_val_loss.items():
        logger.info('{} Average:{:.2f}'.format(k, v/len(val_dataloader)))
        if tb_writer:
            tb_writer.add_scalar('eval/'+k, v/len(val_dataloader), epoch if epoch!=None else global_step)
        if wandb_run:
            wandb.log({f'eval/{k}': v/len(val_dataloader)})
    #evaluation (Recognition:WER,  Translation:B/M)
    dataset2evaluation_results = {}
    for datasetname, results in dataset2results.items():
        evaluation_results = {}
        name = list(results.keys())[0]
        if do_recognition or cfg['task'] == 'T2G':
            evaluation_results['wer'] = 200
            for hyp_name in results[name].keys():
                if not 'gls_hyp' in hyp_name:
                    continue
                k = hyp_name.replace('gls_hyp','')
                if datasetname in ['phoenix', 'phoenix2014tsi', 'phoenix_syn', 'phoenix_syn_gt', 'phoenix_syn_smplx', 'phoenix_syn_smplx_gt']:
                    gls_ref = [clean_phoenix_2014_trans(results[n]['gls_ref']) for n in results]
                    gls_hyp = [clean_phoenix_2014_trans(results[n][hyp_name]) for n in results] 
                elif datasetname in ['phoenix2014', 'phoenix2014si', 'phoenixcomb']:
                    gls_ref = [clean_phoenix_2014(results[n]['gls_ref']) for n in results]
                    gls_hyp = [clean_phoenix_2014(results[n][hyp_name]) for n in results] 
                elif datasetname in ['csl','cslr','csl_syn_gt','wlasl2000','tvb']:
                    gls_ref = [results[n]['gls_ref'] for n in results]
                    gls_hyp = [results[n][hyp_name] for n in results]
                # print(gls_hyp, gls_ref)
                wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
                wer_results_per_sen = wer_list_per_sen(hypotheses=gls_hyp, references=gls_ref)
                evaluation_results[k+'wer_list'] = wer_results
                logger.info('{}-{}WER: {:.2f}'.format(datasetname, k, wer_results['wer']))
                logger.info('{}-{}WER_per_sen: {:.2f}'.format(datasetname, k, wer_results_per_sen['wer']))
                if cfg['task'] == 'T2G':
                    bleu_dict = bleu(references=gls_ref, hypotheses=gls_hyp, level=cfg['data']['level'])
                    rouge_score = rouge(references=gls_ref, hypotheses=gls_hyp, level=cfg['data']['level'])
                    for k,v in bleu_dict.items():
                        logger.info('{}-{} {:.2f}'.format(datasetname, k, v))
                    logger.info('{}-ROUGE: {:.2f}'.format(datasetname, rouge_score))
                    evaluation_results['rouge'], evaluation_results['bleu'] = rouge_score, bleu_dict
                if tb_writer:
                    tb_writer.add_scalar(f'eval_{datasetname}/{k}WER', wer_results['wer'], epoch if epoch!=None else global_step)   
                if wandb_run!=None:
                    wandb.log({f'eval_{datasetname}/{k}WER': wer_results['wer']})  
                evaluation_results['wer'] = min(wer_results['wer'], evaluation_results['wer'])
            
            if datasetname == 'tvb' and not do_translation:
                gls_ref = [clean_tvb(results[n]['gls_ref']) for n in results]
                gls_hyp = [clean_tvb(results[n]['gls_hyp']) for n in results]
                wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
                wer_results_per_sen = wer_list_per_sen(hypotheses=gls_hyp, references=gls_ref)
                logger.info('{}-cleaned-WER: {:.2f}'.format(datasetname, wer_results['wer']))
                logger.info('{}-cleaned-WER_per_sen: {:.2f}'.format(datasetname, wer_results_per_sen['wer']))

        if do_translation:
            txt_ref = [results[n]['txt_ref'] for n in results]
            txt_hyp = [results[n]['txt_hyp'] for n in results]
            bleu_dict = bleu(references=txt_ref, hypotheses=txt_hyp, level=cfg['data']['level'])
            rouge_score = rouge(references=txt_ref, hypotheses=txt_hyp, level=cfg['data']['level'])
            for k,v in bleu_dict.items():
                logger.info('{}-{} {:.2f}'.format(datasetname, k,v))
            logger.info('{}-ROUGE: {:.2f}'.format(datasetname, rouge_score))
            evaluation_results['rouge'], evaluation_results['bleu'] = rouge_score, bleu_dict
            if tb_writer:
                tb_writer.add_scalar(f'eval_{datasetname}/BLEU4', bleu_dict['bleu4'], epoch if epoch!=None else global_step)
                tb_writer.add_scalar(f'eval_{datasetname}/ROUGE', rouge_score, epoch if epoch!=None else global_step)
            if wandb_run!=None:
                wandb.log({f'eval_{datasetname}/BLEU4': bleu_dict['bleu4']})
                wandb.log({f'eval_{datasetname}/ROUGE': rouge_score})
        #save
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f'{datasetname}_results.pkl'),'wb') as f:
                pickle.dump(results, f)
            with open(os.path.join(save_dir, f'{datasetname}_evaluation_results.pkl'),'wb') as f:
                pickle.dump(evaluation_results, f)
            if save_logits:
                with open(os.path.join(save_dir, f'{datasetname}_logits.pkl'),'wb') as f:
                    pickle.dump(logits_dict, f)
        dataset2evaluation_results[datasetname] = evaluation_results
    return dataset2evaluation_results
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SLT baseline Testing")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--save_subdir",
        default='prediction',
        type=str
    )
    parser.add_argument(
        '--ckpt_name',
        default='best.ckpt',
        type=str
    )
    parser.add_argument(
        '--external_logits',
        default=None,
        type=str
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_dir = cfg['training']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    global logger
    logger = make_logger(model_dir=model_dir, log_file='prediction.log')
    cfg['device'] = torch.device('cuda')
    model = build_model(cfg)
    do_translation, do_recognition = cfg['task'] not in ['S2G','T2G'], cfg['task'] not in ['G2T','T2G'] #(and recognition loss>0 if S2T)
    #load model

    #per-dataset
    for datasetname in cfg['datanames']:
        logger.info('Evaluate '+datasetname)
        load_model_path = os.path.join(model_dir,'ckpts',datasetname+'_'+args.ckpt_name)
        if os.path.isfile(load_model_path):
            state_dict = torch.load(load_model_path, map_location='cuda')
            neq_load_customized(model, state_dict['model_state'], verbose=True)
            epoch, global_step = state_dict.get('epoch',0), state_dict.get('global_step',0)
            logger.info('Load model ckpt from '+load_model_path)
        else:
            logger.info(f'{load_model_path} does not exist')
            epoch, global_step = 0, 0
        cfg_ = deepcopy(cfg)
        cfg_['datanames'] = [datasetname]
        cfg_['data'] = {k:v for k,v in cfg['data'].items() if not k in cfg['datanames'] or k==datasetname}
        for split in ['dev', 'test']:
            logger.info('Evaluate on {} set'.format(split))
            dataloader, sampler = build_dataloader(cfg_, split, model.text_tokenizer, model.gloss_tokenizer, mode='test')
            evaluation(model=model, val_dataloader=dataloader, cfg=cfg_, 
                    epoch=epoch, global_step=global_step, 
                    generate_cfg=cfg_['testing']['cfg'],
                    save_dir=os.path.join(model_dir,args.save_subdir,split),
                    do_translation=do_translation, do_recognition=do_recognition, external_logits=args.external_logits, save_logits=True)

