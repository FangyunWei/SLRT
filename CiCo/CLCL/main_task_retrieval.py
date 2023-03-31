from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from datetime import datetime
import torch
import numpy as np
import random
import os
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.optimization import BertAdam

from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT
global logger

torch.distributed.init_process_group(backend="nccl")

def get_args(description='CLCL on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train",action='store_true' , help="Whether to run training.")
    parser.add_argument("--do_eval",action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--distributed", default=True, help="Whether to imply ditributed learning, must set as True for correct gathering.")
    parser.add_argument("--debug", default=False, help="Whether to debug.")


    ##########  DA  ##########
    parser.add_argument('--combine_type', type=str, default='sum', help='feature combine type')
    parser.add_argument('--features_path', type=str, default='sign_feature/h2s_domain_agnostic', help='feature path')
    parser.add_argument('--features_path_retrain', type=str, default='sign_feature/h2s_domain_aware', help='feature path')
    parser.add_argument('--alpha', type=float, default=0.8, help='feature combine weight')
    ##########  DA  ##########

    ##########  CL  ##########
    parser.add_argument("--dual_mix", default=0.5,type=float, help="Mix weight for two similarity matrix")
    parser.add_argument("--mix_design", default='balance',type=str, help="similarity matrix combine type")
    parser.add_argument("--tau", default=0.07,type=float, help="Learning temperature")
    parser.add_argument("--sim_calcu", default='softmax_max', type=str, help="similarity matrix combine type")
    ##########  CL  ##########


    ##########  TA  ##########
    parser.add_argument('--text_aug', default=True, help='whether to use text augmentation')
    parser.add_argument('--text_aug_choosen', type=str, default='random_swap', help='feature path',choices=['synonym_replacement','random_deletion','random_swap','all'])
    parser.add_argument('--aug_choose', type=str, default='t2v')
    ##########  TA  ##########

    ##########  Net Archi  ##########
    parser.add_argument('--coef_lr', type=float, default=1.0, help='coefficient for bert branch.')
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--not_load_visual', default=False, help="Layer NO. of CLIP need to freeze.")
    ##########  Net Archi  ##########


    ##########  Learning paras  ##########
    parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=256, help='batch size eval')
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    ##########  Learning paras  ##########



    ##########  Token length   ##########
    parser.add_argument('--max_words', type=int, default=32, help='')
    parser.add_argument('--feature_len', type=int ,default=64, help="Whether MIL, has a high priority than use_mil.")
    ##########  Token length   ##########



    parser.add_argument('--data_path', type=str, default='data_h2', help='data pickle file path')
    parser.add_argument('--cross_att_layers', type=int, default=1, help='feature path')
    parser.add_argument('--num_thread_reader', type=int, default=8, help='')
    parser.add_argument('--lr_decay', type=float, default=0.001, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=1, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_frames', type=int, default=128, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')
    parser.add_argument('--gpu_ids', type=list, default=[0,1], help='The GPU used when distributed is off')
    parser.add_argument('--cla_weight_dir', type=str, default='cla_weight', help='Num of pair to output from data loader')


    parser.add_argument("--output_dir", default='new_experiment', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', default=True,
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="h2s", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', default=True, help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="Filip",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")

    args = parser.parse_args()


    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.distributed==True:
        world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(args.local_rank)
        args.world_size = world_size
        rank = torch.distributed.get_rank()
        args.rank = rank


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir,distributed=args.distributed, state_dict=model_state_dict, task_config=args)
    model.to(device)

    return model


def prep_optimizer_freeze_clip(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_train_list=[(n, p) for n, p in param_optimizer if "mlp" not in n ]
    train_list=[(n, p) for n, p in param_optimizer if "mlp" in n ]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']



    print('no_train_list')
    print([n for n,p in no_train_list])

    print('train_list')
    print([n for n,p in train_list])


    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_train_list], 'weight_decay': 0, 'lr': 0},
        {'params': [p for n, p in train_list], 'weight_decay': weight_decay, 'lr': args.lr },
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)
    if args.distributed==True:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()

    return optimizer, scheduler, model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module
    if coef_lr==1:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
        no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

        decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n ]#and "visual" not in n
        decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n ]#or "clip.visual" in n
        if local_rank==0:
            print('decay_clip_param_tp')
            print([n for n,p in decay_clip_param_tp])

            print('decay_noclip_param_tp')
            print([n for n,p in decay_noclip_param_tp])
        no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
        no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

        weight_decay = 0.001
        optimizer_grouped_parameters = [
            {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
            {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
            {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
            {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
        ]

        scheduler = None
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                             schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                             t_total=num_train_optimization_steps, weight_decay=weight_decay,
                             max_grad_norm=1.0)
        if args.distributed==True:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              output_device=local_rank, find_unused_parameters=True)
        else:
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids).cuda()

        return optimizer, scheduler, model
    else:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
        no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

        decay_lower_lr_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n and "visual" not in n and "ln_final." not in n and "text_projection" not in n \
                 and "logit_scale" not in n]  #
        decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n or "clip.visual" in n or "ln_final." in n or "text_projection" in n \
                 or "logit_scale" in n]
        if local_rank==0:
            print('decay_clip_param_tp')
            print([n for n, p in decay_clip_param_tp])

            print('decay_lower_lr_clip_param_tp')
            print([n for n, p in decay_lower_lr_clip_param_tp])


        no_decay_lower_lr_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if  "clip." in n and "visual" not in n and "ln_final." not in n and "text_projection" not in n \
                 and "logit_scale" not in n]  #
        no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n or "clip.visual" in n or "ln_final." in n or "text_projection" in n \
                 or "logit_scale" in n]

        weight_decay = 0.001
        optimizer_grouped_parameters = [
            {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr},
            {'params': [p for n, p in decay_lower_lr_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
            {'params': [p for n, p in no_decay_lower_lr_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
            {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0,'lr': args.lr}
        ]

        scheduler = None
        optimizer = BertAdam(optimizer_grouped_parameters,  warmup=args.warmup_proportion,
                             schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                             t_total=num_train_optimization_steps, weight_decay=weight_decay,
                             max_grad_norm=1.0)
        if args.distributed == True:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              output_device=local_rank, find_unused_parameters=True)
        else:
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids).cuda()

        return optimizer, scheduler, model
def save_acc_plot(epochs,acc_record,args):
    import matplotlib.pyplot as plt
    x0 = [i for i in range(epochs)]

    fig = plt.figure(figsize=(12, 8))
    plt.title('Epochs & Train Loss', fontsize=18)
    plt.plot(x0, acc_record, '.-', label='acc')
    plt.legend(prop={'size': 16})
    # plt.yticks([0,250,500,750,1000,1250,1500,1750,2000])
    plt.xlabel('Epochs', fontsize=16)
    plt.tick_params(labelsize=16)
    # plt.legend() # 将样例显示出来
    plt.savefig(os.path.join(args.output_dir,args.output_dir.split("/")[-1]+'acc.png'))


def save_loss_plot(epochs,loss_record,args):
    import matplotlib.pyplot as plt
    x0 = [i for i in range(epochs)]

    fig = plt.figure(figsize=(12, 8))
    plt.title('Epochs & Train Loss', fontsize=18)
    plt.plot(x0, loss_record, '.-', label='loss')
    plt.legend(prop={'size': 16})
    # plt.yticks([0,250,500,750,1000,1250,1500,1750,2000])
    plt.xlabel('Epochs', fontsize=16)
    plt.tick_params(labelsize=16)
    # plt.legend() # 将样例显示出来
    plt.savefig(os.path.join(args.output_dir,args.output_dir.split("/")[-1]+'loss.png'))

def save_code(args):
    import os
    import shutil
    dst=os.path.join(args.output_dir,'code')
    if os.path.exists(dst):
        dst=dst+'_'+str(datetime.now())
    os.makedirs(dst,exist_ok=True)
    dir_list=['dataloaders','modules']
    for dir in dir_list:
        shutil.copytree(dir,os.path.join(dst,dir))
    file_lists=os.listdir('./')
    for file in file_lists:
        if '.py' in file:
            shutil.copy(file,os.path.join(dst,file))

def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss,
            }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    if args.debug==True:
        print("model allocated:")
        print(torch.cuda.memory_allocated()/1024.0/1024)

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        if args.debug == True:
            print("input allocated:")
            print(torch.cuda.memory_allocated()/1024.0/1024)

        input_ids, input_mask, segment_ids, video, video_mask,pairs_text_aug,pairs_mask_aug = batch
        loss = model(input_ids, segment_ids, input_mask, video, video_mask,pairs_text_aug,pairs_mask_aug)

        if args.debug == True:

            print("forward allocated:")
            print(torch.cuda.memory_allocated()/1024.0/1024)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        if args.debug == True:
            print("backward allocated:")
            print(torch.cuda.memory_allocated()/1024.0/1024)
        torch.cuda.empty_cache()
        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step






def _run_on_single_gpu_new_mix(model, batch_list_v, batch_list_t, batch_sequence_output_list, batch_visual_output_list,sequence_cls, visual_cls,is_train,hybird=False,alpha=0.5,dual_mix=0.5):

    with torch.no_grad():
        sim_matrix_i2t = []
        sim_matrix_t2i = []
        for idx1, b1 in enumerate(batch_list_v):
            video_mask = b1
            visual_output = batch_visual_output_list[idx1]
            # visual_output_cls=visual_cls[idx1]
            each_row = []
            each_row_t2i=[]
            for idx2, b2 in enumerate(batch_list_t):
                text_mask = b2
                sequence_output = batch_sequence_output_list[idx2]
                # sequence_output_cls = sequence_cls[idx2]

                i2t,t2i, *_tmp = model.get_similarity_logits(sequence_output, visual_output, text_mask, video_mask,
                                                                         loose_type=model.loose_type,is_train=True)
                i2t = i2t.cpu().detach().numpy()
                t2i=t2i.cpu().detach().numpy()
                each_row.append(i2t*dual_mix+t2i*(1-dual_mix))
                each_row_t2i.append(i2t*dual_mix+t2i*(1-dual_mix))
            each_row_i2t = np.concatenate(tuple(each_row), axis=-1)
            each_row_t2i = np.concatenate(tuple(each_row_t2i), axis=-1)
            sim_matrix_i2t.append(each_row_i2t)
            sim_matrix_t2i.append(each_row_t2i)
        return sim_matrix_i2t,sim_matrix_t2i







def _run_on_single_gpu(model, batch_list_v, batch_list_t, batch_sequence_output_list, batch_visual_output_list,sequence_cls, visual_cls,is_train,hybird=False,alpha=0.5):
    sim_matrix_i2t = []
    sim_matrix_t2i = []
    for idx1, b1 in enumerate(batch_list_v):
        video_mask = b1
        visual_output = batch_visual_output_list[idx1]
        visual_output_cls=visual_cls[idx1]
        each_row = []
        each_row_t2i=[]
        for idx2, b2 in enumerate(batch_list_t):
            text_mask = b2
            sequence_output = batch_sequence_output_list[idx2]
            sequence_output_cls = sequence_cls[idx2]

            i2t,t2i, *_tmp = model.get_similarity_logits(sequence_output, visual_output, text_mask, video_mask,
                                                                     loose_type=model.loose_type,is_train=True)
            i2t = i2t.cpu().detach().numpy()
            t2i=t2i.cpu().detach().numpy()
            if hybird==True:
                sim_i2t = model.filp_cls_loose_similarity(sequence_output_cls, visual_output_cls, video_mask,
                                                                     video_mask)
                sim_i2t = sim_i2t.cpu().detach().numpy().transpose()
                i2t=i2t*alpha+sim_i2t*(1-alpha)
                t2i=t2i*alpha+sim_i2t*(1-alpha)


            each_row.append(i2t)
            each_row_t2i.append(t2i)
        each_row_i2t = np.concatenate(tuple(each_row), axis=-1)
        each_row_t2i = np.concatenate(tuple(each_row_t2i), axis=-1)
        sim_matrix_i2t.append(each_row_i2t)
        sim_matrix_t2i.append(each_row_t2i)
    return sim_matrix_i2t,sim_matrix_t2i




def eval_epoch(args, model, test_dataloader, device, n_gpu,istrain):
    torch.cuda.empty_cache()
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        batch_sequence_output_list_cls, batch_visual_output_list_cls, = [], []

        total_text_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch

            b, *_t = input_ids.shape
            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                mask,visual_output,visual_cls=model.get_visual_output(video, video_mask,shaped=True, video_frame=1,get_hidden=True)
                batch_visual_output_list.append(visual_output)
                batch_visual_output_list_cls.append(visual_cls)
                batch_list_v.append(mask)

                s_, e_ = total_text_num, total_text_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    input_ids, segment_ids, input_mask = input_ids[filter_inds, ...], segment_ids[filter_inds, ...], segment_ids[input_mask, ...]
                    text_mask,sequence_output, sequence_cls= model.get_sequence_output(input_ids, segment_ids, input_mask,get_hidden=True)
                    batch_sequence_output_list.append(sequence_output)
                    batch_sequence_output_list_cls.append(sequence_cls)
                    batch_list_t.append(text_mask)
                total_text_num += b
            else:
                sequence_output, visual_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask, shaped=True)

                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------


    #########################################


        sim_matrix_i2t,sim_matrix_t2i = _run_on_single_gpu_new_mix(model,  batch_list_v,batch_list_t, batch_sequence_output_list, batch_visual_output_list,\
                                                           batch_sequence_output_list_cls, batch_visual_output_list_cls,is_train=False,dual_mix=args.dual_mix)
        sim_matrix_i2t = np.concatenate(tuple(sim_matrix_i2t), axis=0)
        sim_matrix_t2i = np.concatenate(tuple(sim_matrix_t2i), axis=0)

        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix_i2t.shape[0], sim_matrix_i2t.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])

        sim_matrix_new_i2t = []
        sim_matrix_new_t2i = []

        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            # print(e_-s_,max_length-e_+s_)
            new_matrix_n=np.concatenate((sim_matrix_i2t[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix_i2t.shape[1]), -np.inf)), axis=0)
            new_matrix_n_t2i=np.concatenate((sim_matrix_t2i[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix_t2i.shape[1]), -np.inf)), axis=0)
            # print(new_matrix_n.size)
            sim_matrix_new_i2t.append(new_matrix_n)
            sim_matrix_new_t2i.append(new_matrix_n_t2i)

        sim_matrix_i2t = np.stack(tuple(sim_matrix_new_i2t), axis=0)
        sim_matrix_t2i = np.stack(tuple(sim_matrix_new_t2i), axis=0)

        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix_i2t.shape[0], sim_matrix_i2t.shape[1], sim_matrix_i2t.shape[2]))

        vt_metrics = tensor_text_to_video_metrics(sim_matrix_i2t)
        tv_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix_t2i))
        logger.info("Mix_Text-to-Video:")
        logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
        logger.info("Mix_Video-to-Text:")
        logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                    format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))


    R1 = tv_metrics['R1']
    torch.cuda.empty_cache()
    return R1

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    assert  args.task_type == "retrieval"
    model = init_model(args, device, n_gpu, args.local_rank)

    ## ####################################
    # freeze testing
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    print("frozen_layers")
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0 or name.find("visual") == 0 or name.find("seq") == 0 or name.find("cross_atten")==0:
                continue    # need to train
            elif name.find("visual.transformer.resblocks.") == 0 :
                continue
            elif name.find("transformer.resblocks.") == 0:

                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue    # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False
                print(name)

    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["dev"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None and args.local_rank == 0:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)





    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        save_code(args)

    ## ####################################
    # train and eval
    ## ####################################


    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_epoch=0
        best_output_model_file = "None"
        ## ##############################################################
        # resume optimizer state besides loss to continue trainxz
        ## ##############################################################
        resumed_epoch = 0
        if args.resume_model:
            checkpoint = torch.load(args.resume_model, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resumed_epoch = checkpoint['epoch']+1
            resumed_loss = checkpoint['loss']
        
        global_step = 0
        loss_record=[]
        acc_record=[]
        for epoch in range(resumed_epoch, args.epochs):
            if args.distributed==True:
                train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)
            if args.local_rank == 0 and (epoch+1)%10==0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")

                ## Run on val dataset, this process is *TIME-consuming*.
                # logger.info("Eval on val dataset")
                # R1 = eval_epoch(args, model, val_dataloader, device, n_gpu)
            if args.local_rank == 0:
                R1 = eval_epoch(args, model, test_dataloader, device, n_gpu,False)
                if best_score <= R1:
                    best_score = R1
                    best_epoch=epoch
                    # best_output_model_file = output_model_file
                logger.info("The best model is: {}{}, the R1 is: {:.4f}".format(args.output_dir,best_epoch, best_score))
                loss_record.append(tr_loss)
                acc_record.append(R1)
                print(loss_record)
                print(acc_record)
        if args.local_rank == 0:
            save_acc_plot(args.epochs,acc_record,args)
            save_loss_plot(args.epochs,loss_record,args)



    elif args.do_eval:
        if args.local_rank == 0:
            eval_epoch(args, model, test_dataloader, device, n_gpu,False)

if __name__ == "__main__":
    main()
