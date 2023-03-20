import torch
import numpy as np


def compute_accuracy(results, logits_name_lst, cls_num, device, name_lst=[], effective_label_idx=[], all_prob=None, eval_setting='origin'):
    per_ins_stat_dict, per_cls_stat_dict = {}, {}
    if eval_setting == 'origin':
        for k in logits_name_lst:
            correct = correct_5 = correct_10 = num_samples = 0
            top1_t = np.zeros(cls_num, dtype=np.int32)
            top1_f = np.zeros(cls_num, dtype=np.int32)
            top5_t = np.zeros(cls_num, dtype=np.int32)
            top5_f = np.zeros(cls_num, dtype=np.int32)
            top10_t = np.zeros(cls_num, dtype=np.int32)
            top10_f = np.zeros(cls_num, dtype=np.int32)
            for name in results.keys():
                if len(name_lst) > 0 and name not in name_lst:
                    continue

                res = results[name]
                # if len(cfg['data']['input_streams']) == 1:
                #     hyp_lst = res['hyp']
                # elif len(cfg['data']['input_streams']) > 1:
                #     hyp_lst = res['ensemble_last_hyp']
                hyp = res[f'{k}hyp']
                #update hyp list
                if len(effective_label_idx) > 0:
                    hyp_lst = []
                    for h in hyp:
                        if h in effective_label_idx:
                            hyp_lst.append(h)
                            effective_label_idx.remove(h)
                else:
                    hyp_lst = hyp
                ref = res['ref']
                
                if ref == hyp_lst[0]:
                    correct += 1
                    top1_t[ref] += 1
                else:
                    top1_f[ref] += 1

                if ref in hyp_lst[:5]:
                    correct_5 += 1
                    top5_t[ref] += 1
                else:
                    top5_f[ref] += 1

                if ref in hyp_lst[:10]:
                    correct_10 += 1
                    top10_t[ref] += 1
                else:
                    top10_f[ref] += 1
                num_samples += 1

            per_ins_stat = torch.tensor([correct, correct_5, correct_10, num_samples], dtype=torch.float32, device=device)
            per_cls_stat = torch.stack([torch.from_numpy(top1_t),torch.from_numpy(top1_f),
                                        torch.from_numpy(top5_t),torch.from_numpy(top5_f),
                                        torch.from_numpy(top10_t),torch.from_numpy(top10_f)], dim=0).float().to(device)
            per_ins_stat_dict[k] = per_ins_stat
            per_cls_stat_dict[k] = per_cls_stat
        return per_ins_stat_dict, per_cls_stat_dict
    
    elif eval_setting in ['3x', '5x', 'central_random_1', 'central_random_2', '5x_random_1', '5x_random_2',
                            '3x_pad', '3x_left_mid', '3x_left_mid_pad']:
        evaluation_results = {}
        for logits_name in logits_name_lst:
            evaluation_results[logits_name] = {}
            correct = correct_5 = correct_10 = num_samples = 0
            top1_t = np.zeros(cls_num, dtype=np.int32)
            top1_f = np.zeros(cls_num, dtype=np.int32)
            top5_t = np.zeros(cls_num, dtype=np.int32)
            top5_f = np.zeros(cls_num, dtype=np.int32)
            top10_t = np.zeros(cls_num, dtype=np.int32)
            top10_f = np.zeros(cls_num, dtype=np.int32)

            if eval_setting in ['3x', '5x', '3x_pad', '3x_left_mid', '3x_left_mid_pad']:
                k_lst = all_prob['central'].keys()
            else:
                k_lst = all_prob['central_0'].keys()

            for name in k_lst:
                cur_logits_name = name.replace(name.split('_')[-1], '')
                # print(cur_logits_name, logits_name)
                if logits_name not in cur_logits_name:
                    continue

                # avg_prob = all_prob['start'][name]
                if eval_setting in ['3x', '3x_pad']:
                    avg_prob = (all_prob['start'][name]+all_prob['central'][name]+all_prob['end'][name]) / 3
                elif eval_setting in ['3x_left_mid', '3x_left_mid_pad']:
                    avg_prob = (all_prob['left_mid'][name]+all_prob['central'][name]+all_prob['right_mid'][name]) / 3
                elif eval_setting == '5x':
                    avg_prob = (all_prob['start'][name]+all_prob['central'][name]+all_prob['end'][name]+all_prob['left_mid'][name]+all_prob['right_mid'][name]) / 5
                elif eval_setting == 'model_ens':
                    avg_prob = (all_prob['start'][name] + all_prob['extra'][name].to(device)) / 2
                    # avg_prob = (all_prob['m1']['start'][name]+all_prob['m1']['central'][name]+all_prob['m1']['end'][name]+all_prob['m1']['left_mid'][name]+all_prob['m1']['right_mid'][name] +\
                    #     all_prob['m2']['start'][name]+all_prob['m2']['central'][name]+all_prob['m2']['end'][name]+all_prob['m2']['left_mid'][name]+all_prob['m2']['right_mid'][name]) / 10
                elif eval_setting in ['central_random_1', 'central_random_2', '5x_random_1', '5x_random_2']:
                    if 'central' in eval_setting:
                        pos = ['central']
                    else:
                        pos = ['left_mid', 'right_mid', 'start', 'end', 'central']
                    times = int(eval_setting.split('_')[-1])
                    avg_prob = torch.zeros(cls_num).to(device)
                    for i in range(times):
                        for p in pos:
                            avg_prob = avg_prob + all_prob[p+'_'+str(i)][name]
                    avg_prob = avg_prob / (times * len(pos))
                
                video_name = name.replace(logits_name, '')
                if len(name_lst) > 0 and video_name not in name_lst:
                    continue
                ref = results[video_name]['ref']
                hyp = torch.argsort(avg_prob, descending=True)[:2000]
                if len(effective_label_idx) > 0:
                    hyp_lst = []
                    for h in hyp:
                        if h in effective_label_idx:
                            hyp_lst.append(h)
                            effective_label_idx.remove(h)
                else:
                    hyp_lst = hyp
                
                if ref == hyp_lst[0]:
                    correct += 1
                    top1_t[ref] += 1
                else:
                    top1_f[ref] += 1

                if ref in hyp_lst[:5]:
                    correct_5 += 1
                    top5_t[ref] += 1
                else:
                    top5_f[ref] += 1

                if ref in hyp_lst[:10]:
                    correct_10 += 1
                    top10_t[ref] += 1
                else:
                    top10_f[ref] += 1
                num_samples += 1

            # assert len(list(results.keys())) == len(list(all_prob['start'].keys())), f"{len(list(results.keys()))} v.s. {len(list(all_prob['start'].keys()))}"
            evaluation_results[logits_name]['per_ins_top_1'] = correct / num_samples
            evaluation_results[logits_name]['per_ins_top_5'] = correct_5 / num_samples
            evaluation_results[logits_name]['per_ins_top_10'] = correct_10 / num_samples

            # one class missing in the test set of WLASL_2000
            evaluation_results[logits_name]['per_cls_top_1'] = np.nanmean(top1_t / (top1_t+top1_f))
            evaluation_results[logits_name]['per_cls_top_5'] = np.nanmean(top5_t / (top5_t+top5_f))
            evaluation_results[logits_name]['per_cls_top_10'] = np.nanmean(top10_t / (top10_t+top10_f))
        return evaluation_results
