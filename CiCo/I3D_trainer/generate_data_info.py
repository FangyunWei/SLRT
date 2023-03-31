import os
import cv2
def get_frame(video_file):
    if not os.path.isfile(video_file):
        return None
    cap = cv2.VideoCapture(video_file)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return frames
def generate_file_info(paras):
    file = open(f'misc/{paras.dataset}/class.txt')
    class_dict={}
    line = file.readline().strip()
    label, class_name = line.split(' ')
    class_dict[class_name] = label
    while line and line!='':
        line = file.readline().strip()
        if line=='':
            break
        label,class_name=line.split(' ')
        class_dict[class_name]=label
    file.close()
    json_info={}
    path_list=[]
    split_list=[]
    class_name_list=[]
    class_label_list=[]
    frame_list=[]
    for split in ['train','test']:
        root = f"{paras.pseudo_path}/{split}"
        all_words=os.listdir(root)
        for single_word in all_words:
            class_label=class_dict[single_word]
            class_name=single_word
            single_word_dir=os.path.join(root,single_word)
            for single_video in os.listdir(single_word_dir):
                video_path=os.path.join(root,single_word,single_video)
                path_list.append(video_path)
                class_name_list.append(class_name)
                class_label_list.append(class_label)
                frame_list.append(get_frame(video_path))
                split_list.append(split)
    json_info['class_label']=class_label_list
    json_info['class_name']=class_name_list
    json_info['video_path']=path_list
    json_info['frame']=frame_list
    json_info['split']=split_list

    def save_dict(filename, dic):
        import json
        '''save dict into json file'''
        with open(filename, 'w') as json_file:
            json.dump(dic, json_file, ensure_ascii=False)
    save_dict(f"misc/{paras.dataset}/train_test_info.json",json_info)
import argparse
parser = argparse.ArgumentParser(description="generate dataset info")
parser.add_argument('--DEBUG', action='store_true')
parser.add_argument('--pseudo_path', type=str, default='pseudo_from_i3d')
parser.add_argument('--dataset', type=str, default='H2S')
args = parser.parse_args()
generate_file_info(args)