import argparse
import os, json, math, pickle
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from glob import glob
import pandas as pd
from tqdm import tqdm



def crop_video(videofile, seg_box, output_prefix,segments,target_width=256):
    cap = cv2.VideoCapture(videofile)
    w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # Now we start
    imgs = []
    wt=target_width
    ht=wt
    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret==True:
            # get_frames
            imgs.append(frame)
        else:
            break
    cap.release()

    #fps
    frame_segments = []
    for ii, seg in enumerate(segments):
        out_file = os.path.join(output_prefix+f'_clip{ii}.mp4')
        filename=output_prefix.split('/')[-1]+f'_clip{ii}.mp4'
        if filename not in seg_box:
            continue
        box=seg_box[filename]
        x, y, x2, y2, w, h = box['x1'], box['y1'], box['x2'], box['y2'], box['w'], box['h']
        s, e = seg['start_sec'], seg['end_sec']
        sf, ef = int(math.floor(s*fps)), int(math.ceil(e*fps))
        ef = int(min(ef, frames))
        if (ef-sf)<=2:
            print(out_file, 'too short, skip')
            continue
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file, fourcc, fps, (wt, ht))
        for img_ in imgs[sf:ef+1]:
            img_=img_[y:y + h, x:x+ w]
            img_ = Image.fromarray(img_)
            img_=img_.resize((wt,ht))
            out.write(np.asarray(img_))
        entry = {'name':os.path.basename(output_prefix+f'_clip{ii}'),
                 'text':seg['sentence'],
                 'text_id':seg['sentence_id'],
                 'num_frames':ef-sf+1}
        frame_segments.append(entry)
        out.release()
    return frame_segments


parser = argparse.ArgumentParser(description="segment raw_videos by re-aligned boundaries")
parser.add_argument('--DEBUG', action='store_true')
parser.add_argument('--output_dir', type=str, default='./data/How2Sign/processed_videos')
parser.add_argument('--video_dir', type=str, default='./data/How2Sign/raw_videos')
parser.add_argument('--split', type=str, default='train')
args = parser.parse_args()
os.makedirs(os.path.join(args.output_dir,args.split), exist_ok=True)

#get the initial video list
processed_videos = sorted(os.listdir(os.path.join(args.video_dir,args.split)))
dir=args.video_dir
split=args.split
video2segs = defaultdict(list)


#load the bounding box of each segment
with open(f'./data_preparation/bounding_box_coco_{split}.pkl','rb') as f:
    bounding_box_list = pickle.load(f)

#get the re-aligned info
realigned_dataframe = pd.read_csv(f'./data_preparation/how2sign_realigned_{split}.csv',delimiter='\t')
for i in range(len(realigned_dataframe)):
    video_name = realigned_dataframe.at[i,'VIDEO_NAME']
    sentence_id = realigned_dataframe.at[i,'SENTENCE_ID']
    start_sec, end_sec = realigned_dataframe.at[i,'START_REALIGNED'],realigned_dataframe.at[i,'END_REALIGNED']
    sentence = realigned_dataframe.at[i,'SENTENCE']
    video2segs[video_name].append(
        {'start_sec':start_sec, 'end_sec': end_sec,
        'sentence_id':sentence_id, 'sentence':sentence})

for video in video2segs:
    video2segs[video] = sorted(video2segs[video], key=lambda x: x['start_sec'])

print(f'{len(video2segs)} videos (whole set) get sentence boundaries ')

output_data = []

#segment the videos into re-aligned videos and crop each video
for v in tqdm(processed_videos):
    v_ = v.replace('.mp4','')
    if not v_ in video2segs:
        continue
    output_prefix = os.path.join(args.output_dir,split,v.replace('.mp4',''))
    segments = crop_video(videofile=os.path.join(args.video_dir,split, v),
        seg_box = bounding_box_list,
        output_prefix=output_prefix,
        segments=video2segs[v_],
        )

    output_data.extend(segments)
    if args.DEBUG:
        print(output_data)
        break







