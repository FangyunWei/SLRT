import os
import numpy as np
import cv2
import torch
from modelling.models_3d.S3D.model import S3D

def main():
    ''' Output the top 5 Kinetics classes predicted by the model '''
    path_sample = './sample'
    file_weight = './S3D_kinetics400.pt'
    class_names = [c.strip() for c in open('./label_map.txt')]
    num_class = 400

    model = S3D(num_class)

    # load the weight file and copy the parameters
    if os.path.isfile(file_weight):
        print ('loading weight file')
        weight_dict = torch.load(file_weight)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print (' size? ' + name, param.size(), model_dict[name].size())
            else:
                print (' name? ' + name)

        print (' loaded')
    else:
        print ('weight file?')

    model = model.cuda()
    torch.backends.cudnn.benchmark = False
    model.eval()

    list_frames = [f for f in os.listdir(path_sample) if os.path.isfile(os.path.join(path_sample, f))]
    list_frames.sort()

    # read all the frames of sample clip
    snippet = []
    for frame in list_frames:
        img = cv2.imread(os.path.join(path_sample, frame))
        img = img[...,::-1]
        snippet.append(img)

    clip = transform(snippet)

    with torch.no_grad():
        logits = model(clip.cuda()).cpu().data[0]

    preds = torch.softmax(logits, 0).numpy()
    sorted_indices = np.argsort(preds)[::-1][:5]

    print ('\nTop 5 classes ... with probability')
    for idx in sorted_indices:
        print (class_names[idx], '...', preds[idx])


def transform(snippet):
    ''' stack & noralization '''
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)

    return snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)


if __name__ == '__main__':
    main()

