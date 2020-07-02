import sys
import os
import numpy as np
import cv2
import torch
from model import TASED_v2
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageDraw, ImageFont

def main():
    path_output = os.path.join('output', 'comparison')
    if not os.path.isdir(path_output):
        os.makedirs(path_output)
    path_indata = os.path.join('Atari_dataset', 'video')
    before_tuning = produce_frames('TASED_updated.pt', 'TASED_original', path_indata)
    after_tuning = produce_frames(os.path.join('Atari_dataset', 'produced_weight_file.pt'), 'finetuned', path_indata)
    ground_truth = produce_gt('ground_truth', path_indata)

    for i in range(len(before_tuning)):
        video_array = []
        before_frames, dname = before_tuning[i]
        after_frames, dname = after_tuning[i]
        gt_frames, dname = ground_truth[i]

        for j in range(len(before_frames)):
            before_frame = Image.fromarray(before_frames[j])
            after_frame = Image.fromarray(after_frames[j])
            gt_frame = Image.fromarray(gt_frames[j])

            height, width, layers = before_frames[0].shape
            video_size = (3*width, height)

            video_frame = Image.new('RGB', (3*width, height), (255,255,255))
            video_frame.paste(before_frame, (0,0))
            video_frame.paste(after_frame, (width, 0))#, 2*width, height))
            video_frame.paste(gt_frame, (2*width, 0))#, 3*width, height))
            video_frame = np.array(video_frame)
            video_array.append(video_frame)
        video = cv2.VideoWriter(os.path.join(path_output, dname + '.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), 24, video_size)
        print('writing video: ' + os.path.join(path_output, dname + '.mp4'))
        for i in range(len(video_array)):
            video.write(video_array[i])
        video.release()

def produce_gt(method_used, path_indata):
    print('ground truth:')
    path_output = os.path.join('output', method_used)
    path_annt = os.path.join('Atari_dataset', 'annotation')

    list_indata = [d for d in os.listdir(path_indata) if os.path.isdir(os.path.join(path_indata, d))]
    list_indata.sort()
    produced_frames = []
    for dname in list_indata:
        print('processing ' + dname)
        list_frames = [f for f in os.listdir(os.path.join(path_indata, dname)) if os.path.isfile(os.path.join(path_indata, dname, f))]
        list_frames.sort()
        if len(list_frames) > 480:  # 10 seconds videos
            list_frames = list_frames[240:480]

        list_maps = [f for f in os.listdir(os.path.join(path_annt, dname, 'maps')) if os.path.isfile(os.path.join(path_annt, dname, 'maps', f))]
        list_maps.sort()
        if len(list_maps) > 480:  # 10 seconds videos
            list_maps = list_maps[240:480]
        video_array = []
        for i in range(len(list_maps)):
            temp_frame = Image.fromarray(cv2.imread(os.path.join(path_indata, dname, list_frames[i]), cv2.IMREAD_COLOR))
            temp_map = Image.fromarray(cv2.imread(os.path.join(path_annt, dname, 'maps', list_maps[i]), cv2.IMREAD_GRAYSCALE))
            red_img = Image.new('RGB', (160, 210), (0, 0, 255))
            temp_frame.paste(red_img, mask=temp_map)
            font_type = ImageFont.truetype('Arial.ttf', 15)
            draw = ImageDraw.Draw(temp_frame)
            draw.text(xy=(10, 160), text=method_used, fill=(255, 255, 255), font=font_type)
            temp_frame = np.array(temp_frame)
            video_array.append(temp_frame)
        produced_frames.append((video_array, dname))

    return produced_frames

def produce_frames(weights, method_used, path_indata):
    ''' read frames in path_indata and generate frame-wise saliency maps in path_output '''
    # optional two command-line arguments
    path_output = os.path.join('output', method_used)
    if len(sys.argv) > 1:
        path_indata = sys.argv[1]
        if len(sys.argv) > 2:
            path_output = sys.argv[2]
    if not os.path.isdir(path_output):
        os.makedirs(path_output)

    len_temporal = 32
    file_weight = weights

    model = TASED_v2()

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

    # iterate over the path_indata directory
    list_indata = [d for d in os.listdir(path_indata) if os.path.isdir(os.path.join(path_indata, d))]
    list_indata.sort()
    produced_frames = []
    for dname in list_indata:
        print ('processing ' + dname)
        list_frames = [f for f in os.listdir(os.path.join(path_indata, dname)) if os.path.isfile(os.path.join(path_indata, dname, f))]
        list_frames.sort()

        # process in a sliding window fashion
        if len(list_frames) >= 2*len_temporal-1:
            path_outdata = os.path.join(path_output, dname)
            if not os.path.isdir(path_outdata):
                os.makedirs(path_outdata)

            snippet = []
            for i in range(len(list_frames)):
                img = cv2.imread(os.path.join(path_indata, dname, list_frames[i]))
                img = cv2.resize(img, (384, 224))
                img = img[...,::-1]
                snippet.append(img)

                if i >= len_temporal-1:
                    clip = transform(snippet)

                    process(model, clip, path_outdata, i)

                    # process first (len_temporal-1) frames
                    if i < 2*len_temporal-2:
                        process(model, torch.flip(clip, [1]), path_outdata, i-len_temporal+1)

                    del snippet[0]

        else:
            print (' more frames are needed')

        list_maps = [f for f in os.listdir(os.path.join(path_output, dname)) if os.path.isfile(os.path.join(path_output, dname, f))]
        list_maps.sort()
        if len(list_maps)>480: #10 seconds videos
            list_maps = list_maps[240:480]
        if len(list_frames)>480: #10 seconds videos
            list_frames = list_frames[240:480]
        video_array = []
        for i in range (len(list_maps)):
            temp_frame = Image.fromarray(cv2.imread(os.path.join(path_indata, dname, list_frames[i]), cv2.IMREAD_COLOR))
            temp_map = Image.fromarray(cv2.imread(os.path.join(path_output, dname, list_maps[i]), cv2.IMREAD_GRAYSCALE))
            red_img = Image.new('RGB', (160, 210), (0, 0, 255))
            temp_frame.paste(red_img, mask=temp_map)
            font_type = ImageFont.truetype('arial.ttf',15)
            draw = ImageDraw.Draw(temp_frame)
            draw.text(xy=(10,160), text=method_used, fill=(255,255,255), font=font_type)
            temp_frame = np.array(temp_frame)
            video_array.append(temp_frame)
        produced_frames.append((video_array, dname))

    return produced_frames

def transform(snippet):
    ''' stack & noralization '''
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)

    return snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)


def process(model, clip, path_outdata, idx):
    ''' process one clip and save the predicted saliency map '''
    with torch.no_grad():
        smap = model(clip.cuda()).cpu().data[0]

    smap = (smap.numpy()*255.).astype(np.int)/255.
    smap = gaussian_filter(smap, sigma=7)
    smap = cv2.resize(smap, (160, 210))
    cv2.imwrite(os.path.join(path_outdata, '%06d.png'%(idx+1)), (smap/np.max(smap)*255.).astype(np.uint8))

    return (smap/np.max(smap)*255.).astype(np.uint8)

if __name__ == '__main__':
    main()

