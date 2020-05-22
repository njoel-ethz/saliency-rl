import os
import numpy as np
import cv2
import torch
from model import TASED_v2
from scipy.ndimage.filters import gaussian_filter
from threading import Thread

from mss import mss

# From https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
# From https://stackoverflow.com/a/54246290
class VideoStream:
    def __init__(self, src=0):
        self.sct = mss()

        mon = self.sct.monitors[src]
        self.monitor = {
            "top": mon["top"] + 300,  # 100px from the top
            "left": mon["left"] + 200,  # 100px from the left
            "width": 640,
            "height": 480,
            "mon": src,
        }

        sct_img = self.sct.grab(self.monitor)
        self.frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_RGB2BGR)
        #print(self.frame.shape)
        # cv2.imshow('test', self.frame)
        #
        # while True:
        #     if cv2.waitKey(25) & 0xFF == ord('q'):
        #         cv2.destroyAllWindows()
        #         break

        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            sct_img = self.sct.grab(self.monitor)
            self.frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


class OnlineDetector:
    def __init__(self):
        self.len_temporal = 32
        self.file_weight = os.path.join('Atari_dataset', 'weights_trained_on_3000.pt')#'TASED_updated.pt'

        self.current_saliency_map = None
        self.images = []
        self.is_calculating = False

        self.model = TASED_v2()

        # load the weight file and copy the parameters
        if os.path.isfile(self.file_weight):
            print('loading weight file')
            weight_dict = torch.load(self.file_weight)
            model_dict = self.model.state_dict()
            for name, param in weight_dict.items():
                if 'module' in name:
                    name = '.'.join(name.split('.')[1:])
                if name in model_dict:
                    if param.size() == model_dict[name].size():
                        model_dict[name].copy_(param)
                    else:
                        print(' size? ' + name, param.size(), model_dict[name].size())
                else:
                    print(' name? ' + name)

            print(' loaded')
        else:
            print('weight file?')

        self.model = self.model.cuda()
        torch.backends.cudnn.benchmark = False
        self.model.eval()

    def transform(self, snippet):
        ''' stack & noralization '''
        snippet = np.concatenate(snippet, axis=-1)
        snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
        snippet = snippet.mul_(2.).sub_(255).div(255)

        return snippet.view(1, -1, 3, snippet.size(1), snippet.size(2)).permute(0, 2, 1, 3, 4)

    def process_async(self):
        if not self.is_calculating:
            images_resized = []

            for frame in self.images:
                print(frame.shape)
                img = cv2.resize(frame, (384, 224))
                images_resized.append(img)

            self.is_calculating = True
            Thread(target=self.process, args=([images_resized])).start()

    def process_sync(self):
        if not self.is_calculating:
            images_resized = []

            for frame in self.images:
                img = cv2.resize(frame, (384, 224))
                images_resized.append(img)

            self.is_calculating = True
            self.process(images_resized)

    def process(self, images_resized):
        clip = self.transform(images_resized)
        self.current_saliency_map = self.process_clip(clip)
        self.is_calculating = False
        # self.images = []

    def process_clip(self, clip):
        with torch.no_grad():
            smap = self.model(clip.cuda()).cpu().data[0]

        smap = (smap.numpy() * 255.).astype(np.int) / 255.
        smap = gaussian_filter(smap, sigma=7)
        grayscale = (smap / np.max(smap) * 255.).astype(np.uint8)

        color = np.zeros((224, 384, 3), dtype=np.uint8)
        color[:, :, 2] = grayscale
        resized_color = cv2.resize(color, (640, 480))

        return resized_color

    def run(self):
        cap = VideoStream(0)
        cap.start()

        # num_frames = 0

        # start = time.time()

        while (True):
            frame = cap.read()
            # num_frames = num_frames + 1

            if not self.is_calculating:
                #print(len(self.images))
                self.images.append(frame)
                self.images.append(frame)

                if len(self.images) > self.len_temporal:
                    del(self.images[0])
                    del(self.images[0])

                if len(self.images) == self.len_temporal:
                    self.process_async()

            if self.current_saliency_map is not None:
                frame = cv2.addWeighted(frame, .5, self.current_saliency_map, .5, 0)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Time elapsed
            # seconds = time.time() - start
            # print("Time taken : {0} seconds".format(seconds))

            # Calculate frames per second
            # fps = num_frames / seconds;
            # print("fps : {0}".format(fps))

        # When everything done, release the capture
        cap.stop()
        # cap.release()
        cv2.destroyAllWindows()


def main():
    detector = OnlineDetector()
    detector.run()


if __name__ == '__main__':
    main()
