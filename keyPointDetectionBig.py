import torch
import cv2
import numpy as np
import argparse
import sys

from rektnet.kpdetUtilsBig import prep_image
from rektnet.keypoint_net_big import KeypointNetBig

@torch.no_grad()
class KeyPointRegressionBig():
 
    def __init__(self, weights):
        self.weights = weights
        self.model = KeypointNetBig()
        self.model.load_state_dict(torch.load(self.weights).get('model'))
        self.model.eval()
        self.model.to(0)
       

    def vis_tensor_and_save_result(self, h, w, tensor_output, x ,y): #  Возвращает пересчитанный лист с 11 точками для текущего конуса
        a = list()
        i = 0
        for pt in np.array(tensor_output):
            a.append((y + int(pt[0] * w), int(x + pt[1] * h)))
            i += 1

        return a


    def detect(self, img, img_size): #  Детект прогоняет кроп через сетку, возвращает тензор точек в относительных координатах

        image_size = (img_size, img_size)
        image = img.numpy()
        image = prep_image(image=image,target_image_size=image_size)
        image = (image.transpose((2, 0, 1)) / 255.0)[np.newaxis, :]
        image = torch.from_numpy(image).type('torch.cuda.FloatTensor')

        output = self.model(image)
        out = np.empty(shape=(0, output[0][0].shape[2]))
        for o in output[0][0]:
            chan = np.array(o.cpu().data)
            cmin = chan.min()
            cmax = chan.max()
            chan -= cmin
            chan /= cmax - cmin
            out = np.concatenate((out, chan), axis=0)
    
        return output[1][0].cpu().data


    def cropper_big(self, image_path, detections): #  Кропает конусы, запускает для каждого кропа детект и пересчет возвращенных координат, в конце возвращает лист с точками в абсолютных координатах.
    
        h, w = image_path.shape[:2]
        image = torch.from_numpy(image_path).type('torch.FloatTensor')
        result = list()

        for line in detections:
            a = line
            x1 = int(a[2] * h - a[4] * h * 0.5)
            x2 = int(a[2] * h + a[4] * h * 0.5)
            y1 = int(a[1] * w - a[3] * w * 0.5)
            y2 = int(a[1] * w + a[3] * w * 0.5)
            img = image[x1:x2, y1:y2]
            wc = y2 - y1
            hc = x2 - x1

            a = self.detect(img=img, img_size=80)
            b = self.vis_tensor_and_save_result(hc, wc, a, x1, y1)
            result.append(b)

        return result