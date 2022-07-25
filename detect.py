
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
import time
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords

from utils.torch_utils import TracedModel


class Y7Detect:
    def __init__(self, weights):
        """
        params weights: 'yolov7.pt'
        """
        self.weights = weights
        self.model_image_size = 640
        self.conf_threshold = 0.4
        self.iou_threshold = 0.45
        with torch.no_grad():
            self.model, self.device = self.load_model(use_cuda=True)
            self.stride = int(self.model.stride.max())  # model stride
            self.image_size = check_img_size(self.model_image_size, s=self.stride)
            self.half = True if self.device == "cuda:0" else False
            self.trace = False
            if self.trace:
                self.model = TracedModel(self.model, self.device, self.model_image_size)
            if self.half:
                self.model.half()
            self.class_names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.class_names]

    def load_model(self, use_cuda=False):
        if use_cuda:
            use_cuda = torch.cuda.is_available()
            cudnn.benchmark = True

        device = torch.device("cuda:0" if use_cuda else "cpu")
        model = attempt_load(self.weights, map_location=device)
        return model, device

    def preprocess_image(self, image_rgb):
        img = letterbox(image_rgb.copy(), self.image_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def predict(self, image_rgb):
        with torch.no_grad():
            image_rgb_shape = image_rgb.shape
            img = self.preprocess_image(image_rgb)
            pred = self.model(img)[0]
            # apply non_max_suppression
            pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold)
            bboxes = []
            labels = []
            scores = []
            lables_id = []
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_rgb_shape).round()

                for *xyxy, conf, cls in reversed(det):
                    x1 = xyxy[0].cpu().data.numpy()
                    y1 = xyxy[1].cpu().data.numpy()
                    x2 = xyxy[2].cpu().data.numpy()
                    y2 = xyxy[3].cpu().data.numpy()
                    #                        print('[INFO] bbox: ', x1, y1, x2, y2)
                    bboxes.append(list(map(int, [x1, y1, x2, y2])))
                    label = self.class_names[int(cls)]
                    #                        print('[INFO] label: ', label)
                    labels.append(label)
                    lables_id.append(cls.cpu())
                    score = conf.cpu().data.numpy()
                    #                        print('[INFO] score: ', score)
                    scores.append(float(score))

            return bboxes, labels, scores, lables_id

    def run_video(self, source):
        with torch.no_grad():
            webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
                ('rtsp://', 'rtmp://', 'http://', 'https://'))
            if webcam:
                dataset = LoadStreams(source, img_size=self.image_size, stride=self.stride)
            else:
                dataset = LoadImages(source, img_size=self.image_size, stride=self.stride)
            for _, img, im0, vid_cap in dataset:
                start = time.time()
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                pred = self.model(img)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold)

                bboxes = []
                labels = []
                scores = []
                lables_id = []
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        x1 = xyxy[0].cpu().data.numpy()
                        y1 = xyxy[1].cpu().data.numpy()
                        x2 = xyxy[2].cpu().data.numpy()
                        y2 = xyxy[3].cpu().data.numpy()
                        #                        print('[INFO] bbox: ', x1, y1, x2, y2)
                        bboxes.append(list(map(int, [x1, y1, x2, y2])))
                        label = self.class_names[int(cls)]
                        #                        print('[INFO] label: ', label)
                        labels.append(label)
                        lables_id.append(cls.cpu())
                        score = conf.cpu().data.numpy()
                        #                        print('[INFO] score: ', score)
                        scores.append(float(score))
                        # draw_boxes(im0, bboxes[-1], label, round(score*100))
                for idx, box in enumerate(bboxes):
                    icolor = self.class_names.index(labels[idx])
                    draw_boxes(im0, box, labels[idx], round(scores[idx] * 100), self.colors[icolor])
                fps = int(1 / (time.time() - start))
                cv2.putText(im0, 'FPS:' + str(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('video', im0)
                cv2.waitKey(1)

        return bboxes, labels, scores, lables_id


def draw_boxes(image, boxes, label, scores=None, color=None):
    if color is None:
        color = (0, 255, 0)
    xmin, ymin, xmax, ymax = boxes
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(image, label + "-{:d}".format(scores), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image, boxes


if __name__ == '__main__':
    path_models = '/home/duyngu/Desktop/detect_yolov7/weights/yolov7.pt'
    url = '/home/duyngu/Downloads/video_test/TownCentre.mp4'
    y7_model = Y7Detect(weights=path_models)
    y7_model.run_video(url)

