import socket
from PIL import Image
import io

import cv2
import numpy as np

import os
import cv2
import time
import argparse
import numpy as np
from distutils.util import strtobool

from YOLOv3 import YOLOv3
from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

import torch

from resnets.get_active_speaker_model import get_resnet_model

class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.yolo3 = YOLOv3(args.yolo_cfg, args.yolo_weights, args.yolo_names, is_xywh=True, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, use_cuda=use_cuda)
        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)
        self.class_names = self.yolo3.class_names

        self.resnet3d_model, self.resnet_spatial_transform = get_resnet_model()

    def score_is_speaking(self, clip):
        if self.resnet_spatial_transform is not None:
            self.resnet_spatial_transform.randomize_parameters()
            clip = [self.resnet_spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3).unsqueeze(0)
        clip.cuda()

        outputs = self.resnet3d_model(clip)

        talk = outputs[0, 47 - 1]
        chew = outputs[0, 4 - 1]
        smile = outputs[0, 40 - 1]
        smoke = outputs[0, 41 - 1]

        return int((talk + chew - smile - smoke)*100)

    def parse_header(self, header_data):
        img_size = int.from_bytes(header_data, byteorder='little')
        return img_size

    def iterate_tracker(self, img):
        start = time.time()
        ori_im = img.copy()
        im = ori_im.copy()
        bbox_xcycwh, cls_conf, cls_ids = self.yolo3(im)
        if bbox_xcycwh is not None:
            # select class person
            mask = cls_ids == 0

            bbox_xcycwh = bbox_xcycwh[mask]
            bbox_xcycwh[:, 3:] *= 1.2

            cls_conf = cls_conf[mask]
            outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_bboxes(ori_im, bbox_xyxy, identities)
                return_json = dict()
                return_json["tracks"] = []
                for i,box in enumerate(bbox_xyxy):
                    #box = bbox_xyxy[i]
                    point_x_left = box[0]
                    point_x_right = box[2]
                    if (point_x_left + point_x_right)/2 > 240.0:
                        point_x = point_x_left
                    else:
                        point_x = point_x_right
                    point_y = box[1]
                    ret_x = int(point_x / 480.0 * 1920)
                    ret_y = int(point_y / 270.0 * 1080)
                    return_json["curr_speaker_id"] = 0
                    return_json["curr_speaker_score"] = 20
                    track_data = dict()
                    track_data["x"] = ret_x
                    track_data["y"] = ret_y
                    track_data["speaker_id"] = identities[i]-1

                    return_json["tracks"].append(track_data)
                    #return_json["tracks"].append([identities[i]-1, ret_x, ret_y])
                    #return_msg = '{},{}'.format(identities[i]-1,ret_x)
                    #return_msgs.append('{},{}'.format(identities[i]-1,ret_x))
                return_msg = str(return_json).replace("\'", "\"")
            else:
                return_msg = 'None'
        else:
            return_msg = 'None'

        end = time.time()
        print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))

        if self.args.display:
            cv2.imshow("test", ori_im)
            cv2.waitKey(1)

        #if self.args.save_path:
        #    self.output.write(ori_im)

        return return_msg


    def run_server(self):
        while True:
            try:
                TCP_IP = 'bowaldwindows.student.rit.edu'
                TCP_PORT = 3000
                BUFFER_SIZE = 4096

                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind((TCP_IP, TCP_PORT))
                s.listen(1)

                conn, addr = s.accept()
                print('Connection address:', addr)
                constructing_image = False
                bytes_left = 0  # the number of bytes left to reconstruct the image with
                curr_img_bytes = None  # the image byte array, while still in the process of reconstructing
                while True:
                    data = conn.recv(BUFFER_SIZE)
                    if not data:
                        break
                    if len(data) == 4:  # ONLY the header packet can have size 4, BY DESIGN ON THE UNITY SIDE.
                        img_size = self.parse_header(data)
                        bytes_left = img_size
                        curr_img_bytes = None
                        constructing_image = True
                    else:  # the received data is part of the reconstructed image. Add onto curr_img_bytes.
                        if constructing_image:
                            if curr_img_bytes is None:  # If new image...
                                assert len(data) <= bytes_left
                                curr_img_bytes = data  # initialize curr_img_bytes with the received data.
                                bytes_left -= len(data)  # decrement the number of bytes remaining in the image appropriately.
                                continue
                            else:
                                if len(
                                        data) > bytes_left:  # If the current packet exceeds the number of bytes left in the image we are reconstructing...
                                    if len(data) - bytes_left > 1:  # If it isn't a 4-byte array padded to 5-byte array by Unity...
                                        constructing_image = False  # don't construct images from bad data
                                    # If it IS a 4-byte array padded to a 5-byte array by Unity...
                                    curr_img_bytes += data[:-1]  # Simply ignore the last byte and keep adding onto curr_img_bytes.
                                    bytes_left -= (len(data) - 1)
                                else:  # If the current packet doesn't exceed the number of bytes left...
                                    curr_img_bytes += data  # keep adding onto curr_img_bytes.
                                    bytes_left -= len(data)
                                if bytes_left == 0:  # Image reconstruction is done.
                                    #assert len(curr_img_bytes) != img_size
                                    img = Image.open(io.BytesIO(curr_img_bytes))
                                    img = np.array(img)[:, :, ::-1]
                                    print(img.shape)
                                    cv2.imshow('recv', img)
                                    return_msg = self.iterate_tracker(img)
                                    conn.send(return_msg.encode())  # Send the string message for the client to receive.
                                    print('sent data! ' + return_msg)
            except ConnectionResetError and ConnectionAbortedError and ConnectionError:
                continue
            self.__init__(self.args)

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--yolo_cfg", type=str, default="YOLOv3/cfg/yolo_v3.cfg")
    parser.add_argument("--yolo_weights", type=str, default="YOLOv3/yolov3.weights")
    parser.add_argument("--yolo_names", type=str, default="YOLOv3/cfg/coco.names")
    parser.add_argument("--conf_thresh", type=float, default=0.75)
    parser.add_argument("--nms_thresh", type=float, default=0.2)
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.5)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.avi")
    parser.add_argument("--use_cuda", type=str, default="True")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    det = Detector(args)
    det.run_server()