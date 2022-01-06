"""trt_yolo_cv.py

This script could be used to make object detection video with
TensorRT optimized YOLO engine.

"cv" means "create video"
made by BigJoon (ref. jkjung-avt)
"""


import os
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

#SORT
# python interpreter searchs these subdirectories for modules
import sys
import csv
sys.path.insert(0, './sort')
import skimage
from sort import *

import numpy as np

COLORS = np.random.randint(0, 255, size=(200, 3),
    dtype="uint8")
DETECTION_FRAME_THICKNESS = 1

OBJECTS_ON_FRAME_COUNTER_FONT = cv2.FONT_HERSHEY_SIMPLEX
OBJECTS_ON_FRAME_COUNTER_FONT_SIZE = 0.5

LINE_COLOR = (0, 0, 255)
LINE_THICKNESS = 3
LINE_COUNTER_FONT = cv2.FONT_HERSHEY_DUPLEX
LINE_COUNTER_FONT_SIZE = 2.0
LINE_COUNTER_POSITION = (20, 45)


def parse_args():
    """Parse input arguments."""
    desc = ('Run the TensorRT optimized object detecion model on an input '
            'video and save BBoxed overlaid output as another video.')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-v', '--video', type=str, required=True,
        help='input video file name')
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='output video file name')
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    parser.add_argument(
        '-s', '--save_text', action='store_true',
        help='save the results in csv [False]')
    args = parser.parse_args()
    return args


def loop_and_detect(cap, trt_yolo, conf_th, vis, writer, names):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cap: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
      writer: the VideoWriter object for the output video.
    """
    frame_counter = 0
    counter = 0
    save_txt = True
    line_begin=(0, 250)
    line_end=(650, 250)
    line = [line_begin, line_end]
    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    memory = {} 
    memory_counted = [] 
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh) # {plug into parser}

    with open('traffic_measurement.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_line = 'Vehicle Type, frame, direction'
        csv_writer.writerows([csv_line.split(',')])
    # print('names', names)
    while True:
    # while frame_counter < 300:
        frame_counter+=1
        if frame_counter == 1:
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #raj
        if frame_counter%100 ==0:
            print('frame count', frame_counter)
        ret, frame = cap.read()

        if frame is None:  break
        height = frame.shape[0]
        width = frame.shape[1]
        
        boxes, confs, clss = trt_yolo.detect(frame, conf_th)
        if frame_counter < 10:
            print('boxes, confs, clss', boxes, confs, clss)
        if frame_counter  < 0:
            exit()
        
        #TODO: check the loop in classy sort w.r.t det
        dets_to_sort = np.empty((0,6))
        # for x1,y1,x2,y2,conf,detclass in zip(boxes, confs, clss):
        # boxes = scale_coords(frame.shape[2:], boxes, frame.shape).round()
        for i in range(len(boxes)):
            dets_to_sort = np.vstack((dets_to_sort, np.array([boxes[i][0]/width, boxes[i][1]/height,
             boxes[i][2]/width, boxes[i][3]/height, confs[i], clss[i]])))
            # dets_to_sort = np.vstack((dets_to_sort, np.array([boxes[i][0], boxes[i][1],
            #  boxes[i][2], boxes[i][3], confs[i], clss[i]])))
        
        # print('dets_to_sort', dets_to_sort)
        tracked_dets = sort_tracker.update(dets_to_sort)
        boxes_1 = []
        indexIDs = []
        previous = memory.copy()
        memory = {}

        for j, track in enumerate(tracked_dets):
            boxes_1.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[8]))
            memory[indexIDs[-1]] = boxes_1[-1]
        
        bbox_xyxy = []
        identities = []
        categories = []
    
        if len(tracked_dets)>0:
            # for i in range(len(tracked_dets)):
                # print('tracked_dets', tracked_dets)
                bbox_xyxy = tracked_dets[:,:4]
                # bbox_xyxy = bbox_xyxy.astype(int)
                # print('bbox_xyxy', bbox_xyxy)
                
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]
                # bbox_xyxy[i][0] = int(bbox_xyxy[i][0]*width)
                # bbox_xyxy[i][1] = int(bbox_xyxy[i][1]*height)
                # bbox_xyxy[i][2] = int(bbox_xyxy[i][2]*width)
                # bbox_xyxy[i][3] = int(bbox_xyxy[i][3]*height)
                bbox_xyxy[:, 0] = bbox_xyxy[:, 0]*width
                bbox_xyxy[:, 1] = bbox_xyxy[:, 1]*height
                bbox_xyxy[:, 2] = bbox_xyxy[:, 2]*width
                bbox_xyxy[:, 3] = bbox_xyxy[:, 3]*height
                bbox_xyxy = bbox_xyxy.astype(int)
                # print('bbox_xyxy, identities, categories', bbox_xyxy, identities, categories)
                # print('boxes, confs, clss', boxes, confs, clss)
                frame = draw_boxes(frame, bbox_xyxy, identities, categories, names)
                # frame = vis.draw_bboxes(frame, bbox_xyxy, confs, clss)
        
        print('.', end='', flush=True)

        # Write detections to file. NOTE: Not MOT-compliant format.
        if save_txt and len(tracked_dets) != 0:
            for j, tracked_dets in enumerate(tracked_dets):
                
                bbox_x1 = tracked_dets[0]
                bbox_y1 = tracked_dets[1]
                bbox_x2 = tracked_dets[2]
                bbox_y2 = tracked_dets[3]
                category = tracked_dets[4]
                u_overdot = tracked_dets[5]
                v_overdot = tracked_dets[6]
                s_overdot = tracked_dets[7]
                identity = tracked_dets[8]
                
                #raj begin
                color = [int(c) for c in COLORS[indexIDs[j] % len(COLORS)]]
                #print(len([im0, (bbox_x1, bbox_y1), (bbox_x2-bbox_x1, bbox_y2-bbox_y1), color, DETECTION_FRAME_THICKNESS]))
                #cv2.rectangle(im0, (bbox_x1, bbox_y1), (bbox_x2-bbox_x1, bbox_y2-bbox_y1), color, DETECTION_FRAME_THICKNESS)
                if indexIDs[j] in previous:
                    previous_box = previous[indexIDs[j]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    #p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                    #p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                    p0 = (int((bbox_x1 + bbox_x2)/2 -2), int((bbox_y1 + bbox_y2)/2 -2))
                    p1 = (int((bbox_x1 + bbox_x2)/2 +2), int((bbox_y1 + bbox_y2)/2 +2))
                    cv2.line(frame, p0, p1, color, 3)


                    #directions
                    y = (bbox_y2 + bbox_y1)/2
                    if y2 - y > 0:
                        direction = 'up'
                    else:
                        direction = 'down'

                    if indexIDs[j] not in memory_counted and intersect(p0, p1, line[0], line[1]):
                        counter += 1
                        id_csv = indexIDs[j]
                        label_csv = names[int(category)]#category
                        memory_counted.append(indexIDs[j])
                        
                        with open('traffic_measurement.csv', 'a') as f:
                            csv_line = f'{label_csv} {id_csv}, {frame_counter}, {direction}'
                            print('writing to csv, count_frame', csv_line, video_length,\
                                round(frame_counter/video_length*100, 2), "%")
                            csv_writer = csv.writer(f)
                            csv_writer.writerows([csv_line.split(',')])
                        

                #raj end
                #with open(txt_path, 'a') as f:
                #    f.write(f'{frame_idx},{bbox_x1},{bbox_y1},{bbox_x2},{bbox_y2},{category},{u_overdot},{v_overdot},{s_overdot},{identity}\n')
        cv2.line(frame, line[0], line[1], LINE_COLOR, LINE_THICKNESS)

        cv2.putText(frame, str(counter), LINE_COUNTER_POSITION, LINE_COUNTER_FONT, LINE_COUNTER_FONT_SIZE,
                        LINE_COLOR, 2)
        writer.write(frame)
       

    print('\nDone.')


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit('ERROR: failed to open the input video file!')
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)

    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    loop_and_detect(cap, trt_yolo, conf_th=0.3, vis=vis, writer=writer, names=cls_dict)

    writer.release()
    cap.release()


def intersect(A,B,C,D):
    return dist(A, C) < 5

def dist(A, B):
    return  abs(A[1] - B[1])

def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        
        cat = int(categories[i]) if categories is not None else 0
        
        id = int(identities[i]) if identities is not None else 0
        
        color = compute_color_for_labels(id)
        
        label = f'{names[cat]} | {id}'
        # label = f'{id}'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


if __name__ == '__main__':
    main()
