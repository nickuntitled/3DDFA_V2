# coding: utf-8
# Cleardusk coding, for me, nick untitled, modify as library like FAN on Adrian bullet.
__author__ = 'nickuntitled'

import argparse, cv2, yaml, os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '4'

from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA_ONNX import TDDFA_ONNX

# Function get_args
# ======================================
# This function uses for getting arguments.
def get_args():
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/trump_hillary.jpg')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    args = parser.parse_args()
    return args

# Function detect_landmark
# ======================================
# This function uses for detection of facial landmark with specific bounding boxes and model.
def detect_landmark(img, boxes, 
                    tddfa, mode = 'cpu', 
                    opt = '2d_sparse', show_flag = False):
    param_lst, roi_box_lst = tddfa(img, boxes)
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)

    landmark = []
    for landm in ver_lst:
        landm_x = landm[0, :].reshape((-1, 1))
        landm_y = landm[1, :].reshape((-1, 1))
        landmark.append(np.concatenate([landm_x, landm_y], axis = 1))
    return landmark, roi_box_lst

# Function load_model
# ======================================
# This function uses for loading model.
def load_model(config_path):
    cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)
    tddfa = TDDFA_ONNX(**cfg)
    return tddfa

if __name__ == '__main__':
    args = get_args()

    # Init FaceBoxes and TDDFA
    tddfa = load_model(args.config)
    face_boxes = FaceBoxes_ONNX()

    # Face detection
    img = cv2.imread(args.img_fp)
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        print('No face detected.')
        exit(1)

    # Detection
    landmark, bbox = detect_landmark(img, boxes, tddfa)

    print("Bounding Box")
    print(bbox)

    print("Landmark")
    print(landmark)