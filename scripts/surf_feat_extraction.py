#!/usr/bin/env python3

import os
import sys
import threading
import cv2
import numpy as np
import yaml
import pickle
import pdb

# 다운 샘플된 영상 리스트 얻어서 
def get_surf_features_from_video(downsampled_video_filename, surf_feat_video_filename, keyframe_interval, hessian_threshold):
    "Receives filename of downsampled video and of output path for features. Extracts features in the given keyframe_interval. Saves features in pickled file."
    #############
    # TODO
    cap_img = get_keyframes(downsampled_video_filename, keyframe_interval)
    featMat = []
    for img in cap_img:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2, img3 = None, None
        surf = cv2.xfeatures2d.SURF_create()
        surf.setHessianThreshold(hessian_threshold)
        kp, des = surf.detectAndCompute(img, None)
        featMat.append(des)
        
    with open(surf_feat_video_filename +'.pickle', 'wb') as f:
        pickle.dump(data, f)


def get_keyframes(downsampled_video_filename, keyframe_interval):
    "Generator function which returns the next keyframe."

    # Create video capture object
    video_cap = cv2.VideoCapture(downsampled_video_filename)
    frame = 0
    while True:
        frame += 1
        ret, img = video_cap.read()
        if ret is False:
            break
        if frame % keyframe_interval == 0:
            yield img
    video_cap.release()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        exit(1)

    all_video_names = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))

    # Get parameters from config file
    keyframe_interval = my_params.get('keyframe_interval')
    hessian_threshold = my_params.get('hessian_threshold')
    surf_features_folderpath = my_params.get('surf_features')
    downsampled_videos = my_params.get('downsampled_videos')
    #############
    # TODO: Create SURF object
    #############

    # Check if folder for SURF features exists
    if not os.path.exists(surf_features_folderpath):
        os.mkdir(surf_features_folderpath)

    # Loop over all videos (training, val, testing)
    #############
    # TODO: get SURF features for all videos but only from keyframes
    #############    

    fread = open(all_video_names, "r")
    #"video_list -- file containing video names 비디오 파일의 이름을 가지고 있는거 all.videos.lst"
    #"config_file -- yaml filepath containing all parameters: config 파일 찾아서 넣기"
    for line in fread.readlines():
        video_name = line.replace('\n', '')
        downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.ds.mp4')
        surf_feat_video_filename = os.path.join(surf_features_folderpath, video_name + '.surf')

        if not os.path.isfile(downsampled_video_filename):
            continue

        # Get SURF features for one video
        get_surf_features_from_video(downsampled_video_filename,
                                     surf_feat_video_filename, keyframe_interval)
