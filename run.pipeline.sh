#!/bin/bash

# This script performs a complete Media Event Detection pipeline (MED) using video features:
# a) preprocessing of videos, b) feature representation,
# c) computation of MAP scores

# You can pass arguments to this bash script defining which one of the steps you want to perform.
# This helps you to avoid rewriting the bash script whenever there are
# intermediate steps that you don't want to repeat.

#execute: bash /home/ubuntu/11775-hws/hw2_code/run.pipeline.sh -p true -f true -m true -y filepath

# Reading of all arguments:
while getopts p:f:m:y: option      # p:f:m:y: is the optstring here
   do
   case "${option}"
   in
   p) PREPROCESSING=${OPTARG};;       # boolean true or false
   f) FEATURE_REPRESENTATION=${OPTARG};;  # boolean
   m) MAP=${OPTARG};;                 # boolean
   y) YAML=$OPTARG;;                  # path to yaml file containing parameters for feature extraction
   esac
   done

export PATH=~/anaconda3/bin:$PATH

if [ "$PREPROCESSING" = true ] ; then

    echo "#####################################"
    echo "#         PREPROCESSING             #"
    echo "#####################################"

################# ORIGINAL ######################
#     # steps only needed once
#     video_path=~/video  # path to the directory containing all the videos.
#     mkdir -p list downsampled_videos surf cnn kmeans  # create folders to save features
#     awk '{print $1}' ../hw1_code/list/train > list/train.video  # save only video names in one file (keeping first column)
#     awk '{print $1}' ../hw1_code/list/val > list/val.video
#     cat list/train.video list/val.video list/test.video > list/all.video    #save all video names in one file
#################################################
    # steps only needed once
    #video_path=/home/ubuntu/11775-hws/video  # path to the directory containing all the videos.
    #mkdir -p list downsampled_videos surf cnn kmeans  # create folders to save features
    #awk '{print $1}' /home/ubuntu/11775-hws/hw2_code/list/train > list/train.video  # save only video names in one file (keeping first column)
    #awk '{print $1}' /home/ubuntu/11775-hws/hw2_code/list/val > list/val.video
    #cat list/train.video list/val.video /home/ubuntu/11775-hws/hw2_code/list/test.video > list/all.video    #save all video names in one file
#################################################
    #downsampling_frame_len=60
    #downsampling_frame_rate=15

    # 1. Downsample videos into shorter clips with lower frame rates.
    # TODO: Make this more efficient through multi-threading f.ex.
#     start=`date +%s`
#     for line in $(cat "list/all.video"); do
#         ffmpeg -y -ss 0 -i $video_path/${line}.mp4 -strict experimental -t $downsampling_frame_len -r $downsampling_frame_rate downsampled_videos/$line.ds.mp4
#     done
#    end=`date +%s`
#     runtime=$((end-start))
#     echo "Downsampling took: $runtime" #28417 sec around 8h without parallelization
#################################################
    # 2. TODO: Extract SURF features over keyframes of downsampled videos (0th, 5th, 10th frame, ...)
    # python3 surf_feat_extraction.py list/all.video config.yaml
    
#################################################

    # 3. TODO: Extract CNN features from keyframes of downsampled videos
   

fi

if [ "$FEATURE_REPRESENTATION" = true ] ; then

    echo "#####################################"
    echo "#  SURF FEATURE REPRESENTATION      #"
    echo "#####################################"

    # 1. TODO: Train kmeans to obtain clusters for SURF features
    #python3 scripts/train_kmeans.py $surf_file '/surf/' $cluster_num 40 $output_file 'kmeans_40.sav'
    
    # 2. TODO: Create kmeans representation for SURF features
    #python3 scripts/create_kmeans.py $kmeans_model 'kmeans_40.sav' $cluster_num 40 $file_list '/home/ubuntu/11775-hws/all_video.lst'


fi

if [ "$MAP" = true ] ; then

    echo "#######################################"
    echo "# MED with SURF Features: MAP results #"
    echo "#######################################"

    # 1. TODO: Train SVM with OVR using only videos in training set.
    python3 scripts/train_svm.py $feat_dir '/home/ubuntu/11775-hws/bow_40/' $output_file '/home/ubuntu/11775-hws/hw2_code/svm_model_40.sav' $list_file_path '/home/ubuntu/11775-hws/hw2_code/list/train'
    
    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.
     python3 scripts/test_svm.py $model_file '/home/ubuntu/11775-hws/hw2_code/svm_model_40.sav' $feat_dir '/home/ubuntu/11775-hws/bow_40/' $file_list_path '/home/ubuntu/11775-hws/hw2_code/list/val' $output_file '/home/ubuntu/11775-hws/val_pred.txt'
   
    map_path=/home/ubuntu/tools/mAP
    export PATH=$map_path:$PATH
    for event in P001 P002 P003; do
        ap list/${event}_val_label /home/ubuntu/11775-hws/val_pred_${event}_val_label
    done
    
   
   # 3. TODO: Train SVM with OVR using videos in training and validation set.
    python3 scripts/train_svm.py $feat_dir '/home/ubuntu/11775-hws/bow_40/' $output_file '/home/ubuntu/11775-hws/hw2_code/svm_model_40_train_val.sav' $list_file_path '/home/ubuntu/11775-hws/hw2_code/list/train_val'   
   

   # 4. TODO: Test SVM with test set saving scores for submission
   # 4.1 PREDICTION with SVM trained with TRAIN DATA
#    python3 scripts/test_svm.py $model_file '/home/ubuntu/11775-hws/hw2_code/svm_model_40.sav' $feat_dir '/home/ubuntu/11775-hws/bow_40/' $file_list_path '/home/ubuntu/11775-hws/hw2_code/list/test.video' $output_file '/home/ubuntu/11775-hws/test_pred_trn.txt'
    # 4.2 PREDICTION with SVM trained with TRAIN+VALIDATION DATA
    python3 scripts/test_svm.py $model_file '/home/ubuntu/11775-hws/hw2_code/svm_model_40_train_val.sav' $feat_dir '/home/ubuntu/11775-hws/bow_40/' $file_list_path '/home/ubuntu/11775-hws/hw2_code/list/test.video' $output_file '/home/ubuntu/11775-hws/test_pred_trnval.txt'

fi