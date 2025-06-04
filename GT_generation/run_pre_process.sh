#!/bin/bash

for clip_name in {1..58}

do

    reset
    data_path_header="../KRadar"
    save_path_header="../Dense_GT"
    mkdir -p $save_path_header/$clip_name
    cp -r $data_path_header/$clip_name/time_info $save_path_header/$clip_name # From the K-Radar dataset


    echo "1. dyn-sta split"
    python ./1_dynamic_static_split.py $clip_name $data_path_header $save_path_header


    echo "2. ground removal"
    python ./2_static_ground_removal.py $clip_name $save_path_header


    echo "3. lidar odometry"
    # Apply your own lidar odometry algorithm here.


    echo "4. static enhance"
    # Apply your own static enhance algorithm here.


    echo "5. dynamic enhance"
    python ./5_dynamic_pc_enhance.py $clip_name $data_path_header $save_path_header


    echo "6. dyn-sta stitching"
    python ./6_dynamic_static_stitching.py $clip_name $data_path_header $save_path_header


    echo "7. my lidar filter"
    python ./7_lidar_filter_projection.py $clip_name $data_path_header $save_path_header

done