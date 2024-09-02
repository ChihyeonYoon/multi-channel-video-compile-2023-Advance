video_ch1_path='/NasData/home/ych/Multicam_materials/opentalk/camera1_synced.mp4'
video_ch2_path='/NasData/home/ych/Multicam_materials/opentalk/camera2_synced.mp4'
video_ch3_path='/NasData/home/ych/Multicam_materials/opentalk/camera3_synced.mp4'
classification_model='swin_v2_b'
weights='/NasData/home/ych/multi-channel-video-compile-2023-Advance/checkpoints/run0621_0339/snapshot_swin_v2_b_1_0.9550209507386264.pth'
sample_name='./compiled_sample/run240903'

json_file="${sample_name}.json"

if [ -f $json_file ]; then
    echo "File $json_file exists. skipping inference"
else
    echo "File $json_file does not exist. running inference"
    python multi_video_infer.py --video1 $video_ch2_path \
                                --video2 $video_ch3_path \
                                --classification_model $classification_model \
                                --weights $weights \
                                --output $json_file

sleep 3s
fi

if [ -f "$json_file" ]; then
    echo "File $json_file exists. compiling video"
    python frame_select_and_compile.py --video_ch0_path $video_ch1_path \
                                    --video_ch1_path $video_ch2_path \
                                    --video_ch2_path $video_ch3_path \
                                    --inference_result_dict_path $json_file \
                                    --output_video_path "${sample_name}.mp4" \
                                    # --start_time 0 \
                                    # --end_time 1000
else
    echo "File $json_file does not exist. Exiting"
fi
