video_ch1_path='/NasData/home/ych/2024_Multicam/materials/camera1_synced.mp4'
video_ch2_path='/NasData/home/ych/2024_Multicam/materials/camera2_synced.mp4'
video_ch3_path='/NasData/home/ych/2024_Multicam/materials/camera3_synced.mp4'
classification_model='swin_v2_b'
weights='/NasData/home/ych/2024_Multicam/checkpoints/run0621_0339/snapshot_swin_v2_b_2_0.9563032640482664.pth'
sample_name='./compiled_samples/run240902'

json_file="${sample_name}.json"

python multi_video_infer.py --video1 $video_ch2_path \
                            --video2 $video_ch3_path \
                            --classification_model $classification_model \
                            --weights $weights \
                            --output $json_file

sleep 3s

python frame_select_and_compile.py --videw_ch0_path $video_ch0_path \
                                --video_ch1_path $video_ch1_path \
                                --video_ch2_path $video_ch2_path \
                                --inference_result_dict_path $json_file \
                                --out_video_path "${sample_name}.mp4" \