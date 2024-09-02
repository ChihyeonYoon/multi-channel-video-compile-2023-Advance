import json
import cv2
import time
import os
from moviepy.editor import AudioFileClip, VideoFileClip
from collections import Counter
import argparse

def frame_number_to_hhmmss(frame_number, frames_per_second=30):
    total_seconds = frame_number / frames_per_second
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_ch0_path', type=str, default='./materials/camera1_synced.mp4',
                        help='video ch0 path')
    parser.add_argument('--video_ch1_path', type=str, default='./materials/camera2_synced.mp4',
                        help='video ch1 path')
    parser.add_argument('--video_ch2_path', type=str, default='./materials/camera3_synced.mp4',
                        help='video ch2 path')
    
    parser.add_argument('--start_time', type=int, default=None)
    parser.add_argument('--end_time', type=int, default=None)
    
    parser.add_argument('--inference_result_dict_path', type=str, default='/NasData/home/ych/2024_Multicam/result.json',
                        help='inference output path')
    parser.add_argument('--output_video_path', type=str, default='./compiled_samples/sample2.mp4',
                        help='output video path')
    args = parser.parse_args()

    """
    speak_prop = ['silent', 'utter']

    result_dict = {
        "frame_n": [
            {
                'camera': 1,
                'state': 1 or 0,
                'speak_prop': []
                },
            {
                'camera': 2,
                'state': 1 or 0,
                'speak_prop': []
                },
            ],
        ...
    }
    """
    result_dict = json.load(open(args.inference_result_dict_path))
    frames = result_dict.keys()
    frames = list(map(int, frames))
    selected_channels = []
    privious_selected_channel = 0 

    for frame_n, frame_info in result_dict.items():
        camera1_info = frame_info[0]
        camera2_info = frame_info[1]

        camera1_prop = camera1_info['speak_prop'][-1]
        camera2_prop = camera2_info['speak_prop'][-1]

        # 알고리즘 생각하고 수정 -> 0번, 1번, 2번 카메라 중에서 선택
        if camera1_prop > 0.65 and camera2_prop > 0.65: # 두 출연자 모두 말하는 확률이 높은 경우
            selected_channel = 0
        elif camera1_prop > 0.65: # 첫 번째 출연자가 말하는 확률이 높은 경우
            selected_channel = 1
        elif camera2_prop > 0.65: # 두 번째 출연자가 말하는 확률이 높은 경우
            selected_channel = 2 
        else: # 두 출연자 모두 말하는 확률이 낮은 경우
            selected_channel = 0

        selected_channels.append(selected_channel)

    # ================== selected channel adjusting ==================
    adjusted_channels = []
    for i in range(0, len(selected_channels), 30):
        batch = selected_channels[i:i+30]
        mcv = Counter(batch).most_common(1)[0][0]
        adjusted_channels = adjusted_channels+([mcv]*len(batch))

    video_ch0 = cv2.VideoCapture(args.video_ch0_path) # wide channel
    video_ch1 = cv2.VideoCapture(args.video_ch1_path) # 1st channel
    video_ch2 = cv2.VideoCapture(args.video_ch2_path) # 2nd channel
    audio = AudioFileClip(args.video_ch0_path)

    start_frame = args.start_time * 30 if args.start_time is not None else 0
    end_frame = args.end_time * 30 if args.end_time is not None else int(video_ch0.get(cv2.CAP_PROP_FRAME_COUNT))

    # start_frame = frames[0] if args.start_frame is None else args.start_frame
    # end_frame = frames[-1] if args.end_frame is None else args.end_frame

    start_time = frame_number_to_hhmmss(start_frame)
    end_time = frame_number_to_hhmmss(end_frame)

    tmp_path = 'tmp.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(tmp_path, fourcc, 30.0, (1920, 1080))

    run_start_time = time.time()
    min_total_frame = min(video_ch1.get(cv2.CAP_PROP_FRAME_COUNT), video_ch2.get(cv2.CAP_PROP_FRAME_COUNT), video_ch0.get(cv2.CAP_PROP_FRAME_COUNT))
    print('min_total_frame: ', min_total_frame)
    while (video_ch1.isOpened and video_ch2.isOpened and video_ch0.isOpened):

        ret_ch0, frame_ch0 = video_ch0.read()
        ret_ch1, frame_ch1 = video_ch1.read()
        ret_ch2, frame_ch2 = video_ch2.read()
        
        frame_n = int(video_ch0.get(cv2.CAP_PROP_POS_FRAMES))

        if start_frame <= frame_n <= end_frame:
            print("#Frame: {}/{} -------------------------".format(frame_n, end_frame))
            
            selected_channel = adjusted_channels[frame_n-1]
            print('\tselected_channel: ', selected_channel)

            if selected_channel == 0:
                output_video.write(frame_ch0)
            elif selected_channel == 1:
                output_video.write(frame_ch1)
            elif selected_channel == 2:
                output_video.write(frame_ch2)
        # else:
        #     continue

        if frame_n >= min_total_frame or frame_n >= end_frame:
            break
    output_video.release()

    time.sleep(3)
    output_video_with_audio = VideoFileClip(tmp_path)
    audio = audio.subclip(start_time, end_time)
    output_video_with_audio = output_video_with_audio.set_audio(audio)
    output_video_with_audio.write_videofile(args.output_video_path, codec='libx264', audio_codec='aac')
    os.remove(tmp_path)

    print(f'output video saved at {args.output_video_path}')
    print('run time: ', time.time() - run_start_time)



    

