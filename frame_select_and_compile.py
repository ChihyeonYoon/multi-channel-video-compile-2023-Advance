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

def frame_to_seconds(frame_number, fps):
    """
    Convert a frame number to seconds.

    :param frame_number: The frame number you want to convert.
    :param fps: The frames per second (FPS) of the video.
    :return: Time in seconds.
    """
    seconds = frame_number / fps
    return seconds

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
    parser.add_argument('--switching_interval', type=int, default=15,
                    help='switching interval') 
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

        camera1_prop = camera1_info['speak_prop']
        camera2_prop = camera2_info['speak_prop']

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
    for i in range(0, len(selected_channels), args.switching_interval):
        batch = selected_channels[i:i+args.switching_interval]
        mcv = Counter(batch).most_common(1)[0][0]
        adjusted_channels = adjusted_channels+([mcv]*len(batch))
    
    video_ch0 = cv2.VideoCapture(args.video_ch0_path) # wide channel
    video_ch1 = cv2.VideoCapture(args.video_ch1_path) # 1st channel
    video_ch2 = cv2.VideoCapture(args.video_ch2_path) # 2nd channel
    audio = AudioFileClip(args.video_ch1_path)
    fps = video_ch1.get(cv2.CAP_PROP_FPS)

    min_total_frame = min(video_ch1.get(cv2.CAP_PROP_FRAME_COUNT), video_ch2.get(cv2.CAP_PROP_FRAME_COUNT), video_ch0.get(cv2.CAP_PROP_FRAME_COUNT))
    print('min_total_frame: ', min_total_frame)
    
    # start_frame = frames[0]['frame_n']
    # end_frame = frames[-1]['frame_n']
    start_frame = int(args.start_time*video_ch0.get(cv2.CAP_PROP_FPS))+1 if args.start_time is not None else 0
    end_frame = int(args.end_time*video_ch0.get(cv2.CAP_PROP_FPS)) if args.end_time is not None else min_total_frame

    # start_time = frame_number_to_hhmmss(start_frame)
    # end_time = frame_number_to_hhmmss(end_frame)
    start_time = frame_to_seconds(start_frame, video_ch0.get(cv2.CAP_PROP_FPS))
    end_time = frame_to_seconds(end_frame, video_ch0.get(cv2.CAP_PROP_FPS))
    print('start_time: ', start_time)
    print('end_time: ', end_time)

    resolution = (1920, 1080)

    tmp_path = 'tmp.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(tmp_path, fourcc, 30.0, resolution)

    r_start_time = time.time()
    while (video_ch1.isOpened and video_ch2.isOpened and video_ch0.isOpened):

        ret_ch0, frame_ch0 = video_ch0.read()
        ret_ch1, frame_ch1 = video_ch1.read()
        ret_ch2, frame_ch2 = video_ch2.read()
        
        frame_n = int(video_ch0.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_n < start_frame:
            continue
        elif frame_n >= end_frame:
            break


        if str(frame_n) in result_dict.keys():
            selected_channel = adjusted_channels[frame_n - start_frame]

            # if frame_n % 100 == 0 or frame_n == start_frame or frame_n == end_frame:
            print("#Frame: {}/{} -------------------------".format(frame_n, end_frame))
            print('\tselected_channel: ', selected_channel)

            if selected_channel == 0:
                cv2.putText(img=frame_ch0, text="CAM #W", org= (resolution[0]-150, resolution[-1]-50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                output_video.write(frame_ch0)
            elif selected_channel == 1:
                cv2.putText(img=frame_ch1, text="CAM #1", org=(resolution[0]-150, resolution[-1]-50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                output_video.write(frame_ch1)
            elif selected_channel == 2:
                cv2.putText(img=frame_ch2, text="CAM #2", org=(resolution[0]-150, resolution[-1]-50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                output_video.write(frame_ch2)        
        else:
            continue
    
        if frame_n >= min_total_frame or frame_n >= end_frame:
            break
    output_video.release()
    
    time.sleep(3)

    output_video_with_audio = VideoFileClip(tmp_path)
    output_video_with_audio = output_video_with_audio.set_fps(fps)
    os.remove(tmp_path)
    time.sleep(3)
    print(start_time, end_time)


    # Fucking Error Occur
    '''
    # Extract the audio subclip without setting FPS (since it's audio)
    audio = audio.subclip(t_start=start_time, t_end=end_time)

    # Set the audio to the video
    output_video_with_audio = output_video_with_audio.set_audio(audio)

    # Ensure the FPS is set on the video clip if not already
    if output_video_with_audio.fps is None:
        output_video_with_audio = output_video_with_audio.set_fps(fps)  # Adjust FPS as needed

    # Generate the output video file name
    # output_video_file = args.output_video_path.replace('.mp4', '_'+time.strftime('%y%m%d_%X')+'.mp4')

    # Write the final video with the correct codec and audio codec
    output_video_with_audio.write_videofile(args.output_video_path, codec='libx264', audio_codec='aac')

    # Print the time taken for the process
    print('time: {}'.format(time.time() - r_start_time))
    '''