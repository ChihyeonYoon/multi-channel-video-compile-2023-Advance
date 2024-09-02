import os
import cv2
import numpy as np
import math
import multiprocessing as mp
import time
import mediapipe
import traceback
import random
import torch
from PIL import Image
from argparse import ArgumentParser
import json


from model_zoo import get_model

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def euclidean_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.sqrt(np.sum((point1 - point2)**2))

# 회전 행렬 함수
def create_rotation_matrix(yaw, pitch, roll):
    # Yaw (좌우 회전)
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Pitch (상하 회전)
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Roll (기울기 회전)
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # 최종 회전 행렬
    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
    return R

# 얼굴의 기울기를 보정하는 함수 (yaw, pitch, roll)
def correct_angle(landmarks, point, left_eye_index, right_eye_index):
    left_eye = np.array(landmarks[left_eye_index])
    right_eye = np.array(landmarks[right_eye_index])
    point = np.array(point)
    
    eye_center = (left_eye + right_eye) / 2.0
    
    yaw = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    
    pitch = np.arctan2(right_eye[2] - left_eye[2], right_eye[0] - left_eye[0])
    
    roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[2] - left_eye[2])
    
    # 3D 회전 행렬 생성
    rotation_matrix = create_rotation_matrix(yaw, pitch, roll)
    
    # 포인트 보정
    corrected_point = np.dot(rotation_matrix, point - eye_center) + eye_center
    return corrected_point

def is_mouth_open(landmarks, threshold=0.05):
    upper_lip_top_index = 13 
    lower_lip_bottom_index = 14 
    mouth_left_corner_index = 61 
    mouth_right_corner_index = 291  
    left_eye_index = 130 # 130
    right_eye_index = 263 # 263
    
    upper_lip_top = landmarks[upper_lip_top_index]
    lower_lip_bottom = landmarks[lower_lip_bottom_index]
    mouth_left_corner = landmarks[mouth_left_corner_index]
    mouth_right_corner = landmarks[mouth_right_corner_index]

    mouth_height = euclidean_distance(upper_lip_top, lower_lip_bottom)
    
    mouth_width = euclidean_distance(mouth_left_corner, mouth_right_corner)
    
    ratio = mouth_height / mouth_width

    return ratio > threshold

def lip_detection_in_video(video_path, early_q, total_frames):
    # from mediapipe.python.solution import face_mesh as mp_face_mesh
    mp_face_mesh = mediapipe.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    # print(face_mesh)
    
    def lip_detection_in_frame(frame):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lip_indexes = [61, 146, 146, 91, 91, 181, 181, 84, 84, 17, 17, 314, 314, 405, 405, 321,
                    321, 375, 375, 291, 61, 185, 185, 40, 40, 39, 39, 37, 37, 0, 0, 267, 267,
                    269, 269, 270, 270, 409, 409, 291, 78, 95, 95, 88, 88, 178, 178, 87, 87, 14,
                    14, 317, 317, 402, 402, 318, 318, 324, 324, 308, 78, 191, 191, 80, 80, 81,
                    81, 82, 82, 13, 13, 312, 312, 311, 311, 310, 310, 415, 415, 308,]

        lip_contour = [61, 185, 40, 39, 37, 0, 267, 270] + [146, 91, 181, 84, 17, 314, 405, 321] # lip_high: 0, lip_low: 17
        
        upper_lip_top_index = 13 
        lower_lip_bottom_index = 14 
        mouth_left_corner_index = 61 
        mouth_right_corner_index = 291  
        
        lip_landmarks = []
        lip_coords = []
        state = None
        try:
            results = face_mesh.process(frame)

            if results.multi_face_landmarks:
                
                face_landmarks = results.multi_face_landmarks[0]
                # print(len(face_landmarks.landmark))
                landmarks = {i: [landmark.x, landmark.y, landmark.z] for i, landmark in enumerate(face_landmarks.landmark)}
                
                state = is_mouth_open(landmarks)
                lip_landmarks = [landmarks[i][:-1] for i in lip_contour]
                lip_landmarks = [[round(x*frame.shape[1]), round(y*frame.shape[0])] for x, y in lip_landmarks]
                # print(lip_landmarks)
            
                # lip_coords = list(cv2.boundingRect(np.array(lip_landmarks))) # x, y, w, h
                x,y,w,h = cv2.boundingRect(np.array(lip_landmarks))
                # lip_coords = [lip_coords[0], lip_coords[1], lip_coords[0]+lip_coords[2], lip_coords[1]+lip_coords[3]] # x1, y1, x2, y2
                lip_coords = [x, y, x+w, y+h]
                # print(lip_coords)
                return lip_coords, state # coords: x1, y1, x2, y2, state: True or False
            else:
                return None, None # coords: None, state: None

        except Exception as e:
            print(e)
            traceback.print_exc()
            return None, None # coords: None, state: None

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    current_frame = 0
    faild_frames =[]
    last_rect, last_state = None, None
    print(f'min_total_frames: {total_frames}')

    print(f"{mp.current_process().name} Processing video: {video_path}")
    while(cap.isOpened() and current_frame <= total_frames):
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (1280, 720))
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        if ret:
            rect, state = lip_detection_in_frame(frame) # rect: x1, y1, x2, y2, state: True or False
            if rect is not None: # rect is not None and state is not None
                last_rect, last_state = rect, state
                lip_roi = frame[rect[1]:rect[3], rect[0]:rect[2]]
                early_q.put((current_frame, lip_roi, state)) # current_frame, lip_roi, state
                
            else: # rect is None or state is None
                if last_rect:
                    lip_roi = frame[int(last_rect[1]):int(last_rect[3]), int(last_rect[0]):int(last_rect[2])]
                    early_q.put((current_frame, lip_roi, state)) # current_frame, lip_roi, state
                faild_frames.append(current_frame)
        else: # 
            print(f"frame {current_frame}: No frame")
            early_q.put((current_frame, None, None)) # current_frame, lip_roi, state
        
        if current_frame == total_frames:
            break
        # break
        
    # print(f"lip_roi: {lip_roi}, state: {state}")
    print(f'{video_path.split("/")[-1]}: Done, failed frames: {len(faild_frames)}')
    cap.release()

def infer_lip_state(early_q, result_list, model_name, weights):
    # early_q item: (frame_number, lip_roi, state) or (frame_number, None, None) or 'LAST'

    fix_seed(999)
    model, preprocess = get_model(model_name=model_name, num_classes=2)
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    print(f"{mp.current_process().name} Model loaded")

    frame_number_batch = []  # keep len 30
    img_batch = []  # keep len 30
    state_batch = []  # keep len 30

    while True:
        if not early_q.empty():
            try:
                item = early_q.get(block=False)
                if item == 'LAST': 
                    # process remaining items
                    print(f"{mp.current_process().name} Last item received")
                    if len(frame_number_batch) > 0:
                        img_batch = [preprocess(img) for img in img_batch]
                        img_batch = torch.stack(img_batch).cuda()
                        with torch.no_grad():
                            outputs = model(img_batch)
                            outputs = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
                            for idx, state, output in zip(frame_number_batch, state_batch, outputs):
                                result_list[idx-1] = (state, output)
                            frame_number_batch = []
                            img_batch = []
                            state_batch = []
                    break
                
                
                elif item[1] is not None: # lip_roi is not None
                    """
                    gather 30 frames, then process if full
                    when processing, change the items to inferenced value of each index in result_list
                    """

                    # if 0<=item[0]<=60:
                    #     cv2.imwrite(f'./ssibal/{mp.current_process().name}_{item[0]}.jpg', item[1])

                    frame_number_batch.append(item[0])
                    img_batch.append(Image.fromarray(item[1],'RGB')) # lip_roi:
                    state_batch.append(item[2])

                    if len(frame_number_batch) == 30:
                        print(f"{mp.current_process().name} Processing frames: {frame_number_batch[0]} ~ {frame_number_batch[-1]}")
                        img_batch = [preprocess(img) for img in img_batch]
                        img_batch = torch.stack(img_batch).cuda()
                        with torch.no_grad():
                            outputs = model(img_batch)
                            # outputs = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
                            outputs = outputs[:, 1].cpu().numpy().tolist()
                            for idx, state, output in zip(frame_number_batch, state_batch, outputs):
                                result_list[idx-1] = (state, output)
                            frame_number_batch = []
                            img_batch = []
                            state_batch = []

                elif item[1] is None: # lip_roi is None

                    print(f"{mp.current_process().name} Frame {item[0]-1}: No lip_roi")
                    result_list[item[0]-1] = (None, None)
                    pass


            except Exception as e:
                print(e)
                traceback.print_exc()
                break

if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    start = time.time()

    parser = ArgumentParser()
    parser.add_argument('--video1', type=str, default='/home2/multicam/dataset/ETRI/202309/230725_opentalk/clipped/camera3_synced.mp4')
    parser.add_argument('--video2', type=str, default='/home2/multicam/dataset/ETRI/202309/230725_opentalk/clipped/camera2_synced.mp4')
    parser.add_argument('--classification_model', type=str, default='swin_v2_b')
    parser.add_argument('--weights', type=str, default='/home2/multicam/2024_Multicam/checkpoints/run0621_0339/snapshot_swin_v2_b_2_0.9563032640482664.pth')
    parser.add_argument('--output', type=str, default='result.json')
    args = parser.parse_args()

    # video_path1 = '/home2/multicam/dataset/ETRI/202309/230725_opentalk/clipped/camera2_synced.mp4'
    # video_path2 = '/home2/multicam/dataset/ETRI/202309/230725_opentalk/clipped/camera3_synced.mp4'
    video_path1 = args.video1
    video_path2 = args.video2

    cap = cv2.VideoCapture(video_path1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cap = cv2.VideoCapture(video_path2)
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), total_frames)
    cap.release()

    early_q1 = mp.Queue()
    early_q2 = mp.Queue()

    result_list1 = mp.Manager().list([None for _ in range(total_frames)]) # -> index: frame_number, value: (state, speak_prop)
    result_list2 = mp.Manager().list([None for _ in range(total_frames)]) # -> index: frame_number, value: (state, speak_prop)

    prd_p1 = mp.Process(target=lip_detection_in_video, args=(video_path1, early_q1, total_frames,))
    prd_p2 = mp.Process(target=lip_detection_in_video, args=(video_path2, early_q2, total_frames,))
    prd_p1.start()
    prd_p2.start()
    print("Producer started")

    csm_p1 = mp.Process(target=infer_lip_state, args=(early_q1, result_list1, args.classification_model, args.weights))
    csm_p2 = mp.Process(target=infer_lip_state, args=(early_q2, result_list2, args.classification_model, args.weights))
    csm_p1.start()
    csm_p2.start()
    print("Consumer started")

    prd_p1.join()
    prd_p2.join()
    print("Producers joined")

    early_q1.put('LAST')
    early_q2.put('LAST')
    print("Sentinel value sent")

    csm_p1.join()
    csm_p2.join()
    print("Consumers joined")

    """
    result_list: index: frame_number-1, value: (state, speak_prop) or (None, None) of frame_number
    """

    result_dict = {i: [] for i in range(total_frames)}
    for i, (r1, r2) in enumerate(zip(result_list1, result_list2)):
        # print(f"Frame {i+1}: {r1}, {r2}")
        r1_state, r1_speak_prop = r1
        r2_state, r2_speak_prop = r2

        # if r1_state is not None and r1_prop is not None:
        result_dict[i].append(
            {
                'camera': 1,
                'state': r1_state,
                'speak_prop': r1_speak_prop
            }
        )
        
        # if r2_state is not None and r2_prop is not None:
        result_dict[i].append(
            {
                'camera': 2,
                'state': r2_state,
                'speak_prop': r2_speak_prop
            }
        )

    with open(args.output, 'w') as f:
        json.dump(result_dict, f, indent=4)
    
    print(f"Elapsed time: {time.time()-start:.4f} sec")
        