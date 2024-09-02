import cv2
import mediapipe as mp
from IPython.display import Image, display
import numpy as np
import time
import itertools
import traceback

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

# 입이 벌어졌는지 판단하는 함수
def is_mouth_open(landmarks, threshold):
    upper_lip_top_index = 13 
    lower_lip_bottom_index = 14 
    mouth_left_corner_index = 61 
    mouth_right_corner_index = 291  
    left_eye_index = 130 # 130
    right_eye_index = 263 # 263

    original_angles = [np.array(landmarks[13]), np.array(landmarks[14]), np.array(landmarks[61]), np.array(landmarks[291])]
    original_mouth_height = euclidean_distance(landmarks[13], landmarks[14])
    original_mouth_width = euclidean_distance(landmarks[61], landmarks[291])
    
    upper_lip_top = correct_angle(landmarks, landmarks[upper_lip_top_index], left_eye_index, right_eye_index)
    lower_lip_bottom = correct_angle(landmarks, landmarks[lower_lip_bottom_index], left_eye_index, right_eye_index)
    mouth_left_corner = correct_angle(landmarks, landmarks[mouth_left_corner_index], left_eye_index, right_eye_index)
    mouth_right_corner = correct_angle(landmarks, landmarks[mouth_right_corner_index], left_eye_index, right_eye_index)
    
    corrected_angles = [upper_lip_top, lower_lip_bottom, mouth_left_corner, mouth_right_corner]
    # 입 높이 계산
    mouth_height = euclidean_distance(upper_lip_top, lower_lip_bottom)
    
    # 입 너비 계산
    mouth_width = euclidean_distance(mouth_left_corner, mouth_right_corner)
    
    # 비율 계산
    original_ratio = original_mouth_height / original_mouth_width
    correted_ratio = mouth_height / mouth_width

    for ori, cor in zip(original_angles, corrected_angles):
        print(f"ori: {ori}, cor: {cor}")
    print(f"original_ratio: {original_ratio}, correted_ratio: {correted_ratio}")
    # print(f"mouth_height: {mouth_height}, mouth_width: {mouth_width}, ratio: {ratio}")
    # 입이 벌어졌는지 판단
    return correted_ratio > threshold


# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
 
def lip_extraction_and_isOpen(img, face_mesh, threshold=0.05):
    if isinstance(img, str):
        image_path = img
        image = cv2.imread(image_path)
    else:
        image_path = 'tmp.jpg'
        image = img

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # lip_idxs = [61, 185, 40, 39, 37, 0, 267, 270] + [146, 91, 181, 84, 17, 314, 405, 321] # lip_high: 0, lip_low: 17
    lip_contour = [61, 185, 40, 39, 37, 0, 267, 270] + [146, 91, 181, 84, 17, 314, 405, 321] # lip_high: 0, lip_low: 17
    lip_indexes = [61, 146, 146, 91, 91, 181, 181, 84, 84, 17, 17, 314, 314, 405, 405, 321,
                    321, 375, 375, 291, 61, 185, 185, 40, 40, 39, 39, 37, 37, 0, 0, 267, 267,
                    269, 269, 270, 270, 409, 409, 291, 78, 95, 95, 88, 88, 178, 178, 87, 87, 14,
                    14, 317, 317, 402, 402, 318, 318, 324, 324, 308, 78, 191, 191, 80, 80, 81,
                    81, 82, 82, 13, 13, 312, 312, 311, 311, 310, 310, 415, 415, 308,]

    face_contour = [10, 338, 297,332,284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,] 
        # face_high: 10, face_low: 152
    try:
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            

            for face_no, face_landmarks in enumerate(results.multi_face_landmarks):
                # print(dir(face_landmarks))
                # print("face_no: ", face_no)
                lip_landmarks = []
                face_contour_landmarks = []
                # print(len(face_landmarks.landmark), type(face_landmarks.landmark))

                landmarks = {i: [landmark.x, landmark.y, landmark.z] for i, landmark in enumerate(face_landmarks.landmark)}

                for idx, landmark in landmarks.items():
                    if idx in lip_contour:
                        # print(landmark)
                        # x = int(landmark.x * image.shape[1])
                        # y = int(landmark.y * image.shape[0])
                        x, y, z = landmark
                        # print(x, y, z)
                        x = int(x * image.shape[1])
                        y = int(y * image.shape[0])

                        # cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                        # cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        lip_landmarks.append([x, y])
                            # print(x, y)    
                
                coord_x, coord_y, coord_w, coord_h = cv2.boundingRect(np.array(lip_landmarks))
                lip_coords = [coord_x, coord_y, coord_w, coord_h]
                state = is_mouth_open(landmarks, threshold)
                
        return lip_coords, state

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

        return None, None

if __name__ == '__main__':
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    lip_extraction_and_isOpen('/home2/multicam/AIHUB_LIP/face1.png', face_mesh)
