import warnings
import cv2
import numpy as np
import random
# from mediapipe.python.solutions import face_detection 
import time
import multiprocessing as mp
import torch
import os
import json

from swin_binary import swin_binary
from torchvision import transforms

warnings.filterwarnings("ignore")



def face_detection_in_video(video_path: str, early_q: mp.Queue, min_total_frames: int):
    from mediapipe.python.solutions import face_detection
    import cv2
    face_detection_module = face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7,)

    def face_detection_in_frame(frame):

        results = face_detection_module.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return None
        else:
            detection = results.detections[0]
            x1 = int(round(detection.location_data.relative_bounding_box.xmin * frame.shape[1]))
            y1 = int(round(detection.location_data.relative_bounding_box.ymin * frame.shape[0]))
            x2 = int(round((detection.location_data.relative_bounding_box.xmin + detection.location_data.relative_bounding_box.width) * frame.shape[1]))
            y2 = int(round((detection.location_data.relative_bounding_box.ymin + detection.location_data.relative_bounding_box.height) * frame.shape[0]))

            return [x1, x2, y1, y2]

    cap = cv2.VideoCapture(video_path)
    # global MIN_TOTAL_FRAMES
    current_frame = 0
    failed_frames = []
    last_rect = None
    print(f"min_total_frames: {min_total_frames}")
    while(cap.isOpened() and current_frame <= min_total_frames):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if ret:
            rect = face_detection_in_frame(frame)
            if rect:
                last_rect = rect
                froi = frame[rect[2]:rect[3], rect[0]:rect[1]]
                froi = cv2.resize(froi, (224, 224))
                early_q.put([current_frame, froi, True])  # frame number, face roi, is_valid

            else:  # rect = None
                if last_rect:
                    froi = frame[last_rect[2]:last_rect[3], last_rect[0]:last_rect[1]]
                    froi = cv2.resize(froi, (224, 224))
                    early_q.put([current_frame, froi, False])  # frame number, face roi, is_valid

                failed_frames.append(current_frame)

        else:
            print(f"{video_path.split('/')[-1]}: No frame")

        if current_frame == min_total_frames:
            break

    print(f"{video_path.split('/')[-1]}: Failed frames: {failed_frames}")
    cap.release()
    # early_q.put('LAST')

def make_batch_tensor(batch_imgs):
    # batch_imgs = np.array(batch_imgs)
    batch_imgs = torch.stack(batch_imgs)
    return batch_imgs

def speech_recognition(model, batch_tensor):
    with torch.no_grad():
        output = model(batch_tensor)
    # output = output.cpu().numpy().tolist()
    return output

def process_batch(frame_batch: list, img_batch: list, is_rect_batch: list, results: list, swin_talking_model):
    batch_tensor = make_batch_tensor(img_batch)
    batch_tensor = batch_tensor.cuda()
    # batch_tensor = batch_tensor.permute(0, 3, 1, 2)
    outputs = speech_recognition(swin_talking_model, batch_tensor)
    # print(f"outputs: {outputs.size()}") # (30, 2)tensors
    outputs = outputs.cpu().detach().tolist() # (30, 2)list

    # outputs = outputs.cpu().detach().numpy().reshape(-1).tolist()

    for frame_n, output, is_rect in zip(frame_batch, outputs, is_rect_batch):
        # print(f"output: {type(output[0])}")
        results.append([frame_n, output, is_rect])

def consumer(early_q, result_list):
    frame_number_batch = []  # keep len 30
    img_batch = []  # keep len 30
    is_valid_batch = []
    batch_tensor = None

    seed_number = 999
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    swin_talking_model = swin_binary()
    swin_talking_model = torch.nn.DataParallel(swin_talking_model).cuda()
    # swin_talking_model = torch.nn.DataParallel(swin_talking_model, [0,1])
    checkpoint = torch.load('/home2/multicam/AIHUB_FACE/checkpoints/swin_binary_10_0.86389784006207.pth')
    swin_talking_model.module.load_state_dict(checkpoint['model_state_dict'])
    swin_talking_model.eval()
    print("Model loaded")
    
    while True:  # gather items every 30 frames
        if not early_q.empty():
            try:
                item = early_q.get(block=False)

                if item == 'LAST':  # Sentinel value
                    print("Last item received")

                    if img_batch and frame_number_batch and is_valid_batch and \
                            0 < len(img_batch) == len(frame_number_batch) == len(is_valid_batch) < 30:
                        process_batch(frame_number_batch, img_batch, is_valid_batch, result_list, swin_talking_model)
                        print(f"{mp.current_process().name}--{frame_n}--result_list:{len(result_list)}")
                    
                    # result_list.append(results)
                    print(f"{mp.current_process().name}--Last--result_list:{len(result_list)}")

                    break

                else:
                    frame_n, froi, is_vaild = item
                    frame_number_batch.append(frame_n)
                    img_batch.append(froi)
                    is_valid_batch.append(is_vaild)

                    if len(frame_number_batch) == 30 and len(img_batch) == 30 and len(is_valid_batch) == 30:
                        process_batch(frame_number_batch, img_batch, is_valid_batch, result_list, swin_talking_model)
                        print(f"{mp.current_process().name}--{frame_n}--result_list:{len(result_list)}")
                        frame_number_batch = []  # keep len under 30
                        img_batch = []  # keep len under 30
                        is_valid_batch = []  # keep len under 30

                    # early_q.task_done()

            except Exception as e:
                print(f"Error: {e}")
                # early_q.task_done()
                break

def consumer2(early_q, result_list): # 1100초 소요 + 800초 소요
    frame_number_batch = []  # keep len 30
    img_batch = []  # keep len 30
    is_valid_batch = []
    batch_tensor = None
    
    seed_number = 999
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    swin_talking_model = swin_binary()
    swin_talking_model = torch.nn.DataParallel(swin_talking_model).cuda()
    # swin_talking_model = torch.nn.DataParallel(swin_talking_model, [0,1])
    checkpoint = torch.load('/home2/multicam/AIHUB_FACE/checkpoints/swin_binary_10_0.86389784006207.pth')
    swin_talking_model.module.load_state_dict(checkpoint['model_state_dict'])
    swin_talking_model.eval()
    print("Model loaded")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    while True:
        if not early_q.empty():
            try:
                item = early_q.get(block=False)

                if item == 'LAST':
                    print("Last item received")
                    if img_batch and frame_number_batch and is_valid_batch and \
                            0 < len(img_batch) == len(frame_number_batch) == len(is_valid_batch) < 30:
                        process_batch(frame_number_batch, img_batch, is_valid_batch, result_list, swin_talking_model)
                        print(f"{mp.current_process().name}--{frame_n}--result_list:{len(result_list)}")
                    
                    # result_list.append(results)
                    print(f"{mp.current_process().name}--Last--result_list:{len(result_list)}")
                    break

                else:
                    frame_n, froi, is_vaild = item
                    froi = transform(froi)

                    frame_number_batch.append(frame_n)
                    img_batch.append(froi)
                    is_valid_batch.append(is_vaild)

                    if len(frame_number_batch) == 30 and len(img_batch) == 30 and len(is_valid_batch) == 30:
                        
                        process_batch(frame_number_batch, img_batch, is_valid_batch, result_list, swin_talking_model)
                        print(f"{mp.current_process().name}--{frame_n}--result_list:{len(result_list)}")
                        frame_number_batch = []  # keep len under 30
                        img_batch = []  # keep len under 30
                        is_valid_batch = []  # keep len under 30

                    # output = speech_recognition(swin_talking_model, froi)
                    # output = output.cpu().detach().numpy().reshape(-1).tolist() 
                    # result_list.append([frame_n, output, is_vaild])


            except Exception as e:
                print(f"Error: {e}")
                break



if __name__ == '__main__':
    
    mp.set_start_method('spawn', force=True)
    # torch.multiprocessing.set_start_method('spawn', force=True)

    # seed_number = 999
    # random.seed(seed_number)
    # np.random.seed(seed_number)
    # torch.manual_seed(seed_number)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    video_path1 = '/home2/multicam/dataset/ETRI/202309/230725_opentalk/clipped/camera1_synced.mp4'
    video_path2 = '/home2/multicam/dataset/ETRI/202309/230725_opentalk/clipped/camera2_synced.mp4'

    cap = cv2.VideoCapture(video_path1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cap = cv2.VideoCapture(video_path2)
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), total_frames)
    
    early_q1 = mp.Queue()
    early_q2 = mp.Queue()

    result_list1 = mp.Manager().list()
    result_list2 = mp.Manager().list()

    start = time.time()
    prd_p1 = mp.Process(target=face_detection_in_video, args=(video_path1, early_q1, total_frames,))
    prd_p2 = mp.Process(target=face_detection_in_video, args=(video_path2, early_q2, total_frames,))
    prd_p1.start()
    prd_p2.start()
    print("Producers started")

    csm_p1 = mp.Process(target=consumer2, args=(early_q1, result_list1))
    csm_p2 = mp.Process(target=consumer2, args=(early_q2, result_list2))
    csm_p1.start()
    csm_p2.start()
    print("Consumers started")

    prd_p1.join()
    prd_p2.join()
    print("Producers joined")

    early_q1.put('LAST')
    early_q2.put('LAST')
    print("Sentinel value sent")

    csm_p1.join()
    csm_p2.join()
    print("Consumers joined")

    print(f"Time taken: {time.time()-start}")


    results1 = list(result_list1)
    results2 = list(result_list2)

    print(f"results1: {len(results1)}, results2: {len(results2)}")

    result_dict = {}
    previous_channel = 0
    for r1, r2 in zip(results1, results2,):
        frame_n1, probs_ch1, is_valid1 = r1
        frame_n2, probs_ch2, is_valid2 = r2
        # print(f"frame_n1: {frame_n1}, frame_n2: {frame_n2}")

        if is_valid1 and is_valid2:
            selected_channel = 1 if probs_ch1[1] > probs_ch2[1] else 2
            selected_channel = 0 if abs(probs_ch1[1] - probs_ch2[1]) < 0.05 else selected_channel
            
        elif not is_valid1 or not is_valid2:
            selected_channel = previous_channel
        elif not is_valid1 and not is_valid2:
            selected_channel = previous_channel

        previous_channel = selected_channel

        frame_result = {
            'frame_n': frame_n1,
            'previous_channel': previous_channel,
            'selected_channel': selected_channel,
            'probs_ch1': probs_ch1,
            'probs_ch2': probs_ch2,
            'is_valid1': is_valid1,
            'is_valid2': is_valid2
        }
        result_dict[frame_n1] = frame_result

    with open('./tmp3_inference_result_dict.json', 'w') as f:
        json.dump(result_dict, f, indent=4)

    print(f"Processing complete, results saved to tmp4_inference_result_dict.json")


