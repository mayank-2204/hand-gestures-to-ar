import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
#@markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.

global_img = None

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global frame_c,global_img
    print('hand landmarker result: {}'.format(result))
    # global_img = draw_landmarks_on_video_stream(result,output_image)
    global_img = draw_landmarks_on_video_stream(result,frame_c)

def draw_landmarks_on_video_stream(detection_result: HandLandmarkerResult,rgb_image: np.ndarray):
    print("Drawing Landwmarks.")
    # global global_img
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    # rgb_image = mp_image.numpy_view()
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

# STEP 2: Create an HandLandmarker object.
# base_options = python.BaseOptions(model_asset_path='/Users/mayank/Documents/POCS/Hand Gesture To AR/hand_landmarker.task')
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='/Users/mayank/Documents/POCS/Hand Gesture To AR/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=4)
detector = vision.HandLandmarker.create_from_options(options)

vid_support = cv2.VideoCapture(1)
vid_support.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
vid_support.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
vid_client = cv2.VideoCapture(0)
vid_client.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
vid_client.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
frame_num = 0

while(True): 
    # global_img = None
    # Capture the video frame 
    # by frame 
    ret_s, frame_s = vid_support.read() 
    ret_c, frame_c = vid_client.read()
    # Display the resulting frame 
    # cv2.imshow('frame', frame)
    if ret_c:
        if frame_num % 1 == 0:
            
            if ret_s:

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = frame_s)

                # global_img = frame_c

                detector.detect_async(mp_image,mp.Timestamp.from_seconds(time.time()).value)
                if global_img is not None:
                    cv2.imshow('Support Side',frame_s)
                    cv2.waitKey(1) 
                    cv2.imshow('Client Side',global_img)
                    
                frame_num += 1
            
        else: 
            frame_num += 1
            
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    cv2.waitKey(1) 
    if 0xFF == ord('q'):
        break


# After the loop release the cap object 
vid_client.release() 
vid_support.release()
# Destroy all the windows 
cv2.destroyAllWindows() 