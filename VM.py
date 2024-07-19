import  mediapipe as mp
import cv2
import numpy as np
import util
import pyautogui
screen_width,screen_height =pyautogui.size()
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode = False,
    model_complexity =1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7,
    max_num_hands = 1
)

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
     hand_landmarks =processed.multi_hand_landmarks[0] 
     return hand_landmarks.landmark[mp.Hands.HandLandmark.INDEX_FINGER_TIP]
    return None




def get_angle(a,b,c):
    radians = np.arctan2(c[1] -b[1],c[0] -b[0])- np.arctan2(a[1]- b[1], a[0]- b[0])
    angle = np.abs(np.degrees(radians))
    return angle
    
       
def get_distance(landmark_ist):
    if len(landmark_ist) < 2:
        return
    (x1,y1), (x2,y2) = landmark_ist[0],landmark_ist[1]
    L =np.hypot(x2- x1, y2-y1)
    return np.interp(L, [0, 1],[0,1000])










def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    return None



def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x= int(index_finger_tip.x *screen_width)
        y =int(index_finger_tip.y *screen_height)
        pyautogui.moveTo(x,y)
    

def detect_gestures(frame,landmarks_list,processed):
    if len(landmarks_list)>=21:
        index_finger_tip =find_finger_tip(processed)
        thumb_index_dist = util.get_distance([landmarks_list[4],landmarks_list[5]])
        if thumb_index_dist<50 and util.get_angle(landmarks_list[5],landmarks_list[6],landmarks_list[8] > 90):
            move_mouse(index_finger_tip)

def main():
    cap = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)
            
            
            landmarks_list = []
            
            if processed.multi_hand_landmarks:
                hand_landmarks= processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame,hand_landmarks,mpHands.HAND_CONNECTIONS)
                
                
                for lm in hand_landmarks.landmark:
                    landmarks_list.append((lm.x,lm.y))
           
            detect_gestures(frame,landmarks_list,processed)        
                
                
            
            
            

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()