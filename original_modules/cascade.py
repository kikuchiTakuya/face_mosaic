import os
import cv2

# 分類器のパス
face_cascade_path = 'original_modules/haarcascade_xml/haarcascade_frontalface_alt2.xml'
eye_cascade_path = 'original_modules/haarcascade_xml/haarcascade_eye.xml'
lefteye_cascade_path = 'original_modules/haarcascade_xml//haarcascade_lefteye_2splits.xml'
righteye_cascade_path = 'original_modules/haarcascade_xml//haarcascade_righteye_2splits.xml'

assert os.path.isfile(face_cascade_path), "No file face_cascade xml"

# 分類器作成
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
lefteye_cascade = cv2.CascadeClassifier(lefteye_cascade_path)
righteye_cascade = cv2.CascadeClassifier(righteye_cascade_path)

def judge_face(frame):

    ''' 
    グレースケールにしてから分類する（顔）
    引数：画像(3チャンネル)
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    return faces

def judge_eyes(frame, faces):

    for x, y, w, h in faces:

        # グレースケールにしてから分類する（目）
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray)
    
    return eyes

def judge_lefteye(frame, faces):

    for x, y, w, h in faces:

        # グレースケールにしてから分類する（目）
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        face_gray = gray[y:y+h, x:x+w]
        lefteye = lefteye_cascade.detectMultiScale(face_gray)
    
    return lefteye

def judge_righteye(frame, faces):

    for x, y, w, h in faces:

        # グレースケールにしてから分類する（目）
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        face_gray = gray[y:y+h, x:x+w]
        righteye = righteye_cascade.detectMultiScale(face_gray)
    
    return righteye
    