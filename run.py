import cv2
import os
import pyvirtualcam

face_cascade_path = os.path.join(
    cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
)
face_cascade = cv2.CascadeClassifier(face_cascade_path)
mask = cv2.imread("oruga.jpg")
idx = input("Enter camera ID:(default:0)")
try:
    idx = int(idx)
except:
    idx = 0
print(f"camera ID:{idx})
cap = cv2.VideoCapture(idx)
ret,frame = cap.read()
height, width, ch = frame.shape
print("CODE RUNNING.")
with pyvirtualcam.Camera(width=width, height=height, fps=30) as cam:
    while True:
      try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        for x, y, w, h in faces:
          resized_mask = cv2.resize(mask, (w,h))
          frame[y:y+h, x:x+w] = resized_mask
        cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      except Exception as e:
        print(f"stop. reason:{e}")
        cap.release()

