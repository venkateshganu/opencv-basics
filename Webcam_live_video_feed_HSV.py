import cv2
videocapture=cv2.VideoCapture(0)
if not videocapture.isOpened():
    print("can't open camera")
    exit()
windowName="Webcam Live video feed"
showLive=True
while(showLive):
    ret, frame=videocapture.read()
    color_to_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    if not ret:
        print("cannot capture the frame")
        exit()
        
    cv2.imshow(windowName, color_to_hsv)
    if cv2.waitKey(30)>=0:
        showLive=False
        
videocapture.release()
cv2.destroyAllWindows()