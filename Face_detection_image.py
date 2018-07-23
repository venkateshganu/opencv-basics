import sys
import cv2
imagePath=sys.argv[1]
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
image=cv2.imread(imagePath)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray,scaleFactor=1.35,minNeighbors=5,minSize=(30,30),flags=0)
for(x,y,w,h) in faces:
    windowName=None
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("sample",image)
    cv2.waitKey(0)
cv2.waitKey(0)
cv2.destroyAllWindows()