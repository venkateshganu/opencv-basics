import cv2
import sys
imagePath=sys.argv[1]
img1=cv2.imread(imagePath)
cv2.imshow('image',img1)
cv2.waitKey(0)
cv2.destroyAllWindows() 