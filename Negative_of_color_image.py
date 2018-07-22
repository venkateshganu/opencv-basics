import cv2
img1=cv2.imread('beach.jpg')
img3=cv2.bitwise_not(img1)
cv2.imshow('image',img3)
cv2.waitKey(0)
cv2.destroyAllWindows() 