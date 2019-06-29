import cv2
import numpy as np
image=cv2.imread("C:\\Users\\venka\\OneDrive\\Desktop\\Opencv docs\\Color change\\black.png")
image[np.where((image==[0,0,0]).all(axis=2))]=[0,0,255]
cv2.imwrite('red.png',image)
