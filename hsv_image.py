import cv2
import numpy as np

img = cv2.imread('image.png')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue = hsv[:,:,0]
sat = hsv[:,:,1]
val = hsv[:,:,2]
print('hue ' + str(np.mean(hue)))
print('sat ' + str(np.mean(sat)))
print('val ' + str(np.mean(val)))
#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

