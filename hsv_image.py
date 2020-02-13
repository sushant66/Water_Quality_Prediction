import cv2
import numpy as np

img = cv2.imread('image.png')

print('Original Dimensions : ',img.shape)
 
#scale_percent = 60 # percent of original size
#width = int(img.shape[1] * scale_percent / 100)
#height = int(img.shape[0] * scale_percent / 100)
#dim = (width, height)
# resize image
resized = cv2.resize(img, (200, 200), interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape) 
cv2.imshow('image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
hue = hsv[:,:,0]
sat = hsv[:,:,1]
val = hsv[:,:,2]
print('hue ' + str(np.mean(hue)))
print('sat ' + str(np.mean(sat)))
print('val ' + str(np.mean(val)))
#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

