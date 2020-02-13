import cv2
import numpy as np

img = []
for i in range(0,2):    #change as per length of dataset
	img.append(cv2.imread('test_img/image' + str(i) + '.png'))

resized = []
hue, sat, val = ([], [], [])
for i in range(len(img)):
	resized.append(cv2.resize(img[i], (200, 200), interpolation = cv2.INTER_AREA))
 
	hsv = cv2.cvtColor(resized[i], cv2.COLOR_BGR2HSV)
	hue.append(hsv[:,:,0])
	sat.append(hsv[:,:,1])
	val.append(hsv[:,:,2])

for i in range(len(img)):
	print('hue ' + str(np.mean(hue[i])), end = ' ')
	print('sat ' + str(np.mean(sat[i])), end = ' ')
	print('val ' + str(np.mean(val[i])), end = ' ')
	print('\n')
	


	
#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

