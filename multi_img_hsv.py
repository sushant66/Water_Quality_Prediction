import cv2
import numpy as np
import pandas as pd

img = []
for i in range(0,14):    #change as per length of dataset
	img.append(cv2.imread('images/image' + str(i) + '.jpg'))


resized = []
hue, sat, val = ([], [], [])
for i in range(len(img)):
	resized.append(cv2.resize(img[i], (200, 200), interpolation = cv2.INTER_AREA))
 
	hsv = cv2.cvtColor(resized[i], cv2.COLOR_BGR2HSV)
	hue.append(np.mean(hsv[:,:,0]))
	sat.append(np.mean(hsv[:,:,1]))
	val.append(np.mean(hsv[:,:,2]))


# for i in range(len(img)):
# 	print('hue ' + str(np.mean(hue[i])), end = ' ')
# 	print('sat ' + str(np.mean(sat[i])), end = ' ')
# 	print('val ' + str(np.mean(val[i])), end = ' ')
	
# 	print('\n')
out1 = [520,556,518,583,629,703,754,786,866,954,1009,1020,1020,1022]
data = {'Hue': hue, 'Sat': sat, 'Val': val, 'Out': out1}
df = pd.DataFrame(data)
df.to_csv(r'dataset.csv')
print(df)
