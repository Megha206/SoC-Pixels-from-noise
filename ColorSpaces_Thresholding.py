import cv2 as cv
import numpy 
from matplotlib import pyplot as plt

nemo=cv.imread(r"C:\Users\raome\Downloads\nemo0.jpg")
nemo=cv.cvtColor(nemo,cv.COLOR_BGR2RGB)
hsv_nemo = cv.cvtColor(nemo, cv.COLOR_RGB2HSV)
light_orange = (1, 190, 200)
dark_orange = (18, 255, 255)
mask = cv.inRange(hsv_nemo, light_orange, dark_orange)
result = cv.bitwise_and(nemo, nemo, mask=mask)
plt.subplot(1,2,1)
plt.imshow(mask,'gray')
plt.subplot(1,2,2)
plt.imshow(result)
plt.show()