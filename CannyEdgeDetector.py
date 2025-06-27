import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 

def Canny_detector(img, weak_th=None, high_th=None):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  #algo works on pixel intensity so color doesnt matter
    img=cv.GaussianBlur(img,(5,5),1.4)  

    #Finding x and y gradients

    gx=cv.Sobel(np.float32(img), cv.CV_64F, 1,0,3)
    gy=cv.Sobel(np.float32(img), cv.CV_64F, 0,1,3)

    #Convert Cartesian gradient to polar form 

    mag,ang= cv.cartToPolar(gx,gy,angleInDegrees=True)

    mag_max = np.max(mag)
    if not weak_th: weak_th = mag_max * 0.1
    if not high_th: high_th = mag_max * 0.5

    height,width=img.shape

    for p_x in range(width):
        for p_y in range(height):

            grad_ang= ang[p_y,p_x]
            grad_ang=abs(grad_ang-180) if grad_ang>180 else abs(grad_ang)
            if grad_ang <= 22.5:
                    neighb_1_x, neighb_1_y = p_x-1, p_y
                    neighb_2_x, neighb_2_y = p_x+1, p_y
            elif grad_ang > 22.5 and grad_ang <= 67.5:
                    neighb_1_x, neighb_1_y = p_x-1, p_y-1
                    neighb_2_x, neighb_2_y = p_x+1, p_y+1
            elif grad_ang > 67.5 and grad_ang <= 112.5:
                    neighb_1_x, neighb_1_y = p_x, p_y-1
                    neighb_2_x, neighb_2_y = p_x, p_y+1
            elif grad_ang > 112.5 and grad_ang <= 157.5:
                    neighb_1_x, neighb_1_y = p_x-1, p_y+1
                    neighb_2_x, neighb_2_y = p_x+1, p_y-1
            else:
                    neighb_1_x, neighb_1_y = p_x-1, p_y
                    neighb_2_x, neighb_2_y = p_x+1, p_y
            
             # Non-maximum suppression step
            if width>neighb_1_x>= 0 and height>neighb_1_y>= 0:
                if mag[p_y, p_x]<mag[neighb_1_y, neighb_1_x]:
                    mag[p_y, p_x]= 0
                    continue
 
            if width>neighb_2_x>= 0 and height>neighb_2_y>= 0:
                if mag[p_y, p_x]<mag[neighb_2_y, neighb_2_x]:
                    mag[p_y, p_x]= 0
    ids = np.zeros_like(img)
     
    # double thresholding step
    for i_x in range(width):
        for i_y in range(height):
            
            grad_mag = mag[i_y, i_x]
            
            if grad_mag<weak_th:
                mag[i_y, i_x]= 0
            elif high_th>grad_mag>= weak_th:
                ids[i_y, i_x]= 1
            else:
                ids[i_y, i_x]= 2

    mag = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    mag = np.uint8(mag)

    return mag

vid=cv.VideoCapture(0)

while True:
    _, frame= vid.read()

    canny_img=Canny_detector(frame)
    cv.imshow('Canny Edge Detector', canny_img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()








