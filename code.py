import cv2
import time
import numpy as np

#To save the output in a file
vid = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', vid, 20.0, (640, 480))

#Starting the webcam
cap = cv2.VideoCapture(0)

#To allow camera the time to open up completely
time.sleep(2)

#Start capturing background
bg = 0

#1 second = 60 frames
for i in range(0, 60):
    ret, bg = cap.read()

#Flipping the background
bg = np.flip(bg, axis = 1)

#Reading the camera frames as long as it is on - infinite loop
while(cap.isOpened()):
    frame=cv2.resize(640,480)
    image=cv2.resize(640,480)
    ret,img=cap.read()
    if not ret:
        break
    img=np.flip(img,axis=1)
    
    #Converting RGB to HSV (Hue, saturation, value)
    
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Generating mask to detect red colour
    lower_black = np.array([0, 120, 50])
    upper_black = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_black, upper_black)
    
    lower_black = np.array([0, 170, 50])
    upper_black = np.array([10, 45, 155])
    mask2 = cv2.inRange(hsv, lower_black, upper_black)
    mask1=mask1+mask2
    #Open & expand the image where there is mask1
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))

    #Selecting only the part which does not have mask1
    mask2=cv2.bitwise_not(mask1)

    #Keeping only the part of the images without the red colour
    res1=cv2.bitwise_and(img, img,mask=mask2)
    #Keeping only the part of the images with the red colour
    res2=cv2.bitwise_and(bg, bg,mask=mask1)
    final_output=cv2.addWeighted(res1, 1, res2, 1, 0)
    output_file.write(final_output)
    cv2.imshow("I'm in Bangkok", final_output)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
    