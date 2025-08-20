# Vehicle Detection Project
#import opencv and numpy

!pip install opencv-python
!pip install opencv-contrib-python

import cv2
import numpy as np

#web camera
cap = cv2.VideoCapture('video.mp4')

#min height and width of the rectangle for detection
min_width_rect = 80
min_height_rect = 80


#count line position at pixel height
count_line_position = 550

#Initialize Substructor MOG(mixtures of gaussian) separate moving objects from static background
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

#function for handling center and detecting the vechiles around the center
def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

detect = []  # a list to store center coordinates of detected vehicles
offset = 6    # Allowable error between pixel
counter = 0



while True:
    ret,frame1 = cap.read()  # reads each frame from video
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)  # coverts the color from BGR to grey
    blur = cv2.GaussianBlur(grey,(3,3),5)

    # applying on each frame
    img_sub = algo.apply(blur)    #apply algo on blur
    dilat = cv2.dilate(img_sub,np.ones((5,5)))   #dilates the detected vehicles 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))   # it creates ellipse shaped kernel for detection 
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  #finds counter from binrary image of dilatada

    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,127,0),3)
    
    #for rectangle on vehicles
    for (i,c) in enumerate(counterShape):
        (x,y,w,h) =cv2.boundingRect(c)
        validate_counter = (w >= min_width_rect) and (h >= min_height_rect)  #checks for vehicle meets min height and width
        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2) #creates a rectangle on detected on vehicle in green color
        cv2.putText(frame1, "Vehicle"+str(counter),(x,y-20), cv2.FONT_HERSHEY_TRIPLEX,1,(255,244,0),2) # put text on the detected vehicle

        center = center_handle(x,y,w,h) #calculate center of vehicle
        detect.append(center) # append list of detected vehicle
        cv2.circle(frame1,center,4,(0,0,255),-1) # creates the small red circle on detected vehicle
        
        for (x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):   # checks for the center of vehicle in under offset of center line
                counter+=1

                cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,127,0),3)
                detect.remove((x,y)) # removes the vehicle from the list ones it get detected
                
                print("Vehicle Counter :"+str(counter)) #gives count of detected vehicle on console
    
    cv2.putText(frame1,"VEHICLE COUNTER :"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
    
    
    
    #cv2.imshow('Dectector',dilatada)
    cv2.imshow('Video Original',frame1) #show the output of the processed video

    if cv2.waitKey(1) == 13: #press enter key for exist from loop
        break

cv2.destroyAllWindows()
cap.release()


