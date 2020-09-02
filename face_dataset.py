# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:34:32 2020

@author: Nitesh_Mishra
"""

import cv2
import os
#print (cv2.__version__)
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier(r'C:\Users\acer\anaconda3\lib\site-packages\cv2/data/haarcascade_frontalface_default.xml')
# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
print("In process")
while(True):
    ret, img = cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)   
        
        count += 1
        # Save the captured image into the datasets folder
        
        cv2.imwrite(r"C:\Users\acer\Desktop\FacialRecognitionProject\dataset/User." + str(face_id) + '.' +  
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    # if k == 27:
    #     print("3")
    if count >= 30: # Take 30 face sample and stop video 
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()




