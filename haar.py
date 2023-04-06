#importing librarys
import cv2

#loading xml files using cascade classifier
face_cascade =cv2.CascadeClassifier('./haarcascade_frontalface_default (1).xml')
eye_cascade=cv2.CascadeClassifier('C:/Users/Dell/Desktop/face_detection/haarcascade_eye.xml')
smile_cascade= cv2.CascadeClassifier('C:/Users/Dell/Desktop/face_detection/haarcascade_smile.xml')

cap=cv2.VideoCapture(0)

while True:
    ret,frame= cap.read()
   
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=4)
    
    for(x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
        eye= eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        # for(x,y,w,h) in eye:
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            
        smile=smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=0)
        for(x,y,w,h) in smile:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            
            
            
        
        
        
        
        
        
    cv2.imshow('window',frame)
    
    if cv2.waitKey(1) & 0xFF==27:
        break    
        
    
    
