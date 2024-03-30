"""
Project is running with this .py file
"""

import cv2

import MainModule as mm
import DumbbellModule as dm
import PostureModule as pm
import ArmModule as arm

cap = cv2.VideoCapture(0) # capture the video, 0 stands for camera input.
detector = mm.poseDetector() # create the class

while True:
    success, img = cap.read() # read the captured video
    img = cv2.resize(img, (1280, 720)) # resize
    # img = cv2.flip(img, 1)

    img = detector.findPose(img, True) # capture the body
    lmList = detector.findPosition(img, True) # get coordinates
    
    fps = cap.get(cv2.CAP_PROP_FPS) # get FPS value
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0), 5)
            
    cv2.imshow("Exercise", img) # show the captured cam
        
    if cv2.waitKey(5) & 0xFF == ord('q'): # exit
        break  
        
    if cv2.waitKey(5) & 0xFF == ord('1'): # start posture mode
        cv2.destroyAllWindows()
        pm.posture()
        cap = cv2.VideoCapture(0)
     
    if cv2.waitKey(5) & 0xFF == ord('2'): # start dumbbell mode
        cv2.destroyAllWindows()
        dm.dumbbell() 
        cap = cv2.VideoCapture(0)
     
    if cv2.waitKey(5) & 0xFF == ord('3'): # start arm exercise mode
        cv2.destroyAllWindows()
        arm.armposture() 
        cap = cv2.VideoCapture(0)
        
cap.release() # end capturing
cv2.destroyAllWindows()
