import cv2
import numpy as np

import MainModule as mm

def dumbbell():  # Wrapped the codes in a function to call it from the Exercise module
    
    cap = cv2.VideoCapture(0) # capture the video, 0 stands for camera input.
    detector = mm.poseDetector() # create the class
    count = 0 # counter that counts weight lifts
    truth = 0 # used to fix counter (If I raise my arm without fully lowering it, the counter won't increase)
    
    while True:
        success, img = cap.read() # read the captured video
        img = cv2.resize(img, (1280, 720)) # resize
        # img = cv2.flip(img, 1)
        
        img = detector.findPose(img, False) # capture the body
        lmList = detector.findPosition(img, False) # get coordinates
        
        if len(lmList) != 0:
            
            # Right arm
            # 12 shoulder, 14 elbow, 16 wrist
            angle = detector.findAngle3P(img, 12, 14, 16) # calculate the shoulder - elbow - wrist angle
            
            # Left arm
            # 11 shoulder, 13 elbow, 15 wrist
            # angle = detector.findAngle(img, 11, 13, 15, False)
            
            per = np.interp(angle, (210, 310), (0, 100)) # The angle is 210 degrees when the arm is straight and 310 degrees when it is bent...
            # ...With this function, place the angle range 210 - 310 in the range of 0 - 100.
            bar = np.interp(angle, (210, 310), (650, 100))
            
            color = (255, 0, 255)
            if per == 100: # increase count at 100%
                color = (0, 255, 0)
                if truth == 0: # count won't increase again if the per is not 0%
                    count += 0.5
                    truth = 1
                    
            if per == 0:
                color = (0, 255, 0)
                if truth == 1:
                    count += 0.5
                    truth = 0
                    
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3) # Draw outside of the indicator
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED) # Fill the inside of the indicator according to the percentage
            cv2.putText(img, f"{int(per)} %", (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                        color, 4)
            
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED) # Background of the counter
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                        (255, 0, 0), 15) # counter
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                    (255, 0, 0), 5) # Write the FPS value
            
        cv2.imshow("dumbbell Module", img)    
        if cv2.waitKey(5) & 0xFF == ord('q'): # q exit
            break 
        if cv2.waitKey(5) & 0xFF == ord('r'): # r reset counter         
            count = 0
            
    cap.release() # end capturing
    cv2.destroyAllWindows() # close windows
                                    
if __name__ == "__main__":
    dumbbell()
