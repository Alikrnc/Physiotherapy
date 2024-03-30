import cv2
import numpy as np

import MainModule as mm

def armposture():  # Wrapped the codes in a function to call it from the Exercise module
    
    cap = cv2.VideoCapture(0) # capture the video, 0 stands for camera input.
    detector = mm.poseDetector() # create the class
    count = 0 # counter
    truth = 0 # used to fix counter (If I raise my arm without fully lowering it, the counter won't increase)
    up_frames = 0 # arm raising fps
    down_frames = 0 # arm lowering fps
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    up_time = (1 / fps) * up_frames # arm raise time
    down_time = (1 / fps) * down_frames # arm drop time

    while True:
        success, img = cap.read() # read the captured video
        img = cv2.resize(img, (1280, 720)) # resize
        # img = cv2.flip(img, 1) 
        
        img = detector.findPose(img, False) # detects the pose
        lmList = detector.findPosition(img, False) # get the coordinates of the landmarks
        fps = cap.get(cv2.CAP_PROP_FPS)

        if len(lmList) != 0: # if landmarks exist
            
            # Right Arm
            # 12 shoulder, 14 elbow, 16 wrist
            x1, y1 = lmList[12][1:] # shoulder coordinates
            x2, y2 = lmList[14][1:] # elbow coordinates
            x3, y3 = lmList[16][1:] # wrist coordinates
            angle = detector.findAngle3P(img, 12, 14, 16, False) # I will manually create drawing so draw = False
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle - 180)), (x2 - 50, y2 + 50), # In order to show the angle between 0 and 90 degrees we need to subtract 180 degrees...
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2) # ...because the angles are calculated between 180 and 270 degrees.                     
           
            # Left Arm
            # 11 shoulder, 13 elbow, 15 wrist
            # angle = detector.findAngle(img, 11, 13, 15, False)
            
            per = np.interp(angle, (180, 270), (0, 100)) # The open arm is 180 degrees and the raised arm is 270 degrees...
            # ...with a difference of 90 degrees, place it in the range 0 - 100 as a percentage.
            
            bar = np.interp(angle, (180, 270), (650, 100)) # Limit the bar to the percentage.
            angle = angle - 180 # I reduced the angle from 180 - 270 degrees to 0 - 90 for ease of operation.
            
            color = (255, 0, 255)
            if angle > 86 and angle < 94: # arm raise status, I left +-4 margin of error
                up_frames += 1
                down_frames = 0
                up_time = (1 / fps) * up_frames
                down_time = (1 / fps) * down_frames
                color = (0, 255, 0)
                if truth == 0 and up_time > 3: # Increase the count after 3 sec
                    up_frames = 0
                    count += 0.5
                    truth = 1
                    
            if angle > -4 and angle < 4: # arm lowering status, I left +-4 margin of error
                up_frames += 0
                down_frames += 1
                up_time = (1 / fps) * up_frames
                down_time = (1 / fps) * down_frames
                color = (0, 255, 0)
                if truth == 1 and down_time > 3: # Increase count after 3 sec
                    down_frames = 0
                    count += 0.5
                    truth = 0
            
            if up_time > 0:
                time_string = "Up Posture Time: " + str(round(up_time, 1)) + "s"
                cv2.putText(img, time_string, (400, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)# Write the duration in case of lowering the arm
                  
            if down_time > 0:
                time_string = "Down Posture Time: " + str(round(down_time, 1)) + "s"
                cv2.putText(img, time_string, (400, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) # Write the duration in case of raising the arm.
                    
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3) # Draw outside of the indicator
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED) # Fill the inside of the indicator according to the angle
            cv2.putText(img, f"{int(per)} %", (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                        color, 4) # write percentage
            
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED) # Background of the counter
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                        (255, 0, 0), 15) # counter
            
        cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                    (255, 0, 0), 5) # Write the FPS value
            
        cv2.imshow("Arm Module", img)    
        if cv2.waitKey(5) & 0xFF == ord('q'): # q for quit
            break 
        if cv2.waitKey(5) & 0xFF == ord('r'): # r for reset counter
            count = 0
            
    cap.release() # end capturing
    cv2.destroyAllWindows() # close windows
                                    
if __name__ == "__main__":
    armposture()
