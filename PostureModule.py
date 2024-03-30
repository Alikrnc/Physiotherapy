import cv2
import playsound

import MainModule as mm

def posture(): # Wrapped the codes in a function to call it from the Exercise module
    
    cap = cv2.VideoCapture(0) # capture the video, 0 stands for camera input.
    detector = mm.poseDetector() # create class
    good_frames = 0
    bad_frames = 0
    
    while True:
        success, img = cap.read() # read the captured video
        img = cv2.resize(img, (1280, 720)) # resize
        # img = cv2.flip(img, 1)
        
        img = detector.findPose(img, False) # Set draw = False because there is drawing in the recognize body and angle finding module.
        lmList = detector.findPosition(img, False) # get landmark coordinates
        fps = cap.get(cv2.CAP_PROP_FPS) # get the FPS value
        
        if len(lmList) != 0:
            
            # Ear - Shoulder (Right) --- 8 ear, 12 shoulder
            neck_angle = detector.findAngle2P(img, 8, 12) # find shoulder - ear (neck) angle
            
            # Shoulder - Hip (Right) --- 12 shoulder, 24 hip
            torso_angle = detector.findAngle2P(img, 12, 24) # find hip - shoulder (waist) angle
            
            # Ear - Shoulder (Left) --- 7 ear, 11 shoulder
            # neck_angle = detector.findAngle2P(img, 7, 11)
            
            # Shoulder - Hip (Left) --- 11 shoulder, 23 hip
            # torso_angle = detector.findAngle2P(img, 11, 23)   
            
            if neck_angle < 35 and torso_angle < 10: # If the angles are below the specified level, ...
                # ...that is, if the posture is correct, the increase is in the number of fps.
                bad_frames = 0
                good_frames += 1
                
            else: # if the posture is bad
                bad_frames += 1
                good_frames = 0
                
            good_time = (1 / fps) * good_frames # Since the good_frames value increases with fps, ...
            # ...if we divide it by the number of fps, we get the seconds
            bad_time = (1 / fps) * bad_frames   
            alarm_time = bad_time % 5 # Set alarm time equal to 0 every 5th seconds
            
            if good_time > 0:
                time_string = "Good Posture Time: " + str(round(good_time, 1)) + "s"
                cv2.putText(img, time_string, (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # write the time
                  
            if bad_time > 0:
                time_string = "Bad Posture Time: " + str(round(bad_time, 1)) + "s"
                cv2.putText(img, time_string, (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)         

            if alarm_time == 0 and bad_time != 0: # if alarm time is 0 and bad posture time is not 0
                playsound.playsound("./audio.mp3", False) # play alert sound, here the False value plays...
                # ...the sound in the background, if the value is True the audio file would run...
                # ...in the foreground and other processes would stop.
                
    
        cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                    (255, 0, 0), 5) # write the FPS values
        
        cv2.imshow("Posture Module", img) # display the image
        if cv2.waitKey(5) & 0xFF == ord("q"): # quit
            break
        
    cap.release() # end capture
    cv2.destroyAllWindows() # close windows
    
if __name__ == "__main__":
    posture()