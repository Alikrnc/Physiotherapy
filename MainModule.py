import cv2
import mediapipe as mp
import math

class poseDetector():
    
    def __init__(self, mode = False, upBody = False, complexity = 1, smooth = True,
                 detectionCon = 0.5, trackCon = 0.5):
        
        """
        mode: In case of False, it defines that the incoming input is in the form of a video stream.  
         
        upBody: Upper body recognition only. At false value, it recognizes the whole body.
        
        complexity: This can take values 0, 1 and 2. As the value increases, the accuracy of recognizing
        body landmarks will increase, but the processing load will also increase.
        
        smooth: If set to True, it will filter body landmarks to reduce jitter.
        
        detectionCon: The lower limit value required for person recognition to be considered successful.
        
        trackCon: If the assigned lower limit value is exceeded, it will switch to body landmark
        recognition mode; if it is below the value, it will remain in person recognition mode.
        """
        
        self.mode = mode
        self.upBody = upBody
        self.complexity = complexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpDraw = mp.solutions.drawing_utils # Assigning the mediapipe functions to a variable.
        self.mpPose = mp.solutions.pose
        
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.complexity, self.smooth,
                                     self.detectionCon, self.trackCon) # We will perform recognition with the pose() function.
        
    def findPose(self, img, draw = True): # Function that recognizes the body.
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # We converted BGR format to RGB format. This process is not necessary for video, but it is necessary if we are using images.
        self.results = self.pose.process(imgRGB) # The process() function performs recognition by processing the image.
        if self.results.pose_landmarks: # Do it if the image is recognized and there are body markings.
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS) # Draw landmarks.
        return img
    
    def findPosition(self, img, draw = True): # Get landmark coordinates.
        self.lmList = []
        if self.results.pose_landmarks: 
            for id, lm in enumerate(self.results.pose_landmarks.landmark): # Number the results.pose_landmarks.landmark variable, ...
                # ...assign the number to the id variable, the landmark to the lm variable and loop.
                h, w, c = img.shape # Get the height and width values of the image.
                cx, cy = int(lm.x * w), int(lm.y * h) # Multiply the x coordinate of the lm.x landmark by the width of the image, ...
                # ...similarly multiply the y coordinate by the height and get the coordinates on the image.
                self.lmList.append([id, cx, cy]) # add id and coordinates to the list.
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED) # Place a filled circle on the landmark.
        return self.lmList

    def findAngle3P(self, img, p1, p2, p3, draw = True): # 3 point angle finder
        # Here, the variables p1, p2, p3 are the numbers given to the landmarks in the human model used by mediapipe.
        x1, y1 = self.lmList[p1][1:] # get the landmark coordinates.
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2,) -
                             math.atan2(y1 - y2, x1 - x2)) # calculate the angle via the formula.
        
        if angle < 0:
            angle += 360 # If the angle is negative, we add 360.
            
        if draw: # If the draw value is True, connect the coordinates with a line and place a filled circle at each point.
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2) # write the angle at the mid points
        return angle
    
    def findAngle2P(self, img, p1, p2, draw = True, color = (0, 0, 255)): # 2-point angle finding function, only the angle finding formula is different.
        
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        
        radian = math.acos((y2 - y1)*(-y1) / 
                           (math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
        angle = int(180/math.pi)*radian
        angle = 180 - angle
                    
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, color, cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, color, 2)
            cv2.circle(img, (x2, y2), 10, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, color, 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        return angle

def mainModule(): # Function to test the codes in this file.
        cap = cv2.VideoCapture(0) # capture the camera
        detector = poseDetector() # create class
        
        while True:
            success, img = cap.read() # read the captured video
            img = cv2.resize(img, (1280, 720)) # resize
            # img = cv2.flip(img, 1) # Depending on the camera type, the image may appear reverse, so using this wil help
            img = detector.findPose(img) # detect the pose
            lmList = detector.findPosition(img, draw = False) # get the landmark coordinates
            if len(lmList) != 0:
                print(lmList[14]) # place a circle on the right elbow
                cv2.circle(img, (lmList[14][1], lmList[14][2]), 15,
                           (0, 0, 255), cv2.FILLED)
                
            
            fps = cap.get(cv2.CAP_PROP_FPS) # get FPS value
            
            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3) # Write FPS value on screen.
            
            
            cv2.imshow("Main Module", img) # Display the image.
            if cv2.waitKey(5) & 0xFF == ord('q'): # q for quit.
                cap.release() # end capturing
                cv2.destroyAllWindows() # close all windows
            
if __name__ == "__main__":
    mainModule()