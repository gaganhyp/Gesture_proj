from cvzone.HandTrackingModule import HandDetector
import cv2
import mediapipe as mp
import time
import autopy
import numpy as np


pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 7
detector = HandDetector(detectionCon=0.8, maxHands=1)
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
wScr, hScr = autopy.screen.size()
print(wScr)
cTime = 0
pTime = 0

while True:
    success, img=cap.read()
    #imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    hand, img = detector.findHands(img)
    if hand:
        #print(hand)
        hand1 = hand[0]
        lmList = hand1["lmList"]  # List of 21 Landmark points
        bbox = hand1["bbox"] 
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            # print(x1, y1, x2, y2)
        
        # 3. Check which fingers are up
        fingers = detector.fingersUp(hand[0])
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
        (255, 0, 255), 2)
        # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
        
            # 7. Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

    
                

            #mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    #cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(255,0,255),2)

    cv2.imshow("img",img)
    cv2.waitKey(2)


   