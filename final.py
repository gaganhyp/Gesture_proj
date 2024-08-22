import cv2
import time
import math
import autopy
import cvzone
import numpy as np
import mediapipe as mp
from time import sleep
from pynput.keyboard import Controller
from cvzone.HandTrackingModule import HandDetector



cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
finalText = ""
wCam, hCam = 640, 480
frameR = 1 # Frame Reduction
smoothening = 7
wScr, hScr = autopy.screen.size()



pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

keyboard = Controller()

##########functions//classes

def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img, (button.pos[0], button.pos[1], button.size[0], button.size[1]),
                          20, rt=0)
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img

class handD():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        #print("calle dkjbsdjcsdbcsddbbvk")
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):

            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text


buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))
detector_m = handD(),
while True:
    success, img = cap.read()
    success_m, img_m = success, img
    img = detector.findHands(img)
    lmList, bboxInfo = detector.findPosition(img)
    img_m = detector_m.findHands(img)
    lmList_m, bbox_m = detector_m.findPosition(img_m)
    img = drawAll(img, buttonList)

    if len(lmList_m) != 0:
        x1_m, y1_m = lmList_m[8][1:]
        x2_m, y2_m = lmList_m[12][1:]
        # print(x1, y1, x2, y2)
    
    # 3. Check which fingers are up
        fingers_m = detector_m.fingersUp()
        print(fingers_m)
        #cv2.rectangle(img_m, (frameR, frameR), (wCam - frameR, hCam - frameR),(255, 0, 255), 2)
        # 4. Only Index Finger : Moving Mode
        if fingers_m[1] == 1 and fingers_m[2] == 0:
            # 5. Convert Coordinates
            x3_m = np.interp(x1_m, (frameR, wCam - frameR), (0, wScr))
            y3_m = np.interp(y1_m, (frameR, hCam - frameR), (0, hScr))
            # 6. Smoothen Values
            clocX = plocX + (x3_m - plocX) / smoothening
            clocY = plocY + (y3_m - plocY) / smoothening
        
            # 7. Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img_m, (x1_m, y1_m), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
            
        # 8. Both Index and middle fingers are up : Clicking Mode
        if fingers_m[1] == 1 and fingers_m[2] == 1:
            # 9. Find distance between fingers
            length_m, img_m, lineInfo_m = detector_m.findDistance(8, 12, img_m)
            #print(length_m)
            # 10. Click mouse if distance short
            if length_m < 30:
                cv2.circle(img_m, (lineInfo_m[4], lineInfo_m[5]),
                15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    if lmList:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 20, y + 65),
                            cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                l, _, _ = detector.findDistance(8, 12, img, draw=False)
                #print(l)

                ## when clicked
                if l < 30:
                    keyboard.press(button.text)
                    #cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    #cv2.putText(img, button.text, (x + 20, y + 65),
                               # cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    finalText += button.text
                    sleep(0.15)
    
    
    cv2.rectangle(img, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 430),
                cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)