import time
import cv2
import cvzone
import autopy
import numpy as np
import moduleh as htm
import mediapipe as mp
from time import sleep
from pynput.keyboard import Controller
from cvzone.HandTrackingModule import HandDetector
from google.protobuf.json_format import MessageToDict

##########################
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 7
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

flag = "none"



# Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
mouse_detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
key_detector = HandDetector(detectionCon=0.8)
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
finalText = ""

keyboard = Controller()

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

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

fg = 0

def switch_mode():
    if fg == 0:
        flag = "key"
        fg = fg + 1
        pass
    else:
        pass


buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

while True:
    # Read video frame by frame
    success, img = cap.read()
    final_img = img
    """key_img = key_detector.findHands(img)
    mouse_img = mouse_detector.findHands(img)
    m_lmList, m_bboxInfo = mouse_detector.findPosition(mouse_img)
    k_lmList, k_bboxInfo = key_detector.findPosition(key_img)"""
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:    
        # Both Hands are present in image(frame)
        if len(results.multi_handedness) == 2:
            if flag == "none" or  flag == "key" and fg == 0:
                flag = "mouse"
                #sleep(0.5)
                pass
            elif(flag == "mouse" and fg == 0):
                flag = "key"
                pass
            cv2.putText(final_img, 'mode change', (250, 50),cv2.FONT_HERSHEY_COMPLEX,0.9, (0, 255, 0), 2)

    if flag == "mouse":
        mouse_img = cv2.flip(img, 1)
        mouse_img = mouse_detector.findHands(img)
        m_lmList, m_bboxInfo = mouse_detector.findPosition(mouse_img)
        if len(m_lmList) != 0:
            x1, y1 = m_lmList[8][1:]
            x2, y2 = m_lmList[12][1:]
            fingers = mouse_detector.fingersUp()
            # print(fingers)
            cv2.rectangle(mouse_img, (frameR, frameR), (wCam - frameR, hCam - frameR),
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
                cv2.circle(mouse_img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # 8. Both Index and middle fingers are up : Clicking Mode
            if fingers[1] == 1 and fingers[2] == 1:
                length, mouse_img, lineInfo = mouse_detector.findDistance(8, 12, mouse_img)
                print(length)
                # 10. Click mouse if distance short
                if length < 40:
                    cv2.circle(mouse_img, (lineInfo[4], lineInfo[5]),15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()
        
        final_img = mouse_img 
    if flag == "key":
        key_img = key_detector.findHands(img)
        k_lmList, k_bboxInfo = key_detector.findPosition(key_img)
        key_img = drawAll(key_img, buttonList)
        if k_lmList:
            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                if x < k_lmList[8][0] < x + w and y < k_lmList[8][1] < y + h:
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                    cv2.putText(key_img, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    l, _, _ = key_detector.findDistance(8, 12, key_img, draw=False)
                    ## when clicked
                    if l < 30:
                        keyboard.press(button.text)
                        cv2.rectangle(key_img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                        cv2.putText(key_img, button.text, (x + 20, y + 65),cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                        finalText += button.text
                        sleep(0.15)

        cv2.rectangle(key_img, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)
        cv2.putText(key_img, finalText, (60, 430),cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)
        final_img = key_img  


    # Display Video and when 'q'
    # is entered, destroy the window
    cv2.imshow('Image', final_img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
