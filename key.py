from cvzone.HandTrackingModule import HandDetector
import cv2
import mediapipe as mp
from time import sleep
import numpy as np
import cvzone
from pynput.keyboard import Controller
import autopy

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
wScr, hScr = autopy.screen.size()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


detector = HandDetector(detectionCon=0.8, maxHands=1)
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


buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    #lmList, bboxInfo = detector.findPosition(img)
    img = drawAll(img, buttonList)

    #print(hands)
    if hands:
        a = hands[0]["center"]
        #print(a[0])
        #autopy.mouse.move()
        #print(hands)
        autopy.mouse.move(a[0], a[1])
        hand1 = hands[0]
        lmList = hand1["lmList"]  # List of 21 Landmark points
        bbox = hand1["bbox"]  # Bounding box info x,y,w,h

        if lmList:
            for button in buttonList:
                x, y = button.pos
                w, h = button.size
                #print("l",lmList[8][0] , x, lmList[8][1] , y)
                if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65),cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    l, n, m = detector.findDistance(lmList[8][:2],lmList[8][:2],img)
                    
                    print(l)
                    #print("n",n)
                    #print(detector.findDistance(lmList[8][:2],lmList[8][:2],img))
                    if l < 30:
                        keyboard.press(button.text)

                        cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 20, y + 65),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                        finalText += button.text
                        sleep(0.15)



    cv2.rectangle(img, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 430),
                cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)