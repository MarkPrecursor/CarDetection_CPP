# -*- coding: utf-8 -*-
"""
*创建于2016.3.13
*作者：Mark
"""
import cv2
import time

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4,
                        minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def Detect_and_Draw(img):
    #读取分类器
    cascade = cv2.CascadeClassifier("LBPcascade.xml")
    start = time.clock()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = detect(gray, cascade)
    vis = img.copy()
    draw_rects(vis, rects, (0, 255, 0))
    end = time.clock()
    print("The function run time is : %.03f seconds" %(end-start))
    cv2.imshow('Result', vis)

def Main():
    img=cv2.imread('111.jpg')
    cv2.namedWindow("Result")
    Detect_and_Draw(img)
    cv2.waitKey (0)
    cv2.destroyAllWindows()

Main()