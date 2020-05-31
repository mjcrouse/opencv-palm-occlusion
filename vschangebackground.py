import numpy as np
import cv2 as cv
import imutils
import time
import webcamvideostream

cap = webcamvideostream.WebcamVideoStream().start()
time.sleep(5.0)
width = 1280
height = 720
#cap.set(3, width)
#cap.set(4, height)

background = None
aWeight = 0.5
num_frames = 0
#top, right, bottom, left = 10, 680, 700, 1200
#[ylow, crlow, cblow, yhigh, crhigh, cbhigh]
thresholds = [0,0,0,0,0,0]
ch1, ch2, ch3 = None, None, None
sample1, sample2, sample3, sample4 = None, None, None, None
sampled = [False,False]

def run_avg(image, aWeight):
    global background

    if background is None:
        background = image.copy().astype("float")
        return
    cv.accumulateWeighted(image, background, aWeight, mask=None)

# def segment(image, threshold=25):
#     global background
#     diff = cv.absdiff(background.astype("uint8"), image)

#     thresholded = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)[1]

#     cnts, _ = cv.findContours(thresholded.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#     if len(cnts) == 0:
#         return
#     else:
#         segmented = max(cnts, key=cv.contourArea)
#         return (thresholded, segmented)

def drawSampleRec(image):
    sampling = image.copy()
    cv.rectangle(sampling, (250, 275), (225, 300), (200, 100, 100))
    cv.rectangle(sampling, (250, 350), (225, 375), (200, 100, 200))
    return sampling

#rename to ch1-3? needs finetuning, seems to be working better for me with HSV (possible 0-255 V range)
def getSkinColor(image):
    global sample1, sample2, sample3, sample4, thresholds
    lthreshoffset = 25
    hthreshoffset = 35
    if sample1 is None:
        sample1 = image[275:300, 225:250]
        sample2 = image[350:375, 225:250]
        sample1 = cv.mean(sample1)
        sample2 = cv.mean(sample2)
    else:
        sample3 = image[275:300, 225:250]
        cv.imshow("sample3", sample3)
        sample4 = image[350:375, 225:250]
        sample3 = cv.mean(sample3)
        sample4 = cv.mean(sample4)
        ch1low = min(sample1[0], sample2[0], sample3[0], sample4[0]) - lthreshoffset
        ch2low = min(sample1[1], sample2[1], sample3[1], sample4[1]) - lthreshoffset
        ch3low = 0 #min(sample1[2], sample2[2], sample3[2], sample4[2]) - lthreshoffset
        ch1high = max(sample1[0], sample2[0], sample3[0], sample4[0]) + hthreshoffset
        ch2high = max(sample1[1], sample2[1], sample3[1], sample4[1]) + hthreshoffset
        ch3high = 255 #max(sample1[2], sample2[2], sample3[2], sample4[2]) + hthreshoffset
        thresholds = [ch1low, ch2low, ch3low, ch1high, ch2high, ch3high]   

def splitChannels(image):
    channels = cv.split(image)
    ch1 = channels[0]
    ch2 = channels[1]
    ch3 = channels[2]

def removeBackground(image):
    global background, width, height
    # os = 10
    # for i in range(0,height-1):
    #     for j in range(0,width-1):
    #         fp = image[i,j]
    #         bp = background[i,j]
    #         if fp >= bp - os and fp <= bp + os:
    #             image[i,j] = 0
    #         else:
    #             image[i,j] = 255

while True:
    frame = cap.read()
    frame = cv.flip(frame, 1)
    clone = frame.copy()
    height, width = frame.shape[:2]
    #roi = frame[top:bottom, right:left]

    # bg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # bg = cv.GaussianBlur(bg, (7, 7), 0)
    ch1, ch2, ch3 = cv.split(frame)
    if num_frames == 0:
        #bg = (cv.GaussianBlur(frame, (11, 11), 0)).astype("float")
        bg1 = (cv.GaussianBlur(ch1, (11, 11), 0)).astype("float")
        bg2 = (cv.GaussianBlur(ch2, (11, 11), 0)).astype("float")
        bg3 = (cv.GaussianBlur(ch3, (11, 11), 0)).astype("float")
    if num_frames < 60:
        #cv.accumulateWeighted(frame,bg,0.5)
        cv.accumulateWeighted(ch1,bg1,0.5)
        cv.accumulateWeighted(ch2,bg2,0.5)
        cv.accumulateWeighted(ch3,bg3,0.5)
    # diff=cv.absdiff(frame,bg.astype("uint8"))
    # diff=cv.cvtColor(diff,cv.COLOR_BGR2GRAY)
    # thre,diff=cv.threshold(diff,25,255,cv.THRESH_BINARY)
    # cv.imshow("j",diff)
    kernel = np.ones((3,3),np.uint8)
    diff1=cv.absdiff(ch1,bg1.astype("uint8"))
    thre1,diff1=cv.threshold(diff1,25,255,cv.THRESH_BINARY)
    diff1 = cv.morphologyEx(diff1, cv.MORPH_CLOSE, kernel)
    diff2=cv.absdiff(ch2,bg2.astype("uint8"))
    thre2,diff2=cv.threshold(diff2,25,255,cv.THRESH_BINARY)
    diff2 = cv.morphologyEx(diff2, cv.MORPH_CLOSE, kernel)
    diff3=cv.absdiff(ch3,bg3.astype("uint8"))
    thre3,diff3=cv.threshold(diff3,25,255,cv.THRESH_BINARY)
    diff3 = cv.morphologyEx(diff3, cv.MORPH_CLOSE, kernel)
    diff2 = cv.add(diff1, diff2)
    diff3 = cv.add(diff2, diff3)
    #cv.imshow("j1",diff3)
    
 
    
    face = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #frame_gray = cv.equalizeHist(frame_gray)
    faces = face.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        cv.rectangle(diff3, (x,y), (x+w, y+h), (0, 0, 0), -2)
    #cv.imshow("faces", frame_gray)
    

    masked = cv.bitwise_and(frame, frame, mask = diff3)
    cv.imshow("masked", masked)
    # else:
    #     hand = segment(gray)

    #     if hand is not None:
    #         (thresholded, segmented) = hand
    #         cv.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
    #         cv.imshow("Thresholded", thresholded)

    getSamples = drawSampleRec(clone)
    # cv.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

    #convert to YCrCb
    ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    frame_threshold = cv.inRange(hsv, tuple(thresholds[0:3]), tuple(thresholds[3:6]))
    
    #subtract background - won't work
    #backSub = cv.createBackgroundSubtractorMOG2()
    #fgMask = backSub.apply(hcv)

    # background = np.zeros((768, 1360, 3), np.uint8)

    # cv.imshow('hsv', hsv)
    # cv.imshow('frame', frame)
    
    if sample3 is None:
        cv.imshow("Video Feed", getSamples)
    else:
        cv.destroyWindow("Video Feed")
        cv.imshow("Other Video Feed", clone)
        #cv.imshow("skin", frame_threshold)

    num_frames += 1

    key = cv.waitKey(1) & 0xFF

    #end program if q or Esc pressed
    if key == ord('q'):
        break
    if key == 27:
        break
    
    #recalc background on r key - needs to be rewritten
    # if key == ord('r'):
    #     background = None
    #     run_avg(bg, aWeight)

    #extract colour from sample rectangles on f key
    if key == ord('f'):
        getSkinColor(hsv)
        print(thresholds)
        print(tuple(thresholds[0:3]), tuple(thresholds[3:6]))

cap.release()
cv.destroyAllWindows()

# background = np.zeros((768, 1360, 3), np.uint8)
