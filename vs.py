import numpy as np
import cv2 as cv
import imutils

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

background = None
aWeight = 0.5
num_frames = 0
top, right, bottom, left = 10, 680, 700, 1200
#[ylow, crlow, cblow, yhigh, crhigh, cbhigh]
thresholds = [0,0,0,0,0,0]
y, cr, cb = None, None, None
channels, sample1, sample2, sample3, sample4 = None, None, None, None, None
sampled = [False,False]

def run_avg(image, aWeight):
    global background

    if background is None:
        background = image.copy().astype("float")
        return
    cv.accumulateWeighted(image, background, aWeight, mask=None)

def segment(image, threshold=25):
    global background
    diff = cv.absdiff(background.astype("uint8"), image)

    thresholded = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)[1]

    cnts, _ = cv.findContours(thresholded.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv.contourArea)
        return (thresholded, segmented)

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
        ylow = min(sample1[0], sample2[0], sample3[0], sample4[0]) - lthreshoffset
        crlow = min(sample1[1], sample2[1], sample3[1], sample4[1]) - lthreshoffset
        cblow = 0 #min(sample1[2], sample2[2], sample3[2], sample4[2]) - lthreshoffset
        yhigh = max(sample1[0], sample2[0], sample3[0], sample4[0]) + hthreshoffset
        crhigh = max(sample1[1], sample2[1], sample3[1], sample4[1]) + hthreshoffset
        cbhigh = 255 #max(sample1[2], sample2[2], sample3[2], sample4[2]) + hthreshoffset
        thresholds = [ylow, crlow, cblow, yhigh, crhigh, cbhigh]

        

def splitChannels(image):
    global y, cr, cb, channels
    channels = cv.split(image, channels)
    y = channels[0]
    cr = channels[1]
    cb = channels[2]


while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    clone = frame.copy()
    height, width = frame.shape[:2]
    roi = frame[top:bottom, right:left]

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 0)

    if num_frames < 30:
        run_avg(gray, aWeight)
        
    else:
        hand = segment(gray)

        if hand is not None:
            (thresholded, segmented) = hand
            cv.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
            cv.imshow("Thresholded", thresholded)

    getSamples = drawSampleRec(clone)
    # cv.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)


    
    #convert to YCrCb
    ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    #channels = cv.split(frame, channels)

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
        cv.imshow("skin", frame_threshold)

    num_frames += 1

    key = cv.waitKey(1) & 0xFF

    #end program if q or Esc pressed
    if key == ord('q'):
        break
    if key == 27:
        break
    
    #recalc background on r key
    if key == ord('r'):
        background = None
        run_avg(gray, aWeight)

    #extract colour from sample rectangles on f key
    if key == ord('f'):
        getSkinColor(hsv)
        print(thresholds)
        print(tuple(thresholds[0:3]), tuple(thresholds[3:6]))

cap.release()
cv.destroyAllWindows()

# class BackgroundRemover:

#     def __init__(self, background):
#         self.background = np.zeros((768,1360,3), np.uint8)

#     def removebackground(stream):
#         cv.cvtColor(stream, background, CV_BGR2GRAY)

# background = np.zeros((768, 1360, 3), np.uint8)
# def removebackground(stream):
#     cv.cvtColor(stream, background, cv.COLOR_BGR2GRAY)
    