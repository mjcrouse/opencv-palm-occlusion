#imports
import time
import numpy as np
import cv2 as cv
import imutils
import webcamvideostream

#setup camera stream
cap = webcamvideostream.WebcamVideoStream().start()
#delay for camera to warm up
time.sleep(2.0)

num_frames = 0

while True:
    frame = cap.read()
    frame = cv.flip(frame, 1)

    ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCR_CB)
    #hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    ch1, ch2, ch3 = cv.split(ycrcb)
    if num_frames == 0:
        bg1 = (cv.GaussianBlur(ch1, (13, 13), 0)).astype("float")
        bg2 = (cv.GaussianBlur(ch2, (13, 13), 0)).astype("float")
        bg3 = (cv.GaussianBlur(ch3, (13, 13), 0)).astype("float")
    if num_frames < 60:
        cv.accumulateWeighted(ch1, bg1, 0.5)
        cv.accumulateWeighted(ch2, bg2, 0.5)
        cv.accumulateWeighted(ch3, bg3, 0.5)

    #background subtraction
    #kernel = np.ones((3,3),np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    diff1 = cv.absdiff(ch1, bg1.astype("uint8"))
    thre1, diff1 = cv.threshold(diff1, 25, 255, cv.THRESH_BINARY)
    diff1 = cv.morphologyEx(diff1, cv.MORPH_OPEN, kernel, iterations=2)
    diff2 = cv.absdiff(ch2, bg2.astype("uint8"))
    thre2, diff2 = cv.threshold(diff2, 8, 255, cv.THRESH_BINARY)
    diff2 = cv.morphologyEx(diff2, cv.MORPH_OPEN, kernel, iterations=2)
    diff3 = cv.absdiff(ch3, bg3.astype("uint8"))
    thre3, diff3 = cv.threshold(diff3, 8, 255, cv.THRESH_BINARY)
    diff3 = cv.morphologyEx(diff3, cv.MORPH_OPEN, kernel, iterations=2)
    diff = cv.add(diff1, diff2)
    diff = cv.add(diff, diff3)
    # diff = cv.merge((diff1, diff2, diff3))
    # diff = cv.bitwise_and(diff, diff, mask=frame)
    # cv.imshow("diff", diff)
    # masked = cv.bitwise_and(ycrcb, ycrcb, mask = diff)

    # face removal
    face_def = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame_gray = cv.equalizeHist(frame_gray)
    faces = face_def.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        #cv.rectangle(diff, (x, y), (x+w, y+h), (0, 0, 0), -2)
        cv.rectangle(diff, (x-35, y-35), (x+w+35, y+h+35), (0, 0, 0), -2)
    # cv.imshow("faces", frame_gray)

    #canny edges
    canny = cv.GaussianBlur(frame, (5,5),0)
    canny = cv.Canny(frame, 40, 100)
    #cannyinvert = cv.bitwise_not(canny)
    can = np.zeros((canny.shape)).astype('uint8')
    can = cv.cvtColor(can, cv.COLOR_GRAY2BGR)
    #diff_w_canny = cv.bitwise_and(diff, canny)
    cannycontours, h = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(can, cannycontours, -1, (255, 255, 255), 3)
    can = cv.cvtColor(can, cv.COLOR_BGR2GRAY)
    can = cv.bitwise_not(can)
    # cv.imshow("canny", canny)
    # cv.imshow("can", can)
    # cv.imshow("diffwcanny", diff_w_canny)
    #cv.imshow("inverted", cannyinvert)

    #skin detection
    frame_threshold = cv.inRange(ycrcb, (0, 131, 77), (235, 173, 135))
    frame_threshold2 = cv.GaussianBlur(frame_threshold, (5, 5), 0)
    frame_threshold2 = cv.morphologyEx(frame_threshold2, cv.MORPH_OPEN, kernel, iterations=2)
    # cv.imshow('inRange', frame_threshold)
    # cv.imshow('inRange2', frame_threshold2)
    #masked = cv.bitwise_and(frame, frame, mask = diff)

    # cv.imshow("diff1", diff1)
    # cv.imshow("diff2", diff2)
    # cv.imshow("diff3", diff3)
    # cv.imshow("diffsum", diff)

    skintest = cv.bitwise_and(diff, frame_threshold2)
    
    # skintest = cv.bitwise_and(diff, frame_threshold2)
    skintest2 = cv.GaussianBlur(skintest, (9, 9), 0)
    skintest2 = cv.morphologyEx(skintest2, cv.MORPH_OPEN, kernel, iterations=2)
    skintest = cv.bitwise_and(skintest, can)
    # contours
    withcontours = cv.cvtColor(skintest, cv.COLOR_GRAY2BGR)
    contours, hierarchy = cv.findContours(skintest, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # firstcontours = False
    # if len(contours) > 0 and (firstcontours == False):
    #     time.sleep(1.0)
    #     firstcontours = True
    if len(contours) < 2:
        pass
    else:
        sortedcontours = sorted(contours, key=cv.contourArea)
        withcontours = cv.drawContours(withcontours, sortedcontours[-2:], -1, (255, 0, 0), 1)
        c1 = sortedcontours[-1]
        c2 = sortedcontours[-2]
        
        x,y,w,h = cv.boundingRect(c1)
        cv.rectangle(withcontours,(x,y),(x+w,y+h),(0,255,0),2)
        x2,y2,w2,h2 = cv.boundingRect(c2)
        cv.rectangle(withcontours,(x2,y2),(x2+w,y2+h),(0,255,0),2)
        
        # centroid (bounding rectangle)
        if x < x2:
            lefthand = sortedcontours[-1]
            lefthandcentroid = (int(x+w/2.0), int(y+h/2.0))
            righthand = sortedcontours[-2]
            righthandcentroid = (int(x2+w2/2.0), int(y2+h2/2.0))
        else:
            lefthand = sortedcontours[-2]
            lefthandcentroid = (int(x2+w2/2.0), int(y2+h2/2.0))
            righthand = sortedcontours[-1]
            righthandcentroid = (int(x+w/2.0), int(y+h/2.0))
        cv.circle(withcontours, lefthandcentroid, 5, (0, 0, 255), thickness=-2)
        cv.circle(withcontours, righthandcentroid, 5, (0, 0, 255), thickness=-2)

        # centroid (contour)
        # bigM = cv.moments(sortedcontours[-2])
        # bigMx = int(bigM['m10']/bigM['m00'])
        # bigMy = int(bigM['m01']/bigM['m00'])
        # cv.circle(withcontours, (bigMx, bigMy), 5, (0, 0, 255), thickness=-2)
        # bigM2 = cv.moments(sortedcontours[-1])
        # bigM2x = int(bigM2['m10']/bigM2['m00'])
        # bigM2y = int(bigM2['m01']/bigM2['m00'])
        # cv.circle(withcontours, (bigM2x, bigM2y), 5, (0, 0, 255), thickness=-2)

        # palm detection, largest inscribed circle, not working
        # dist=np.zeros((180, 320))
        # justcontours = np.zeros((720,1280)).astype('uint8')
        # justcontours = cv.cvtColor(justcontours, cv.COLOR_GRAY2BGR)
        # justcontours = cv.drawContours(justcontours, sortedcontours[-2:], -1, (0, 255, 0), 2)
        # justcontours = cv.resize(justcontours, (320,180))
        # r = range(320)
        # for ind_y in r[:180:4]:
        #     for ind_x in r[::4]:
        #         dist[ind_y,ind_x] = cv.pointPolygonTest(sortedcontours[-1],(ind_y,ind_x),True)
        # minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(dist)
        # print(minVal, maxVal, minLoc, maxLoc)
        # if maxVal < 0:
        #     maxVal = 0
        #cv.circle(justcontours, (maxLoc[1], maxLoc[0]), (int(maxVal)), (0, 255, 0), thickness=-2)
        #cv.imshow("justcontours", justcontours)

        # extreme points
        extLeft1 = tuple(righthand[righthand[:, :, 0].argmin()][0])
        # extRight1 = tuple(c1[c1[:, :, 0].argmax()][0])
        # extTop1 = tuple(c1[c1[:, :, 1].argmin()][0])
        # extBot1 = tuple(c1[c1[:, :, 1].argmax()][0])
        cv.circle(withcontours, extLeft1, 8, (0, 0, 255), -1)
        # cv.circle(withcontours, extRight1, 8, (0, 255, 0), -1)
        # cv.circle(withcontours, extTop1, 8, (255, 0, 0), -1)
        # cv.circle(withcontours, extBot1, 8, (255, 255, 0), -1)
        # extLeft2 = tuple(c2[c2[:, :, 0].argmin()][0])
        # if extLeft1[0] < extLeft2[0]:
        #     cv.putText(withcontours, "extLeft1 < extLeft2", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # else:
        #     cv.putText(withcontours, "extLeft2 < extLeft1", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        #check if leftmost point of right hand in left hand contour (doesn't work, point never in hand countour)
        # if cv.pointPolygonTest(lefthand, extLeft1, False) > 0:
        #     cv.putText(withcontours, "TRUE", (40, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # else:
        #     cv.putText(withcontours, "FALSE", (40, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #retry crossover
        x,y,w,h = cv.boundingRect(lefthand)
        if x < extLeft1[0] and (x+w) > extLeft1[0] and y < extLeft1[1] and y+h < extLeft1[1]:
            print(True)
    # cv.imshow("skintest", skintest)
    # cv.imshow("skintest2", skintest2)
    cv.imshow("contours", withcontours)

    # nb_components, output, stats, centroids = cv.connectedComponentsWithStats(skintest)
    # sizes = stats[1:, -1]
    # nb_components = nb_components - 1
    # largestblobs = np.zeros((output.shape))
    # for i in range(0, nb_components):
    #     if sizes[i] > 600:
    #         largestblobs[output == i + 1] = 255
    # output = output.astype("uint8")
    # cv.imshow("output", output)

    # masked = cv.bitwise_and(frame, frame, mask = skintest)
    # cv.imshow('test', test)
    # cv.imshow("masked", masked)
    # cv.imshow('ycrcb', ycrcb)
    # cv.imshow('hsv', hsv)

    cv.imshow("Video Feed", frame)

    num_frames += 1

    # end program if q or Esc pressed
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == 27:
        break

cap.stop()
cv.destroyAllWindows()
