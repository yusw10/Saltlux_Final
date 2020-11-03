# USAGE
# python text_detection.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import glob
import itertools
import os

# construct the argument parser and parse the arguments
# 콘솔창에서 값을 입력 받아 넣음
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to input image")
ap.add_argument("-east", "--east", type=str,help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5, help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320, help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320, help="resized image height (should be multiple of 32)")


# load the input image and grab the image dimensions
# image = cv2.imread(args["image"])

class image_detection():
    def __init__(self,cnt,frame):
        # print(image_path)
        args = vars(ap.parse_args(['--image', 'image_path',
                                '--east', 'frozen_east_text_detection.pb']))
        # image = cv2.imread(image_path)
        # image = cv2.imread(args["image"])
        image = self.frame
        # image = cv2.resize(image, dsize=(128, 128))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orig = gray_image.copy()
        (H, W) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        # 사이즈
        # (newW, newH) = (args["width"], args["height"])
        (newW, newH) = (320, 320)
        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")

        # 모델 입력
        # net = cv2.dnn.readNet(args["east"])
        net = cv2.dnn.readNet("C:\\Users\\JH\\Downloads\\AI\\yolov5\\detection\\frozen_east_text_detection.pb")

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        end = time.time()

        # show timing information on text prediction
        print("[INFO] text detection took {:.6f} seconds".format(end - start))

        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

        # loop over the number of columns
            for x in range(0, numCols):
        # if our score does not have sufficient probability, ignore it
        #       if scoresData[x] < args["min_confidence"]:
                if scoresData[x] < 0.5:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # loop over the bounding boxes
        for i,(startX, startY, endX, endY) in enumerate(boxes):
            # scale the bounding box coordinates based on the respective
            # ratios
            # padding
            startX, startY = int(int(startX * rW) * 0.9), int(int(startY * rH) * 0.9)
            endX, endY = int(int(endX * rW) * 1.1), int(int(endY * rH) * 1.1)

            # startX, startY = (0 if startX<0 else orig.shape[0] if startX>orig.shape[0] else startX),
            #     (0 if startY<0 else orig.shape[0] if startY>orig.shape[0] else startY)

            startX, startY = {startX<0 : 0 , startX>orig.shape[0] : orig.shape[0]}.get(True,startX),{startY<0 : 0 , startY>orig.shape[1]:orig.shape[1]}.get(True,startY)
            endX, endY = {endX<0 : 0 , endX>orig.shape[0] : orig.shape[0]}.get(True,endX),{endY<0 : 0 , endY>orig.shape[1]:orig.shape[1]}.get(True,endY)
            if(startX==endX or startY==endY):
                continue
            print(startX, startY, endX, endY, orig.shape[0], orig.shape[1])
            # draw the bounding box on the image
            # cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # cv2.imshow("asdf", orig)
            # cv2.waitKey(0)
            cv2.imwrite(f'detection_img/test_{cnt}_{i}.jpg', orig[startY:endY, startX:endX])
        



if __name__ == '__main__':
    file = "D:\study\image"
    # file="./demo_image"

    # png, jpg 확장자를 가진 사진들 이름 불러오기
    fnames = os.listdir(file)

    cap = cv2.VideoCapture('')

    for i,fname in enumerate(fnames):
        image_detection(i,file + "/" + fname)
    cv2.destroyAllWindows()
