import cv2
import dlib
import numpy as np
import time
from scipy import ndimage
from math import atan2, degrees, pi
import sys

VIDEO_CAPTURE_DEVICE = str(sys.argv[1:][0])
RESIZE_LEVEL = float(sys.argv[1:][1])

if len(VIDEO_CAPTURE_DEVICE) == 1:
    VIDEO_CAPTURE_DEVICE = int(VIDEO_CAPTURE_DEVICE) #if 1st argument is 0 or 1 (not path to file)

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()
overlayImg = cv2.imread("moustache.png", -1) #load with transparency
lastFrameTimestamp = int(round(time.time() * 1000))
DEBUG_MODE = False
DEBUG_MODE_NUM = False

def calculateAngle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    rads = atan2(-dy,dx)
    rads %= 2*pi
    degs = degrees(rads)
    return degs

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """
	https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv#40709055
	Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] + alpha_inv * img[y1:y2, x1:x2, c])

def get_faces(im):
    rects = detector(im, 1)
    return rects

def get_landmarks(image, face):
    return np.matrix([[p.x, p.y] for p in predictor(image, face).parts()])

def annotate_landmarks(im, landmarks, color2):
    #im = im.copy()
    overlayImgEdited = overlayImg
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        if DEBUG_MODE:
            cv2.circle(im, pos, 1, color, -1)
        if DEBUG_MODE_NUM:
            cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(000,000,000))

    """
    leftEyeCenterX = int((landmarks[39][0, 0] + landmarks[36][0, 0]) / 2)
    leftEyeCenterY = int((landmarks[37][0, 1] + landmarks[41][0, 1]) / 2)
    leftEyeWidth = int(landmarks[39][0, 0] - landmarks[36][0, 0])
    leftEyeHeight = int(landmarks[41][0, 1] - landmarks[37][0, 1])
    print("Left eye is at: " + str(leftEyeCenterX) + "-" + str(leftEyeCenterY))
    print("Left eye width: " + str(leftEyeWidth))
    cv2.circle(im, (leftEyeCenterX, leftEyeCenterY), int(leftEyeWidth / 2), (000,000,255), 2)
    """
	
    moustacheCenterX = int((landmarks[33][0,0] + landmarks[51][0,0]) / 2)
    moustacheCenterY = int((landmarks[33][0,1] + landmarks[51][0,1]) / 2)

    mouthWidth = int((landmarks[54][0,0] - landmarks[48][0,0]))
    
	#resize
    new_width = int(1.5 * mouthWidth)
    new_height = int((new_width * overlayImgEdited.shape[0]) / overlayImgEdited.shape[1])
    overlayImgEdited = cv2.resize(overlayImgEdited, (new_width, new_height))
    
    #rotate overlay image
	#angle of rotation is the angle between 2 mouth landmarks
    overlayImgEdited = ndimage.rotate(overlayImgEdited, calculateAngle(landmarks[48][0,0], landmarks[48][0,1], landmarks[54][0,0], landmarks[54][0,1]))

    #calculate upper left location of overlay image
    overlayImgLeftUpX = int(moustacheCenterX - (overlayImgEdited.shape[1] / 2))
    overlayImgLeftUpY = int(moustacheCenterY - (overlayImgEdited.shape[0] / 2))

	#add overlay image
    overlay_image_alpha(im, overlayImgEdited[:, :, 0:3], (overlayImgLeftUpX, overlayImgLeftUpY), overlayImgEdited[:, :, 3] / 255.0)  
    return im

##################################################################

cap = cv2.VideoCapture(VIDEO_CAPTURE_DEVICE)
frameNumber = 0
while True:
	ret, frame = cap.read()  
	if ret == False: #reached end of video
		break
	else:
		print("")
		frameTimestampStart = int(round(time.time() * 1000))

		frameNumber += 1
		print("Frame " + str(frameNumber))
		
		image = frame
		image = cv2.resize(frame, None, fx=RESIZE_LEVEL, fy=RESIZE_LEVEL, interpolation = cv2.INTER_LANCZOS4)
		image_with_landmarks = image
		
		#DEBUGTimestampStart = int(round(time.time() * 1000))
		faces = get_faces(image)
		#DEBUGTimestampEnd = int(round(time.time() * 1000))
		#print(str(DEBUGTimestampEnd - DEBUGTimestampStart) + "ms for get_faces() function")
		
		for index, face in enumerate(faces):
			landmarks = get_landmarks(image, face)
			color = (255,255,255)
			if index == 0: #1st face
				color = (000, 255, 000)
			if index == 1: #2nd face
				color = (000, 255, 255)
			image_with_landmarks = annotate_landmarks(image, landmarks, color)

		#resize back to original frame size
		#image_with_landmarks = cv2.resize(image_with_landmarks, None, fx=float(1/RESIZE_LEVEL), fy=float(1/RESIZE_LEVEL), interpolation = cv2.INTER_LANCZOS4)

		frameTimestampEnd = int(round(time.time() * 1000))
		frameTimeToProcess = frameTimestampEnd - frameTimestampStart
		print(str(frameTimeToProcess) + "ms = " + str(round(1000/frameTimeToProcess)) + " FPS")

		cv2.imshow("frame", image_with_landmarks)
		print("image_with_landmarks size:" + str(image_with_landmarks.shape[0]) + "," + str(image_with_landmarks.shape[1]))

		keyPressed = cv2.waitKey(1)
		if keyPressed == 13: #13 is the Enter Key
			break
		elif keyPressed == 49: #pressing keyboard key 1 shows facial landmarks as dots.
				if DEBUG_MODE:
					DEBUG_MODE = False
				else:
					DEBUG_MODE = True
				print('DEBUG_MODE=' + str(DEBUG_MODE))
		elif keyPressed == 50: #pressing keyboard key 2 show facial landmarks as numbers.
				if DEBUG_MODE_NUM:
					DEBUG_MODE_NUM = False
				else:
					DEBUG_MODE_NUM = True
				print('DEBUG_MODE_NUM=' + str(DEBUG_MODE_NUM))

cap.release()
cv2.destroyAllWindows()  
#cv2.imwrite('lastFrame.png',image_with_landmarks)
