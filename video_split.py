import cv2
import os
import time

st = time.time()
video_path = './vid1.avi'
frames_path = './frames'
#video = cv2.VideoCapture(video_path)
video = cv2.VideoCapture(f"v4l2src device=/dev/video2 ! video/x-raw, width={VID_RESO[0]}, height={VID_RESO[1]}, framerate=60/1, format=(string)UYVY ! decodebin ! videoconvert ! appsink", cv.CAP_GSTREAMER)

while True:
	image_exists, image = video.read()
	cnt = 0
	while image_exists is True:
	    if not os.path.isdir(frames_path):
		os.makedirs(frames_path)
	    cv2.imwrite(frames_path+'/input%03d.png'%cnt, image)
	    image_exists, image = video.read()
	    cnt += 1
	else:
		break
	end=time.time()
	print(end-st)
	
