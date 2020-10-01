import cv2
import time
import argparse

aq = argparse.ArgumentParser()

aq.add_argument('-t', '--tracker', required=True, help="the name of the tracker")

aq.add_argument('-v', '--video', required=True, help="realtive path of the video \
	or 0 for webcam")

args = vars(aq.parse_args())

TRACKER_OPTIONS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create,
	"goturn":  cv2.TrackerGOTURN_create
}

if(args['video']== '0'):
	cap = cv2.VideoCapture(0)
else:
	cap = cv2.VideoCapture(args['video'])



print(cv2.__version__)

#videoname = "video.avi"

tracker = TRACKER_OPTIONS[args['tracker']]()
success, img = cap.read()
bbox = cv2.selectROI(args['tracker'].upper(), img, False)
tracker.init(img, bbox);


height, width, _ = img.shape
#video = cv2.VideoWriter(videoname, 0, 20, (height, width), False)

def drawBox(img, bbox):
	x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), 
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3, 1)
	cv2.putText(img, "Tracking", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

total, count = 0 ,0

while True:
	time.sleep(0.02)
	timer = cv2.getTickCount()
	success, img = cap.read()

	if img is None:
		print('Video Completed')
		break

	total += 1
	success, bbox = tracker.update(img)

	if success:
		drawBox(img, bbox)
		count += 1
	else:
		cv2.putText(img, "Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


	fps = cv2.getTickFrequency()//(cv2.getTickCount()-timer)
	cv2.putText(img, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	#video.write(img)

	cv2.imshow(args['tracker'].upper(), img)
	cv2.waitKey(1)


#video.release()


	# if cv2.waitKey(1) & 0xff== ord('q'):
	# 	break


print("Accuracy: ", count/total)

