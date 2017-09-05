import cv2
import keras
from scipy.misc import imresize
import numpy as np
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
cascade_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
height = width = 20



def detect_face(image):
	faces = cascade_classifier.detectMultiScale(image, 1.3, 5)
	for (x,y,w,h) in faces:
		image = image[y:y+h, x:x+w]
		return True,image
	return False,image


cap = cv2.VideoCapture("http://192.168.1.104:8080/video?x.mjpeg")

model = keras.models.load_model('cnn3.h5')

while cap.isOpened():
    # Read first frame
    ret, img = cap.read()
    if ret:
		x=[]
		gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
		flag,face = detect_face(gray)
		if flag:
			gray = imresize(face, [height, width], 'bilinear')
			gray = np.dstack((gray,) * 3)
			x.append(gray)
			x = np.asarray(x)
			result=model.predict( x, batch_size=8, verbose=0)
			for index,emotion in enumerate(EMOTIONS):
				cv2.putText(img, emotion, (10,index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1);
				cv2.rectangle(img, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)
				print(result)
		cv2.imshow('frame',img)
		if cv2.waitKey(1) &0xff==ord('q'): #\n is the Enter Key
			break
else:
	print "camera not opened"
cv2.destroyAllWindows()
