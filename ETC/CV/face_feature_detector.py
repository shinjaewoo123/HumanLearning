import cv2
import numpy as np
import argparse
import cv2
import collections
import dlib
import sys
import os 

import imutils

class FaceAttDetect:
	''' '''
	FACIAL_LANDMARKS_IDXS = collections.OrderedDict([
	('mouth', (48, 68)),
	('right_eyebrow', (17, 22)),
	('left_eyebrow', (22, 27)),
	('right_eye', (36, 42)),
	('left_eye', (42, 48)),
	('nose', (27, 35)),
	('jaw', (0, 17))
	])

	def __init__(self):
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

	
	def rect_to_bb(self, rect):
		# convert form to (x, y, w, h)
		x = rect.left()
		y = rect.top()
		w = rect.right() - x
		h = rect.bottom() - y

		return (x, y, w, h)

	def shape_to_np(self, shape, dtype='int'):
		# init list of (x, y) : coordinates
		coords = np.zeros((68, 2), dtype=dtype)

		# 68 facial landmarks and convert them to 2-tuple of (x, y)
		for i in range(0, 68):
			coords[i] = (shape.part(i).x, shape.part(i).y)

		# list of (x, y)
		return coords

	def visualize_facial_landmarks(self, image, shape, colors=None, alpha=0.75):
		overlay = image.copy()
		output = image.copy()

		if colors is None:
			colors = [(19, 199, 109), 
				(79, 76, 240), (230, 159, 23),
				(168, 100, 168), (158, 163, 32), 
				(163, 38, 32), 
				(180, 42, 220)]

		for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
			(j, k) = FACIAL_LANDMARKS_IDXS[name]
			pts = shape[j:k]

			if name == 'jaw':
				for l in range(1, len(pts)):
					ptA = tuple(pts[l - 1])
					ptB = tuple(pts[l])
					cv2.line(overlay, ptA, ptB, colors[i], 2)

			else:
				hull = cv2.convexHull(pts)
				cv2.drawContours(overlay, [hull], -1, colors[i], -1)

		cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

		return output

	def get_color_face_att(self, path):
		''' '''
		self.get_video(self.detector, self.predictor)
		
	def get_video(self, detector, predictor):
		''' '''
		cap = cv2.VideoCapture(0)
		if cap.isOpened():
			print('width : {}, height : {}'.format(cap.get(3), cap.get(4)))

		while True:
			ret, frame = cap.read()

			if ret:
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				
				# detect face 
				rects = detector(gray, 1)
				for (i, rect) in enumerate(rects):
					# determine the facial landmarks ofr the face region.
					# convert facial landmark (x, y) -> numpy array
					shape = predictor(gray, rect)
					# shape = face_utils.shape_to_np(shape)
					shape = shape_to_np(shape)
					'''
					# convert rect to cv-style bbox (x, y, w, h)
					# (x, y, w, h) = face_utils.rect_to_bb(rect)
					(x, y, w, h) = rect_to_bb(rect)
					cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

					# show the face number
					cv2.putText(frame, 'Face #{}'.format(i + 1), 
						(x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

					for (x, y) in shape:
						cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
					'''
					for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():
						clone = frame.copy()
						cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
							0.7, (0, 0, 255), 2)

						for (x, y) in shape[i:j]:
							cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
						
						(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
						roi = frame[y:y + h, x:x + w]
						roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

						cv2.imshow('ROI', roi)
						cv2.imshow('Image', clone)
						cv2.waitKey(1)
						
					output = visualize_facial_landmarks(frame, shape)
					cv2.imshow('frame', output)
					cv2.waitKey(1)
				'''
				cv2.imshow('video', frame) # gray
				'''
				k = cv2.waitKey(1) & 0xFF
				if k == 27:
					break
			else:
				print('error !')

		cap.release()
		cv2.destroyAllWindows()

	def get_image(self, detector, predictor, img):
		''' '''
		img = cv2.imread('')
		print(img.shape)
		pass

def main():
	# args = parse_arg() # sys.argv[1:])
	img_parent_path = 'res\\'
	img_path = 'img_03.jpg' # img = cv2.imread(img_parent_path + '000002.jpg')
	
	facedetect = FaceDetect()
	img = facedetect.get_color_face(img_parent_path + img_path)
	# cv2.imshow('Image', img.copy())
	
	cv2.imwrite('image.jpg', img.copy())
	# print(img.shape)

if __name__ == '__main__':
	main()