import cv2
import numpy as np
import argparse
import cv2
import collections
import dlib
import sys
import os 

import imutils

class FaceDetect:
	''' '''
	def __init__(self):
		self.detection_model_path = 'detection_models/haarcascade_frontalface_default.xml'
		self.face_detection_model = self.load_detection_model(self.detection_model_path)

	def load_detection_model(self, model_path):
		return cv2.CascadeClassifier(model_path)
	
	def detect_faces(self, detection_model, gray_image_array):
		return detection_model.detectMultiScale(gray_image_array, 1.3, 5)
	
	def get_color_face(self, path):
		''' Return None or color_face'''
		try:
			color_img = cv2.imread(path, cv2.IMREAD_COLOR)
			gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
			color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
		except Exception as e:
			print(e)
			print('The error is occurred in processing img open ' + path)
			return None

		# faces = cv2.resize(color_img, (299, 299))
		# faces = np.squeeze(faces).astype('float32')
		try:
			faces = self.detect_faces(self.face_detection_model, gray_img)
		except Exception as e:
			print(e)
			print('The error is occurred in processing ' + path)
			return None

		face_len = len(faces)
		if face_len == 0:
			print('FAIL, len = 0')
			return None

		if face_len > 1:
			faces = sorted(faces, reverse=True, key=lambda element: element[2])
			faces = faces[0]
		if face_len == 1:
			x, y, w, h = faces[0]
		elif face_len == 4:
			x, y, w, h = faces
		
		color_face = color_img[y:y+h, x:x+w]
		color_face = cv2.resize(color_face, (299, 299))
		color_face = np.squeeze(color_face).astype('float32')

		
		color_face = cv2.cvtColor(color_face, cv2.COLOR_RGB2BGR)
		return color_face[:]


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