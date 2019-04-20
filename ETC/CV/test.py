import cv2 
from histogram_equalization import caip_2017

if __name__ == '__main__':
	img = cv2.imread('res/img_11.jpg', cv2.IMREAD_COLOR)
	caip_2017(img)
