import cv2 as cv
import numpy as np
import math
import PySimpleGUI as sg
class pho:
	def __init__(self):
		self.img=None
	def translate(self, img, vec):
		if(img is not None):
			rows, cols, _ = img.shape
			M = np.float32([[1, 0, vec[0]], [0, 1, vec[1]]])
			dst = cv.warpAffine(img, M, (cols, rows))
			return dst
		return

	def rotate(self, img, angle):
		if(img is not None):
			(h, w) = img.shape[:2]
			center = (w / 2, h / 2)
			M = cv.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
			rotated = cv.warpAffine(img, M, (w, h))
			return rotated
		return

	def scale(self, img):
		if(img is not None):
			resized = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
			return resized
		return

	def skew(self, img, shear):
		afine_tf = cv.AffineTransform(shear=shear)
		modified = cv.warp(image=img, inverse_map=afine_tf)
		return modified

	def draw(self,img,x,y,size_brush,color_brush):
		cv.circle(img, (x,y), size_brush,  color_brush, -1)
		return 		