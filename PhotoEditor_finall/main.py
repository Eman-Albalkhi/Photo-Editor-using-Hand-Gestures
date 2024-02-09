import cv2 as cv
import keyboard
import numpy as np
from photo_GUI import GUI
from photo import pho


import Hand

Window= 'Choose photo'

frame_width = 300 
frame_x_offset=frame_y_offset=20

def main():
	Gui=GUI()
	directory_photo=Gui.Directory_photo()
	imge = cv.imread(directory_photo)


	capture = cv.VideoCapture(0)


	_, frame = capture.read()

	asp_ratio = frame.shape[0] / frame.shape[1]
	start_center = 0


	hand = Hand.hand(frame)
	ph=pho()

	imge_2=imge
	size_brush=5
	color_brush=[0,255,255]
	while True:

		pressed_key = cv.waitKey(1)
		_, frame = capture.read()
		frame = cv.flip(frame, 1)

		if pressed_key & 0xFF == ord('r'):
			hand = Hand.hand(frame)
		elif hand.hand_hist_True:
			hand.hand_tracking(frame)

			hand.draw_point_far(frame, [0, 255, 255])
			hand.draw_tips(frame, [255, 0, 255])
			hand.draw_convex_hull(frame)
			hand.draw_contours(frame)

		elif pressed_key & 0xFF == ord('a'):
			hand.hand_histogram(frame)
		else:
			hand.draw_rectangle(frame)


		if keyboard.is_pressed("s") and (imge is not None):
			if(hand.number==2):
				size_brush=5
			if(hand.number==33):
				size_brush=10
			#imge_2=imge
			#imge_2=ph.draw(imge_2,(int)(hand.cnt_centroid[0]*imge_2.shape[0]/frame.shape[0]),(int)(hand.cnt_centroid[1]*imge_2.shape[1]/frame.shape[1]),size_brush,color_brush)
			
		elif keyboard.is_pressed("c") and (imge is not None):
			if(hand.number==2):
				color_brush=[0, 255, 255]
			if(hand.number==33):
				color_brush=[0, 0, 255]
			# imge_2=imge
			# imge_2=ph.draw(imge_2,(int)(hand.cnt_centroid[0]*imge_2.shape[0]/frame.shape[0]),(int)(hand.cnt_centroid[1]*imge_2.shape[1]/frame.shape[1]),size_brush,color_brush)
		
		elif keyboard.is_pressed("d") and (imge is not None):
			imge_2=imge
			imge_2=ph.draw(imge_2,(int)(hand.cnt_centroid[0]*imge_2.shape[0]/frame.shape[0]),(int)(hand.cnt_centroid[1]*imge_2.shape[1]/frame.shape[1]),size_brush,color_brush)
		
		elif hand.number==2:
			if start_center == 0:
				start_center = hand.cnt_centroid
			
			trans_vec=((hand.cnt_centroid[0] - start_center[0]) / frame.shape[1], (hand.cnt_centroid[1] - start_center[1]) / frame.shape[0])
			imge_2 = imge
			imge_2 = ph.translate(imge_2, np.multiply(trans_vec, imge.shape[0]).astype(int))

		elif hand.number==5:
			imge_2=img
			imge_2=ph.rotate(imge_2,60)

		elif hand.number==33:
			if(imge is not None):
				imge_2=imge
				imge_2=ph.scale(img)
		elif hand.number==4:
			if(imge is not None):
				imge_2=imge
		else:
			img = imge_2
			start_center = 0

		


		frame_height = int(frame_width * asp_ratio)
		frame = cv.resize(frame, (frame_width, frame_height))
		if(imge_2 is not None):
			show_img = imge_2.copy()


		show_img[frame_y_offset:frame_y_offset + frame.shape[0],
		frame_x_offset:frame_x_offset + frame.shape[1]] = frame



		if keyboard.is_pressed("ESC"):
			break



		cv.imshow(Window, show_img)

		if cv.waitKey(1) & 0xFF == ord('x'):  
			break
	capture.release()
	cv.destroyAllWindows()


if __name__ == '__main__':
	main()