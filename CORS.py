import numpy as np
import cv2
# from cvutil.basicf import *
# from cvutil.filter import *
import matplotlib.pyplot as plt
from CORS_utils import *


def intensity(bgr_img):
	"""Return intensity image computed by (R+G+B)/3. Used with color_opponency as an intensity channel."""
	if np.ndim(bgr_img) != 3:
		raise ValueError('bgr_img must be BGR image (3 dimensions), not ' + str(np.ndim(bgr_img)))
	img_b, img_g, img_r = cv2.split(bgr_img.astype(np.float))
	imgray = (img_b+img_g+img_r)/3.0
	#imgray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float)
	return imgray


def color_opponency(bgr_img):
	"""Return tuple of color opponency from BGR color image(in format: [RG,GR,BY,YB])"""
	if np.ndim(bgr_img) != 3:
		raise ValueError('bgr_img must be BGR image (3 dimensions), not ' + str(np.ndim(bgr_img)))
	img_b, img_g, img_r = cv2.split(bgr_img.astype(np.float))
	fimg_R = NoNegative(img_r - (img_g+img_b)/2.0)
	fimg_G = NoNegative(img_g - (img_r+img_b)/2.0)
	fimg_B = NoNegative(img_b - (img_r+img_g)/2.0)
	fimg_Y = NoNegative((img_r+img_g)/2.0 - np.abs(img_r-img_g)/2.0 - img_b)
	fimg_RG = NoNegative(fimg_R-fimg_G)
	fimg_GR = NoNegative(fimg_G-fimg_R)
	fimg_BY = NoNegative(fimg_B-fimg_Y)
	fimg_YB = NoNegative(fimg_Y-fimg_B)
	return [fimg_RG, fimg_GR, fimg_BY, fimg_YB]


def LocalFigureNorm(src,figureRatio=0.2):
	"""
		Subtract src from boxFilter version of itself, with radius 0.2 of its side.
		Parameters:
		==========
			- figureRatio is approximated figure's size compared to src's width.
	"""
	h,w = src.shape[:2]
	th,tw = int(h*figureRatio), int(w*figureRatio)
	figApprx = cv2.boxFilter(src, -1, (tw,tw))
	
	return WNorm(MinMaxNorm(NoNegative(src-figApprx)))


def WNorm(src):
	"""Divide by sqrt of number of local maxima, e.g. too promote few local peaks"""
	peaks = local_maxima(src, 3)

	if len(peaks)==0:
		return src
	else:
		return src/float(np.sqrt(len(peaks)))


def OnOffFM(i_gpyr):
	"""On,off center-surround intensity maps"""
	onpyr = []
	offpyr = []
	for i in xrange(0,len(i_gpyr)): # scale 2,3,4
		curi = i_gpyr[i]
		surround3 = cv2.boxFilter(curi, -1, (7,7))
		surround7 = cv2.boxFilter(curi, -1, (15,15))
		on3 = NoNegative(curi - surround3)
		on7 = NoNegative(curi - surround7)
		off3 = NoNegative(surround3 - curi)
		off7 = NoNegative(surround7 - curi)
		onpyr.append(on3+on7)
		offpyr.append(off3+off7)
	onmap = AcrossScaleAddition(onpyr)
	offmap = AcrossScaleAddition(offpyr)
	return onmap, offmap


def saliency_map(imbgr,GaussR=9):
	"""
	Compute corner saliency maps.

		Parameters:
		-----------
		- imbgr is input BGR colour image.
		- cornerForFigure is true for saliency map generation 
			(e.g. take log and exhibit more corners to cue figures)
			- if false, will not take log, and show less, but more accurate corner locations.
	"""
	imbgr = resize_image(imbgr, 256)

	# R,G,B,Y
	colops = color_opponency(imbgr)
	# intensity
	imgray = intensity(imbgr)
	
	colors = ApplyEach(colops,laplacian_pyramid,[2])
	Ion,Ioff = OnOffFM(gaussian_pyramid(imgray,2))

	# all channels (r,g,b,y,i_on,i_off) feature maps
	lpyrs = [l1 for l1,l2 in colors] + [Ion,Ioff]


	edgepyrs = ApplyEach(lpyrs, complex_cells_response, [Gbs_r,Gbs_im])
	
	mul_f = lambda x,y: x*y
	log_f = lambda x: np.log(x+1)

	# Corner Feature:
	
	# Extract corner features by multiply all orientations together,
	# thus leaves with only locations with multiple orientations.
	#	--> take log on corner response to show more corner, to better cue figure locations
	cornermaps = [log_f(reduce(mul_f, edgepyrs[i])) for i in xrange(len(edgepyrs))]

	# make corner maps all channels comparable e.g. range (0,1)
	cornermaps = ApplyEach(cornermaps, MinMaxNorm)

	# suppress too abundance corner informations
	cornermaps = ApplyEach(cornermaps, LocalFigureNorm, [0.2])

	# combine all "figure cues"	
	cornermap = sum(cornermaps)
	cornermap = resize_image(cornermap, 128)
	cornermap = cv2.GaussianBlur(cornermap, (GaussR,GaussR), 0,0 )
	return cornermap


def prepare_gabor_kernels():
	global Gbs_r, Gbs_im, OrientedGaussianKernels
	Gbs_r, Gbs_im = get_simple_cell_style_gabor_kernels(frequency=0.2, nOrientations=4)
	print('Gabor set up.'),


if __name__ == '__main__':
	prepare_gabor_kernels()
	imbgr = cv2.imread('1.jpg')
	salmap = saliency_map(imbgr)
	plt.figure()
	plt.subplot(1,2,1)
	plt.imshow(cv2.cvtColor(imbgr,cv2.COLOR_BGR2RGB))
	plt.title('input')

	plt.subplot(1,2,2)
	plt.imshow(salmap)
	plt.title('saliency map')
	plt.show()


