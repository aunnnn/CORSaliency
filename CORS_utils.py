import numpy as np
import cv2
from skimage.filters import gabor_kernel
from skimage.feature import peak_local_max


def gaussian_pyramid(img, scale):
	"""Return Gaussian pyramid in list of images"""
	# if len(img.shape) == 3:
	# 	raise ValueError('img must be grayscale. img dim=%s' % len(img.shape))
	gp = [img]
	G = img
	for i in xrange(scale - 1):
		G = cv2.pyrDown(G)
		gp.append(G)    
	return gp


def laplacian_pyramid(img, scale):    
	"""Return Laplacian pyramid in list of images"""
	if len(img.shape) == 3:
		raise ValueError('img must be grayscale. img dim=%s' % len(img.shape))    
	# build laplacian directly
	levels = []    
	for i in xrange(scale-1):        
		next_img = cv2.pyrDown(img)        
		h, w = img.shape
		img1 = cv2.pyrUp(next_img, dstsize=(w, h))
		diff = cv2.subtract(img,img1)		 
		levels.append(diff)
		img = next_img
	levels.append(img)
	return levels


def get_simple_cell_style_gabor_kernels(frequency=0.25, nOrientations=4):
	gb_reals = []
	gb_ims = []
	for i in range(nOrientations):
		theta = i / 4. * np.pi
		gbs_skimage = gabor_kernel(frequency=frequency, theta=theta,  n_stds=3 )
		r = np.real(gbs_skimage)
		im = np.imag(gbs_skimage)
		gb_reals.append(r)
		gb_ims.append(im)
	return gb_reals, gb_ims


def complex_cells_response(src, gbs_r=None, gbs_im=None, frequency=0.25):
	"""Combine pair(even,odd) of simple cell responses 
	(sqrt(even^2+odd^2)) into one complex cell responses."""
	if gbs_r is None or gbs_im is None:
		gbs_r, gbs_im = get_simple_cell_style_gabor_kernels(frequency=frequency)
	spedrs = [cv2.filter2D(src, -1, kern) for kern in gbs_r]
	spedims = [cv2.filter2D(src, -1, kern) for kern in gbs_im]

	combsp = [np.sqrt(e**2+o**2) for e,o in zip(spedrs,spedims)]
	return combsp



def AcrossScaleAddition(pyr, finallyScaleTo=0, interpolation=cv2.INTER_NEAREST):
    """Scale each map to same size as pyr[finallyScaleTo], 
        then elementwise addition and finally minmax normalization."""
    if finallyScaleTo < 0 or finallyScaleTo >= len(pyr):
        raise ValueError('finallyScaleTo must be between 0 and less than ' + str(len(groupingpyr)))
    r, c = pyr[finallyScaleTo].shape[:2]    
    n_finalmap = np.zeros((r,c)).astype(np.float)
    for ngmap in pyr:
        n_finalmap += cv2.resize(ngmap, (c,r), interpolation=interpolation)
    return n_finalmap


def local_maxima(gray, mindist):    
	"""
		Return local maxima's coordinates in format [ [r,c], ... ]
	"""
	coords = peak_local_max(gray, min_distance=mindist)
	return coords


def MinMaxNorm(src):
	"""
		src-min / max-min+1
	"""
	return (src-np.min(src))/float(np.max(src)-np.min(src)+1)

def NoNegative(a):
	"""
		Clip all negative value to 0.
	"""
	a[a < 0] = 0
	return a

def ApplyEach(l, f, args=[]):
	"""
		Apply f on each item in l, specific args for f in a list if needed.
		E.g. l2 = ApplyEach(l,func,['param1', 'param2'])

	"""
	if len(args) != 0:
		return [f(tim, *args) for tim in l]
	else:
		return [f(tim) for tim in l]


def resize_image(src, width):
	"""
		Resize to given width and keep aspect ratio.
	"""
	if width <= 0:
		return src
	else:
		ratiohw = src.shape[0]/float(src.shape[1])
		return cv2.resize(src, (int(width),int(width*ratiohw)), interpolation=cv2.INTER_LINEAR)