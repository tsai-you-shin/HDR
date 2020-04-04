from __future__ import print_function
from __future__ import division
from math import exp
import cv2 as cv
import numpy as np

def PhotographicGlobal(hdr, delta, a, white):
	Lw = np.dot(hdr[...,:3], [0.06, 0.67, 0.27])
	LwMean = exp(np.mean(np.log(delta + Lw)))
	Lm = (a / LwMean) * Lw
	Lm[np.isnan(Lm)] = 0
	Ld = (Lm * (1 + Lm / (white**2))) / (1 + Lm);
	ldr = np.zeros([len(hdr), len(hdr[0]), 3], dtype=np.double)
	for channel in range(3):
		Cw = hdr[:, :, channel] / Lw
		Cw[np.isnan(Cw)] = 0
		ldr[ :, :, channel] = Cw * Ld.astype(np.double)
	#gamma correction
	ldr = np.power(ldr, 1/2.2)
	return ldr

def PhotographicLocal(hdr, delta, a, phi, epsilon, s_range):
	Lw = np.dot(hdr[...,:3], [0.06, 0.67, 0.27])
	LwMean = exp(np.mean(np.log(delta + Lw)))
	L = (a / LwMean) * Lw
	Lblur_s = np.zeros([len(Lw), len(Lw[0]), s_range]);
	Ld = np.zeros([len(Lw), len(Lw[0])])
	for i in range(1, s_range):
		s = 1.6 ** i
		Lblur_s[:, :, i] = cv.GaussianBlur(L,(5, 5), s, cv.BORDER_DEFAULT)
		cv.imwrite('Lblur_s'+str(i)+".jpg", Lblur_s[:, :, i] * 255)
	for x in range(len(L)):
		for y in range(len(L[0])):
			smax = 1
			for i in range(1, s_range - 1):
				s = 1.6 ** 2
				denominator = ((2 ** phi) * a / s**2) + Lblur_s[x, y, i]
				if denominator == 0:
					Vsxy = 0
				else:
					Vsxy = (Lblur_s[x, y, i] - Lblur_s[x, y, i+1]) / denominator
				if abs(Vsxy) < epsilon:
					smax = i
			if(1 + Lblur_s[x, y, smax] == 0):
				Ld[x, y] = 0
			else:
				Ld[x, y] = L[x, y] / (1 + Lblur_s[x, y, smax])
	ldr = np.zeros([len(hdr), len(hdr[0]), 3])
	for channel in range(3):
		one_color_channel = Ld * (hdr[:, :, channel] / Lw) 
		ldr[ :, :, channel] = one_color_channel
	#gamma correction
	ldr = np.power(ldr, 1/2.2)
	return ldr
