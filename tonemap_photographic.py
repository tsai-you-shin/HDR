from __future__ import print_function
from __future__ import division
from math import exp
import cv2 as cv
import numpy as np
import argparse
import os

def PhotographicGlobal(hdr, delta, a, white):
	Lw = np.dot(hdr[...,:3], [0.06, 0.67, 0.27])
	N = len(Lw) * len(Lw[0])
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

def PhotographicLocal(hdr, delta, a, white, phi, epsilon, s_range):
	Lw = np.dot(hdr[...,:3], [0.06, 0.67, 0.27])
	N = len(Lw) * len(Lw[0])
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

def loadExposureSeq(path):
    images = []
    times = []
    with open(os.path.join(path, 'list.txt')) as f:
        content = f.readlines()
    for line in content:
        tokens = line.split()
        images.append(cv.imread(os.path.join(path, tokens[0])))
        times.append(1/float(tokens[1]))
    return images, np.asarray(times, dtype=np.float32)

parser = argparse.ArgumentParser(description='Code for High Dynamic Range Imaging tutorial.')
parser.add_argument('--input', type=str, help='Path to the directory that contains images and exposure times.')
args = parser.parse_args()
if not args.input:
    parser.print_help()
    exit(0)

images, times = loadExposureSeq(args.input)
calibrate = cv.createCalibrateDebevec()
response = calibrate.process(images, times)
merge_debevec = cv.createMergeDebevec()
hdr = merge_debevec.process(images, times, response)
a = 0.18
tnmp_global = PhotographicGlobal(hdr, delta = 1e-10, a = a , white = 200)
cv.imwrite('tone_mapped_ldr_global.jpg', tnmp_global * 255)
tnmp_local = PhotographicLocal(hdr, delta = 1e-10, a = a, white = 100, phi = 15.0, epsilon = 0.05, s_range = 11)
cv.imwrite('tone_mapped_ldr_local.jpg', tnmp_local * 255)