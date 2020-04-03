import cv2 as cv
import numpy as np
import argparse
import os
from matplotlib import pyplot as plt

#### THIS FUNCTION IS USED TO CALCULATE THE WEIGHT OF DIFFERENT BIT VALUE
# w(z) is weighting function value for pixel value z
def weight(value):
	v_max = 255
	v_min = 0
	v_mid = (v_max+v_min)//2
	if value > v_mid:
		return v_max - value
	else:
		return value - v_min

def loadExposureSeq(path):
    images = []
    times = []
    with open(os.path.join(path, 'list2.txt')) as f:
        content = f.readlines()
    for line in content:
        tokens = line.split()
        image = cv.imread(os.path.join(path, tokens[0]))
        image = cv.resize(image, (0,0), fx=0.2,fy=0.2)
        images.append(image)
        times.append(float(tokens[1]))

    return images, np.asarray(times, dtype=np.float32)

def bgr2gray(bgr):
    return np.dot(bgr[...,:3], [0.1140, 0.5870, 0.2989])
   
def grayscale(images):
    gray = []
    for img in images:
        gray.append(bgr2gray(img))
    return gray

def radianceMap(g,B,images,r,c):
	images = np.array(images)
	result = np.zeros((r,c))

	for i in range(r):
		for j in range(c):
			a = 0
			b = 0
			for p in range(images.shape[0]):
				w = weight(images[p][i][j])
				a += w*(g[int(images[p][i][j])]-B[p])
				b += w
			if b ==0:
				result[i,j] = 0
			else:
				result[i,j] = a/b
	return result

def logTimes(times):
	times = np.log(times)
	return times

def ginitial():
	g = np.ones(256,dtype=np.float32)
	for i in range(256):
		g[i] = i/128.0
	return g

def reshapePoint(images):
	images = np.asarray(images)
	images = np.reshape(images,(images.shape[0],-1))
	return images

def optimize_E(g,Z,B,E):
	for i in range(E.shape[0]):
		a = 0.0
		b = 0.0
		for j in range(Z.shape[0]):
			w = weight(Z[j][i])
			a += w * g[int(Z[j][i])]*B[j]
			b += w*(B[j]**2)
		if b==0:
			continue
		else:
			E[i] = a/b
	return E

def optimize_g(g,Z,B,E):
	for v in range(256):
		j,i = np.where(Z[:,:]==v)
		b = len(j)
		if b == 0:
			continue
		a = 0
		for k in range(b):
			a +=  E[i[k]]*B[j[k]]
		g[v] = a/b
	g /= g[128]
	return g

def HDR2(images, times):
	itertimes = 10
	images = grayscale(images)
	r,c = images[0].shape
	#print(r,c)
	Z = reshapePoint(images)  #Z(ij) is the pixel values of pixel location number i in image j
	B = logTimes(times) # B[j] is the log delta t for image j
	g = ginitial()
	E = np.zeros(images[0].shape[0]*images[0].shape[1],dtype=np.float32)
	for i in range(itertimes):
		print(i)
		E = optimize_E(g,Z,B,E)
		g = optimize_g(g,Z,B,E)

	HDR_image = radianceMap(g,B,images,r,c)

	return HDR_image
	
if __name__ == '__main__':
	## Parse the arguemnt and Load images
	parser = argparse.ArgumentParser(description='Code for HDR.')
	parser.add_argument('--input', type=str, help='Path to the directory that contains images and exposure times.')
	args = parser.parse_args()
	if not args.input:
		parser.print_help()
		exit(0)
	
	## load in the images
	print("Loading images...")
	images, times = loadExposureSeq(args.input)
	
	## calculate HDR
	print("Calculating...")
	HDR_image = HDR2(images, times)
	plt.imshow(HDR_image)
	plt.savefig("HDR2.png")
	plt.show()


