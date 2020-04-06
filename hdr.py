import cv2 as cv
import numpy as np
import argparse
import os
from matplotlib import pyplot as plt

#### THIS FUNCTION IS USED TO CALCULATE THE WEIGHT OF DIFFERENT BIT VALUE
def weight(value):# w(z) is weighting function value for pixel value z
	v_max = 255
	v_min = 0
	v_mid = (v_max+v_min)//2
	if value > v_mid:
		return v_max - value
	else:
		return value - v_min
###
def loadExposureSeq(path):
    images = []
    times = []
    with open(os.path.join(path, 'list.txt')) as f:
        content = f.readlines()
    for line in content:
        tokens = line.split()
        images.append(cv.imread(os.path.join(path, tokens[0])))
        times.append(float(tokens[1]))

    return images, np.asarray(times, dtype=np.float32)

def bgr2gray(bgr):
    return np.dot(bgr[...,:3], [0.1140, 0.5870, 0.2989])
   
def grayscale(images):
    gray = []
    for img in images:
        gray.append(bgr2gray(img))
    return gray

def radianceMap(g,B,images):
	images = np.array(images)
	result = np.zeros((images.shape[1],images.shape[2]))

	for i in range(images.shape[1]):
		for j in range(images.shape[2]):
			a = 0
			b = 0
			for p in range(images.shape[0]):
				a += weight(images[p,i,j])*(g[int(images[p,i,j])]-B[p])
				b += weight(images[p,i,j])
			result[i,j] = a/b

	return result

#### THIS FUNCTION IS USED TO CHOOSE POINTS
def choosePoint(images):
	images = np.array(images)
	numOfImages = len(images)
	print(images.shape)
	result = np.zeros((256,numOfImages))

	for i in range(0,256,8):
		r,c = np.where((images[numOfImages//2,:,:]>=i-0.5)&(images[numOfImages//2,:,:]<=i+0.5))
		if len(r) != 0:
			#print(r.shape)
			index = np.random.randint(len(r), size = 1)
			r = r[index][0]
			c = c[index][0]
			#print(r,c)
			result[i,:] = images[:,r,c]
		else:
			print("no pixel value", i)
			result[i,:] = -1
	return result
####

def logTimes(times):
	times = np.log(times)
	return times

def gsolve(Z,B,L):
	n = 256 #pixel value range 0-255

	A = np.zeros((Z.shape[1]*Z.shape[0]+1+(n-2),n+Z.shape[0]), dtype=np.float32)
	b = np.zeros((np.size(A,0),1), dtype=np.float32)

	k = 1
	for i in range(Z.shape[0]):
		for j in range(Z.shape[1]):
			w = weight(Z[i][j])
			A[k][int(Z[i][j])] = w
			A[k][n+i] = -w
			b[k][0] = w*B[j]
			k += 1

	A[k][127] = 1
	k+=1

	for i in range(1,n-2):
		w = weight(i)
		A[k][i-1] = L*w
		A[k][i] = -2*L*w
		A[k][i+1] = L*w
		k+=1

	invA = np.linalg.pinv(A)
	x = np.dot(invA,b)

	return x[0:n-1]


def HDR(images, times):

	images = grayscale(images)
	Z = choosePoint(images)  #Z(ij) is the pixel values of pixel location number i in image j
	B = logTimes(times) # B[j] is the log delta t for image j

##### THIS PARAMETER CAN BE TRY
	L = float(100) # L is lambda, the constant that determines the amout of smoothness
#####
	g = gsolve(Z,B,L)
	HDR_image = radianceMap(g,B,images)

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
	HDR_image = HDR(images, times)
	plt.imshow(HDR_image)
	plt.savefig("HDR.png")
	plt.show()


