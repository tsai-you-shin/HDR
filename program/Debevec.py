import cv2 as cv
import numpy as np
import argparse
import os
from matplotlib import pyplot as plt


#### THIS FUNCTION IS USED TO CALCULATE THE WEIGHT OF DIFFERENT BIT VALUE
def weight(value):  # w(z) is weighting function value for pixel value z
    v_max = 255
    v_min = 0
    v_mid = (v_max + v_min) // 2
    if value > v_mid:
        return v_max - value
    else:
        return value - v_min


###

def bgr2gray(bgr):
    return np.dot(bgr[..., :3], [0.1140, 0.5870, 0.2989])


def grayscale(images):
    gray = []
    for img in images:
        gray.append(bgr2gray(img))
    return gray


def radianceMap(g, B, images):
    images = np.array(images)
    result = np.zeros((images.shape[1], images.shape[2], 3))
    for channel in range(3):
        for i in range(images.shape[1]):
            for j in range(images.shape[2]):
                a = 0
                b = 0
                for p in range(images.shape[0]):
                    a += weight(images[p, i, j, channel]) * (g[int(images[p, i, j, channel]),channel] - B[p])
                    b += weight(images[p, i, j, channel])
                result[i, j, channel] = a/b

    result[np.isnan(result)] = 0
    result = np.exp(result)
    result[np.isnan(result)] = 0
    result[np.isinf(result)] = 0
    print("checking hdr...")
    result_max = np.amax(result)
    for channel in range(3):
        for i in range(images.shape[1]):
            for j in range(images.shape[2]):
                if result[i, j, channel] <= 1:
                    #print("hi")
                    for p in range(images.shape[0]):
                        if images[p, i, j, channel] > 200:
                            #print("bingo")
                            result[i, j, channel] = result_max
    return result


#### THIS FUNCTION IS USED TO CHOOSE POINTS
def choosePoint(images):
    images = np.array(images)
    numOfImages = len(images)
    print(images.shape)
    result = np.zeros((256, numOfImages))

    for i in range(0, 256, 8):
        r, c = np.where((images[numOfImages // 2, :, :] >= i - 0.5) & (images[numOfImages // 2, :, :] <= i + 0.5))
        if len(r) != 0:
            # print(r.shape)
            index = np.random.randint(len(r), size=1)
            r = r[index][0]
            c = c[index][0]
            # print(r,c)
            result[i, :] = images[:, r, c]
        else:
            print("no pixel value", i)
            result[i, :] = -1
    return result


####

def logTimes(times):
    times = np.log(times)
    return times


def gsolve(Z, B, L):
    n = 256  # pixel value range 0-255

    A = np.zeros((Z.shape[1] * Z.shape[0] + 1 + (n - 2), n + Z.shape[0]), dtype=np.float32)
    b = np.zeros((np.size(A, 0), 1), dtype=np.float32)

    k = 1
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            w = weight(Z[i][j])
            A[k][int(Z[i][j])] = w
            A[k][n + i] = -w
            b[k][0] = w * B[j]
            k += 1

    A[k][127] = 1
    k += 1

    for i in range(1, n - 2):
        w = weight(i)
        A[k][i - 1] = L * w
        A[k][i] = -2 * L * w
        A[k][i + 1] = L * w
        k += 1

    invA = np.linalg.pinv(A)
    x = np.dot(invA, b)
    return np.ravel(x[0:n])

def take_one_channel(images, channel):
	one_channel = np.zeros([len(images), len(images[0]), len(images[0][0])])
	for i in range(len(images)):
		one_channel[i] = images[i][ :, :, channel]

	return  one_channel
def HDR(images, times):
    B = logTimes(times)  # B[j] is the log delta t for image j
    ##### THIS PARAMETER CAN BE TRY
    L = float(100)  # L is lambda, the constant that determines the amout of smoothness
    #####
    print('calculating camera response function by gsolve.')
    g = np.zeros([256, 3])

    for channel in range(3):
        Z = choosePoint(take_one_channel(images, channel))  # Z(ij) is the pixel values of pixel location number i in image j
        g[:, channel] = gsolve(Z, B, L)
    print(g)
    print("constructing radience map...")
    HDR_image = radianceMap(g, B, images)
    return HDR_image
