from __future__ import print_function
from __future__ import division
from math import exp
import cv2 as cv
import numpy as np
import argparse
import os
from mtb import *
from robertson import *
from Debevec import *
from tonemap_photographic import *

def loadExposureSeq(path):
    images = []
    times = []
    with open(os.path.join(path, 'list.txt')) as f:
        content = f.readlines()
    for line in content:
        tokens = line.split()
        image = cv.imread(os.path.join(path, tokens[0]))
        image = cv.resize(image, (0, 0), fx=0.05, fy=0.05)
        images.append(image)
        #images.append(cv.imread(os.path.join(path, tokens[0])))
        times.append(float(tokens[1]))
    return images, np.asarray(times, dtype=np.float32)


parser = argparse.ArgumentParser(description='Code for VFX project 1.')
parser.add_argument('--input', type=str, help='Path to the directory that contains images and exposure times.')
args = parser.parse_args()
if not args.input:
    parser.print_help()
    exit(0)

## load in the images
print("Loading images...")
images, times = loadExposureSeq(args.input)

## align images with MTB
#alignment(images)

## calculate hdr

print("Calculating...")
hdr = HDR(images, times)
plt.imshow(hdr)
plt.savefig("HDR.png")
plt.show()
cv.imwrite('hdr.hdr', hdr)

## tonemapping with Reinhard

print("Tonemapping...")
a = 0.18
tnmp_global = PhotographicGlobal(hdr, delta = 1e-10, a = a , white = 100)
cv.imwrite('tone_mapped_ldr_global.jpg', tnmp_global * 255)
tnmp_local = PhotographicLocal(hdr, delta = 1e-10, a = a, phi = 8.0, epsilon = 0.05, s_range = 11)
cv.imwrite('tone_mapped_ldr_local.jpg', tnmp_local * 255)