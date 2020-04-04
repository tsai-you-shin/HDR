from statistics import median
import numpy as np
import cv2 as cv

def bgr2gray(bgr):
    return np.dot(bgr[...,:3], [0.1140, 0.5870, 0.2989])

def grayscale(images):
    gray = []
    for img in images:
        gray.append(bgr2gray(img))
    return gray
    
def threshold_bitmap(img):
    m = median(img.ravel())
    tb = cv.threshold(img,m,255,cv.THRESH_BINARY)[1]
    return tb
    
def exclusion_bitmap(img):
    m = median(img.ravel())
    eb = 255 - cv.inRange(img, m - 2, m + 2)
    return eb
    
def MTB( img1, img2, shift_bits, shift_ret):
    cur_shift = np.zeros(2)
    if shift_bits > 0:
        sml_img1 = cv.resize(img1, (0,0), fx=0.5, fy=0.5) 
        sml_img2 = cv.resize(img2, (0,0), fx=0.5, fy=0.5) 
        cur_shift = MTB(sml_img1, sml_img2, shift_bits - 1, cur_shift)
        cur_shift[0] *= 2
        cur_shift[1] *= 2
    else:
        cur_shift[0] = 0
        cur_shift[1] = 0
    tb1 = threshold_bitmap(img1)
    tb2 = threshold_bitmap(img2)
    eb1 = exclusion_bitmap(img1)
    eb2 = exclusion_bitmap(img2)
    min_err = len(img1) * len(img1[0])
    for i in range(-1, 2, 1):
        for j in range(-1, 2, 1):
            xs = cur_shift[0] + i
            ys = cur_shift[1] + j
            translation_matrix = np.float32([ [1,0,xs], [0,1,ys]])
            shifted_tb2 = cv.warpAffine(tb2, translation_matrix, (tb2.shape[1], tb2.shape[0]))
            shifted_eb2 = cv.warpAffine(eb2, translation_matrix, (eb2.shape[1], eb2.shape[0]))
            diff_b = np.logical_xor(tb1, shifted_tb2);
            diff_b = np.logical_and(diff_b, eb1);
            diff_b = np.logical_and(diff_b, shifted_eb2);
            err = np.sum(diff_b)
            if err < min_err: 
                shift_ret[0] = xs
                shift_ret[1] = ys
                min_err = err;
    return shift_ret
    
def alignment(images):
    print("aligning")
    gray = grayscale(images)
    for i in range(len(gray)-1):
        shift_bits = 5
        shift_ret = np.zeros(2)
        xs, ys = MTB(gray[i], gray[i+1], shift_bits, shift_ret)
        print("i:", i, "x: ", xs, ", y: ", ys)
        translation_matrix = np.float32([ [1,0,xs], [0,1,ys]])
        images[i + 1] = cv.warpAffine(images[i + 1], translation_matrix, (images[i].shape[1], images[i].shape[0]))
        gray[i + 1] = cv.warpAffine(gray[i + 1], translation_matrix, (gray[i].shape[1], gray[i].shape[0]))
