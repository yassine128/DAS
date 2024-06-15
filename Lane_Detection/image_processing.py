import numpy as np
from scipy import ndimage
import cv2
import utils

def kernel_val(x: int, y: int, sigma: int) -> float:
    return np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

def gaussian_kernel(sigma: int):
    kernel_size = 6 * sigma + 1 
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    total = 0.0
    mid = kernel_size // 2

    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - mid
            y = j - mid
            coeff = kernel_val(x, y, sigma)
            
            kernel[i, j] = coeff
            total += coeff

    kernel /= total
    return kernel

def gaussian_blur(img, kernel):
    return cv2.filter2D(img, -1, kernel)

def gradient_intensity(image):
    y_half = image.shape[1] // 2

    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobelx**2 + sobely**2)
    angle = np.arctan2(sobely, sobelx) * 180 / np.pi  
    return magnitude, angle

def non_maximum_suppression(magnitude, angle):
    M, N = magnitude.shape
    output = np.zeros((M, N), dtype=np.float32)
    angle = angle % 180 

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    val1, val2 = magnitude[i, j+1], magnitude[i, j-1]
                elif (22.5 <= angle[i, j] < 67.5):
                    val1, val2 = magnitude[i+1, j-1], magnitude[i-1, j+1]
                elif (67.5 <= angle[i, j] < 112.5):
                    val1, val2 = magnitude[i+1, j], magnitude[i-1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    val1, val2 = magnitude[i-1, j-1], magnitude[i+1, j+1]

                if magnitude[i, j] >= val1 and magnitude[i, j] >= val2:
                    output[i, j] = magnitude[i, j]
            except IndexError as e:
                pass

    return output
        
def hysteresis_thresholding(img, low_threshold, high_threshold):
    M, N = img.shape
    output = np.zeros((M,N), dtype=np.int32)

    strong = 255
    weak = 75

    strong_i, strong_j = np.where(img >= high_threshold)
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    output[strong_i, strong_j] = strong
    output[weak_i, weak_j] = weak

    for i in range(1, M-1):
        for j in range(1, N-1):
            if output[i, j] == weak:
                if (output[i+1, j-1] == strong or output[i+1, j] == strong or output[i+1, j+1] == strong
                    or output[i, j-1] == strong or output[i, j+1] == strong
                    or output[i-1, j-1] == strong or output[i-1, j] == strong or output[i-1, j+1] == strong):
                    output[i, j] = strong
                else:
                    output[i, j] = 0

    return output

def adaptive_thresholds(magnitude):
    high_threshold = np.median(magnitude) * 2
    low_threshold = high_threshold / 2
    return low_threshold, high_threshold

def apply_mask(img, mask): 
    return cv2.bitwise_and(img, mask)

def hough_transform(img): 
    rho = 1			
    theta = np.pi/180
    threshold = 20	
    minLineLength = 200
    maxLineGap = 500
    return cv2.HoughLinesP(img, rho = rho, theta = theta, threshold = threshold,
						minLineLength = minLineLength, maxLineGap = maxLineGap)

def draw_lines(image, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0] 

            m = (y2 - y1) / (x2 - x1)
            if abs(m) > 1:  
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    return image

def pipeline(img, kernel): 
    img_blur = gaussian_blur(img, kernel)
    magnitude, angle = gradient_intensity(img_blur)
    out = non_maximum_suppression(magnitude, angle)
    low, high = adaptive_thresholds(magnitude)
    out = hysteresis_thresholding(out, low, high).astype(np.uint8)
    mask = utils.create_road_mask(out)
    img_masked = apply_mask(out, mask)
    coord = hough_transform(img_masked)
    img_f = draw_lines(img, coord)
    return img_f
