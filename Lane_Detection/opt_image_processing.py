import cv2
import numpy as np 
import matplotlib.pyplot as plt
import time

class LaneDetection: 
    def new_img(self, img): 
        self.img = np.copy(img)
        self.smooth()

    def smooth(self): 
        self.img = cv2.GaussianBlur(self.img, (5,5), 0)
        self.canny()
        
    def canny(self): 
        self.img = cv2.Canny(self.img, 50, 150)
        self.roi()

    def roi(self): 
        height = self.img.shape[0]
        width = self.img.shape[1]
        polygons = np.array([
        [(0, height), (width, height), (width//2, height//2)]
        ])
        self.mask = np.zeros_like(self.img)
        cv2.fillPoly(self.mask, polygons, 255)
        self.img = cv2.bitwise_and(self.img, self.mask)
        self.hough()
    
    def hough(self): 
        self.lines = cv2.HoughLinesP(self.img, 2, np.pi/180, 100, np.array([]), minLineLength=10, maxLineGap=5 )
        self.average_slope()

    def make_coordinates(self, line_parameters):
        slope, intercept = line_parameters
        y1 = self.img.shape[0]
        y2 = int(y1*(3/5))

        x1 = ((y1-intercept)/slope)
        x2 = ((y2-intercept)/slope)
        return np.array([x1, x2, y1, y2])

    def average_slope(self): 
        left_fit = []
        right_fit = []
        for line in self.lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2), (y1,y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if  slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = self.make_coordinates(left_fit_average)
        right_line = self.make_coordinates(right_fit_average)
        self.avg_lines = np.array([left_line, right_line])
        self.draw_lines()

    def draw_lines(self): 
        self.line_image = np.zeros_like(self.img)
        if self.lines is not None:
            for x1, y1, x2, y2 in self.lines[0]:
                cv2.line(self.line_image, (x1,y1), (x2,y2), (255, 0, 0), 10)
    
    def show_lanes(self): 
        print(self.avg_lines)
        plt.imshow(self.line_image)
    
    def show_final(self): 
        cv2.imshow("result", cv2.addWeighted(self.img, 0.8, self.line_image, 1, 1))

    def show_mask(self): 
        plt.imshow(self.mask)

