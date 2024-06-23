import cv2
import numpy as np 
import matplotlib.pyplot as plt

class LaneDetection: 
    def new_img(self, img): 
        self.img = img
        self.original = self.img
        self.img = np.copy(self.img)
        self.preprocess()

    def adjust_gamma(self, image, gamma): 
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def isolate_color_mask(self, img, low_thresh, high_thresh):
        return cv2.inRange(img, low_thresh, high_thresh)
    
    def to_hls(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    def preprocess(self): 
        darkened_img = self.adjust_gamma(self.img, 0.5)
        white_mask = self.isolate_color_mask(self.to_hls(self.img), np.array([0, 200, 0], dtype=np.uint8), np.array([200, 255, 255], dtype=np.uint8))
        yellow_mask = self.isolate_color_mask(self.to_hls(self.img), np.array([10, 0, 100], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8))
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        self.img = cv2.bitwise_and(darkened_img, darkened_img, mask=mask)
        self.smooth()

    def darken_image(self):
        self.img = cv2.convertScaleAbs(self.img, alpha=0.5, beta=0) 
        self.smooth()

    def smooth(self): 
        self.img = cv2.GaussianBlur(self.img, (3,3), 0)
        self.canny_func()
        
    def canny_func(self): 
        self.img = cv2.Canny(self.img, 50, 150)
        self.canny = self.img
        self.roi()

    def roi(self): 
        rows, cols = self.img.shape[:2]
        self.mask = np.zeros_like(self.img)
        
        left_bottom = [cols * 0.1, rows]
        right_bottom = [cols * 0.95, rows]
        left_top = [cols * 0.4, rows * 0.6]
        right_top = [cols * 0.6, rows * 0.6]
        
        vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
        
        if len(self.mask.shape) == 2:
            cv2.fillPoly(self.mask, vertices, 255)
        else:
            cv2.fillPoly(self.mask, vertices, (255, ) * self.mask.shape[2])
        self.img = cv2.bitwise_and(self.img, self.mask)
        self.hough()
    
    def hough(self): 
        self.lines = cv2.HoughLinesP(self.img, 1, np.pi/180, 20, np.array([]), minLineLength=20, maxLineGap=300)
        if len(self.lines) > 0: 
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

        if self.avg_lines is not None:
            for x1, x2, y1, y2 in self.avg_lines:
                cv2.line(self.line_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 10)
    
    def show_lanes(self): 
        plt.imshow(self.line_image)
    
    def show_final(self): 
        line = self.line_image[..., np.newaxis]
        return self.original | line

    def show_mask(self): 
        plt.imshow(self.mask)

    
    def show_canny(self): 
        return self.canny