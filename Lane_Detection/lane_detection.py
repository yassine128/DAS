import image_processing as ip
import cv2
import numpy as np
from PIL import ImageGrab

kernel = ip.gaussian_kernel(2)
bounding_box = (0, 0, 600, 600)

while (True): 
    img = ImageGrab.grab(bbox=bounding_box)
    img_np = np.array(img)
    grey_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    cv2.imshow('screen', np.array(ip.pipeline(grey_img, kernel)))

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        cv2.destroyAllWindows() 
        break