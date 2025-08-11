import easyocr
import cv2

use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
reader = easyocr.Reader(['en'], gpu=use_gpu)
