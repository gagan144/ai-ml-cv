"""
Ref:
    https://learnopencv.com/opencv-dnn-with-gpu-support/
"""

import cv2


print(f"Opencv version: {cv2.__version__}")

count_cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
print(f"\nNo of cuda enabled devices: {count_cuda_devices}")