import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np


print("Tensorflow Version: "+tf.__version__)
print("\nNumpy Version: "+np.__version__)


print("\nis_built_with_cuda: ", tf.test.is_built_with_cuda() )

print(f"\ntf.test.is_gpu_available: {tf.test.is_gpu_available()}")

print(f"\ntf.test.gpu_device_name: {tf.test.gpu_device_name()}")




print(f"\nlist_local_devices: \n{device_lib.list_local_devices()}")

print("\nDevices: ")
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

visible_devices = tf.config.get_visible_devices()
for devices in visible_devices:
    print(devices)