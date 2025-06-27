import platform
import subprocess
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np


print("Python: "+platform.python_version())
print("Tensorflow Version: "+tf.__version__)
print("Numpy Version: "+np.__version__)
print("is_built_with_cuda: ", tf.test.is_built_with_cuda() )
print(f"tf.test.is_gpu_available: {tf.test.is_gpu_available()}")
print("--------------------------------------")

print(f"\ntf.test.gpu_device_name: {tf.test.gpu_device_name()}")
print("--------------------------------------")

print(f"\nlist_local_devices: \n{device_lib.list_local_devices()}")
print("--------------------------------------")

print("\nDevices: ")
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

visible_devices = tf.config.get_visible_devices()
for devices in visible_devices:
    print(devices)
print("--------------------------------------")

print("\nNvidia SMI:")
try:
    print(subprocess.check_output(["nvidia-smi"]).decode("utf-8"))
except Exception as ex:
    print(f"Error fetching Nvidia GPU details: {ex}")

print("\n\nCompleted!")