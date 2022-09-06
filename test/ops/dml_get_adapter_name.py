"""
Helper that outputs the logs of DirectML device creation
"""

from tensorflow.python.client import device_lib

if __name__ == "__main__":
    device_lib.list_local_devices()
