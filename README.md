# coral-pi-mqtt-server
Expose deep learning models on a Coral usb accelerator via MQTT on a raspberry pi.

## Coral Accelerator Official code
* The edgetpu code doesn't appear to live on Github, but is downloaded from a google machine. I have placed some of the python files from that library in this repo for reference. In particular `edgetpu/classification/engine.py` contains the `ClassificationEngine` class used to perform inferencing. The good news is that the only python dependencies (apart from some device specific code) are `numpy` and `PIL`.
* [Coral Python API & demos](https://coral.withgoogle.com/docs/edgetpu/api-intro/)

## Models
* The official pre-compiled models are at -> https://coral.withgoogle.com/models/
* It is [also possible to train your own models](https://coral.withgoogle.com/tutorials/edgetpu-models-intro/) -> try using Google Colaboratory as the free environment for training or -> https://cloud-annotations.github.io/training/object-detection/cli/index.html

## MQTT
MQTT is very common in IOT and simpler to implement than rest. We will need to have a camera source that is publishing images over MQTT (e,g, in Home-Assistant we will need a custom image_processing component)

## References
* [Using the official pi camera with Coral](https://github.com/nickoala/edgetpu-on-pi)
* https://github.com/snmsung716/Following-camera-with-Coral-Accelerator-on-Raspberry-Pi
* https://github.com/lemariva/raspbian-EdgeTPU
* [Pyimagesearch article on Coral](https://www.pyimagesearch.com/2019/04/22/getting-started-with-google-corals-tpu-usb-accelerator/)
* [picamera-mqtt](https://github.com/ethanjli/picamera-mqtt)
