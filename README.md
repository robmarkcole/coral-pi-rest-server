# coral-pi-rest-server
Expose deep learning models on a Coral usb accelerator a Flask app on a raspberry pi.

## Coral Accelerator
* The edgetpu code doesn't appear to live on Github, but is downloaded from a google machine. The good news is that the only python dependencies (apart from some device specific code) are `numpy` and `PIL`.
* [Coral Python API & demos](https://coral.withgoogle.com/docs/edgetpu/api-intro/)
* The official pre-compiled models are at -> https://coral.withgoogle.com/models/
* It is [also possible to train your own models](https://coral.withgoogle.com/tutorials/edgetpu-models-intro/) -> try using Google Colaboratory as the free environment for training or -> https://cloud-annotations.github.io/training/object-detection/cli/index.html

## Flask app
* We will be forking the code in this excellent article -> https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
* The Flask app exposes a rest end point which is easy to call using CURL or using python requests. 

## References
* https://github.com/google-coral
* [Using the official pi camera with Coral](https://github.com/nickoala/edgetpu-on-pi)
* https://github.com/snmsung716/Following-camera-with-Coral-Accelerator-on-Raspberry-Pi
* https://github.com/lemariva/raspbian-EdgeTPU
* [Pyimagesearch article on Coral](https://www.pyimagesearch.com/2019/04/22/getting-started-with-google-corals-tpu-usb-accelerator/)
* [Hands on with coral + docker](https://lemariva.com/blog/2019/04/edge-tpu-coral-usb-accelerator-dockerized)
* [Official Raspicam example](https://github.com/google-coral/examples-camera/blob/master/raspicam/classify_capture.py)
