Expose deep learning models on a Coral usb accelerator via a Flask app. To run the app and expose over a network: 
```
$ python3 coral-app.py
```
Then use curl to query:
```
curl -X POST -F image=@images/test-image3.jpg 'http://localhost:5000/v1/vision/detection'

{'predictions': [{'confidence': 0.953125,
   'label': 'person',
   'x_max': 918,
   'x_min': 838,
   'y_max': 407,
   'y_min': 135},
  {'confidence': 0.58203125,
   'label': 'person',
   'x_max': 350,
   'x_min': 226,
   'y_max': 374,
   'y_min': 143}]
 'success': True}
```

See the [Jupyter notebook](https://github.com/robmarkcole/coral-pi-rest-server/blob/master/usage/coral-app-usage.ipynb) for usage with python requests library.

## Models 
You need to update the `MODELS_DIR`, `MODEL` and `LABELS` paths in `coral-app.py`. For compatability with the way these paths are hard coded in this repo, you can on a pi `cd ~`, `mkdir edgetpu`, `mkdir all_models`, `cd all_models`, `wget https://dl.google.com/coral/canned_models/all_models.tar.gz`, `tar xf all_models.tar.gz`, `rm all_models.tar.gz`

## Pi setup
I am running the server on a pi 3 with the raspi camera below. FYI the camera is mounted on a [pan-tilt stage](https://shop.pimoroni.com/products/pan-tilt-hat).

<p align="center">
<img src="https://github.com/robmarkcole/coral-pi-rest-server/blob/master/images/my_setup.png" width="500">
</p>

## Models
* The official pre-compiled models are at -> https://coral.withgoogle.com/models/
* It is [also possible to train your own models](https://coral.withgoogle.com/tutorials/edgetpu-models-intro/) -> try using Google Colaboratory as the free environment for training or -> https://cloud-annotations.github.io/training/object-detection/cli/index.html

## Deepstack & Home Assistant
The data returned by the app is as close as possible in format to that returned by Deepstack object detection endpoint, allowing you to use this app as the backend for [HASS-Deepstack-object](https://github.com/robmarkcole/HASS-Deepstack-object)

## References
* https://github.com/google-coral
* [Using the official pi camera with Coral](https://github.com/nickoala/edgetpu-on-pi)
* https://github.com/snmsung716/Following-camera-with-Coral-Accelerator-on-Raspberry-Pi
* https://github.com/lemariva/raspbian-EdgeTPU
* [Pyimagesearch article on Coral](https://www.pyimagesearch.com/2019/04/22/getting-started-with-google-corals-tpu-usb-accelerator/)
* [Hands on with coral + docker](https://lemariva.com/blog/2019/04/edge-tpu-coral-usb-accelerator-dockerized)
* [Official Raspicam example](https://github.com/google-coral/examples-camera/blob/master/raspicam/classify_capture.py)

## Credit
I forked the code in this excellent article -> https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html [code](https://github.com/jrosebr1/simple-keras-rest-api)
