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

## Pi setup
Install one of the disk images from [edgetpu-platforms](https://github.com/google-coral/edgetpu-platforms). In the `/home/pi` directory `git clone` this repository. You wil now have a file stucture like:
```
pi@raspberrypi:~ $ ls

all_models  coral-pi-rest-server  edgetpu_api  examples-camera  project-posenet  project-teachable  simple-demo
```

Use the `cd` command to enter `coral-pi-rest-server` and (system wide, no viretual environment) install the required dependencies:
```
pi@raspberrypi:~/coral-pi-rest-server $ pip3 install -r requirements.txt
```
Now run the app:
```
pi@raspberrypi:~/coral-pi-rest-server $ python3 coral-app.py
```

I am running on a pi3 with the raspi camera below. FYI the camera is mounted on a [pan-tilt stage](https://shop.pimoroni.com/products/pan-tilt-hat).

<p align="center">
<img src="https://github.com/robmarkcole/coral-pi-rest-server/blob/master/images/my_setup.png" width="500">
</p>

Using the pi3 (which has only usb2) processing a single request from another machine via this server takes ~ 300 to 500 ms. Therefore this is certainly fast enough to process images at 1 FPS which in my experience is suitable for tasks such as counting people in a room.

## Models
If you have installed the raspberry pi disk images from edgetpu-platforms then you already have all the models in `home/pi/all_models`. If you are using a linux desktop you will need to download the models.
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
