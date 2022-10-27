## coral-pi-rest-server
Perform inference using tensorflow-lite deep learning models with hardware acceleration provided by a Coral usb accelerator running on a raspberry pi or linux/mac. The models are exposed via a REST API allowing inference over a network. To run the app with default model:
```
$ python3 coral-app.py --models-directory models
```
Then use curl to query:
```
curl -X POST -F image=@images/test-image3.jpg -F min_confidence=.5 'http://localhost:5000/v1/vision/detection'

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

To see the help run:
```
$ python3 coral-app.py -h
```

See the [coral-app-usage.ipynb](https://github.com/robmarkcole/coral-pi-rest-server/blob/master/coral-app-usage.ipynb) Jupyter notebook for usage from python.

**Warning**: it has been quite frustrating working with this hardware, with many tedious setup problems, hardware timeouts. Therefore I created a fork of this project that does not require a Coral at https://github.com/robmarkcole/tensorflow-lite-rest-server On an RPI 4 inference times are very fast even without a coral.

## Models
If you have installed the raspberry pi disk images from edgetpu-platforms then you already have all the models in `home/pi/all_models`. If you are using a mac/linux desktop you can download the models [from here](https://github.com/google-coral/edgetpu/tree/master/test_data). It is [also possible to train your own models](https://coral.withgoogle.com/tutorials/edgetpu-models-intro/) -> try using Google Colaboratory as the free environment for training or -> https://cloud-annotations.github.io/training/object-detection/cli/index.html

## Pi setup on existing pi
Follow the instructions on -> https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime Note I had headaches with this approach, better to use the disk image.

## Pi setup with disk image
Install one of the disk images from [edgetpu-platforms](https://github.com/google-coral/edgetpu-platforms). In the `/home/pi` directory `git clone` this repository. You wil now have a file structure like (pi3 &pi4 only, pi-zero [differs](https://github.com/google-coral/edgetpu-platforms/issues/13)):
```
$ ls

all_models  coral-pi-rest-server  edgetpu_api  examples-camera  project-posenet  project-teachable  simple-demo
```

Use the `cd` command to enter `~/simple-demo` and test you installation:
```
pi@pi:~/simple-demo $ python3 classify_image.py --model /home/pi/all_models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite --label  /home/pi/all_models/inat_bird_labels.txt --image  parrot.jpg

INFO: Initialized TensorFlow Lite runtime.
---------------------------
Ara macao (Scarlet Macaw)
Score :  0.76171875
```

Assuming you saw the result above, use the `cd` command to enter `~/coral-pi-rest-server` and (system wide, no virtual environment) install the required dependencies:
```
~/coral-pi-rest-server $ pip3 install -r requirements.txt
```
Now run the app:
```
~/coral-pi-rest-server $ python3 coral-app.py
```
### Service
You can run the app as a [service](https://www.raspberrypi.org/documentation/linux/usage/systemd.md). Edit `coral.service` file to fit your needs and copy into `/etc/systemd/system` as root using `sudo cp coral.service /etc/systemd/system/coral.service`. Once this has been copied, you can attempt to start the service using `sudo systemctl start coral.service`. See the status and logs with `sudo systemctl status coral.service`. Stop the service with `sudo systemctl stop coral.service`. You can have it start automatically on reboot by using `sudo systemctl enable coral.service`.

### Pi 3
I am running a pi3 with the raspi camera below. FYI the camera is mounted on a [pan-tilt stage](https://shop.pimoroni.com/products/pan-tilt-hat).

<p align="center">
<img src="https://github.com/robmarkcole/coral-pi-rest-server/blob/master/images/my_setup.png" width="500">
</p>

Using the pi3 (which has only usb2) processing a single request from another machine via this server takes ~ 300 to 500 ms. Therefore this is certainly fast enough to process images at 1 FPS which in my experience is suitable for tasks such as counting people in a room.

### Pi 4
The pi4 has USB3 so we would expect better speeds. However I get inference time consistently ~ 500 ms when querying from another machine. Therefore it appears the coral library does not yet take advantage of the USB3.

## Pi-zero
I found that I get inference time in the range 2.5 to 5s when querying from another machine, so significantly slower than the pi3/pi4.

## Power
I recommend using a power supply which can supply 3 Amps, [I use this one](https://www.amazon.co.uk/gp/product/B017YW2CKM/ref=ppx_yo_dt_b_asin_title_o00_s00?ie=UTF8&psc=1). Note that the official RPI supply delivers only 2.5 Amp and I found this was causing an issue where the stick would 'go to sleep' after a day of continual use.

## Mac
To use the coral with a mac follow the official install instructions [here](https://coral.ai/docs/accelerator/get-started/#on-mac)

## No coral?
I created a fork of this project that does not require a Coral at https://github.com/robmarkcole/tensorflow-lite-rest-server

## Deepstack & Home Assistant
The data returned by the app is as close as possible in format to that returned by Deepstack object detection endpoint, allowing you to use this app as the backend for [HASS-Deepstack-object](https://github.com/robmarkcole/HASS-Deepstack-object). There is also support for running as an addon via [github.com/grinco/HASS-coral-rest-api](https://github.com/grinco/HASS-coral-rest-api)

## Troubleshooting
-----------------------------
Q: I get the error: `HandleQueuedBulkIn transfer in failed. Not found: USB transfer error 5 [LibUsbDataInCallback]`
A: I reflashed the SD card and tried again with success

Q: I get error `ValueError: Failed to load delegate from libedgetpu.1.dylib`
A: libedgetpu is not in the expected location

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

## Development Mac
Note: important to use USB-C to USB-C cable. Had some connection issues initially documented in [this issue](https://github.com/google-coral/pycoral/issues/35)

* `python3 -m venv venv`
* `source venv/bin/activate`
* `pip install -r requirements.txt`
* `python3 -m pip install --index-url https://google-coral.github.io/py-repo/ --extra-index-url=https://pypi.python.org/simple pycoral`
* `python3 coral-app.py`
