Expose deep learning models on a Coral usb accelerator via a Flask app. To run the app and expose over a network: 
```
$ python3 coral-app.py
```
Then use curl to query:
```
$curl -X POST -F image=@people_car.jpg 'http://localhost:5000/predict'
 
response = {'predictions': [
   {'bounding_box': {'x1': 0.87, 'x2': 0.28, 'y1': 0.96, 'y2': 0.85},
   'confidence': '95.31',
   'label': 'person'},
  {'bounding_box': {'x1': 0.32, 'x2': 0.24, 'y1': 0.64, 'y2': 0.75},
   'confidence': '91.02',
   'label': 'car'},
  {'bounding_box': {'x1': 0.24, 'x2': 0.3, 'y1': 0.37, 'y2': 0.78},
   'confidence': '58.2',
   'label': 'person'},
  {'bounding_box': {'x1': 0.22, 'x2': 0.44, 'y1': 0.36, 'y2': 0.87},
   'confidence': '26.95',
   'label': 'bicycle'},
  {'bounding_box': {'x1': 0.44, 'x2': 0.36, 'y1': 0.47, 'y2': 0.44},
   'confidence': '21.09',
   'label': 'person'},
  {'bounding_box': {'x1': 0.37, 'x2': 0.34, 'y1': 0.41, 'y2': 0.43},
   'confidence': '21.09',
   'label': 'person'},
  {'bounding_box': {'x1': 0.32, 'x2': 0.31, 'y1': 0.36, 'y2': 0.44},
   'confidence': '21.09',
   'label': 'person'},
  {'bounding_box': {'x1': 0.01, 'x2': 0.02, 'y1': 0.02, 'y2': 0.09},
   'confidence': '21.09',
   'label': 'traffic light'},
  {'bounding_box': {'x1': 0.25, 'x2': 0.54, 'y1': 0.34, 'y2': 0.85},
   'confidence': '16.02',
   'label': 'bicycle'},
  {'bounding_box': {'x1': 0.31, 'x2': 0.32, 'y1': 0.37, 'y2': 0.62},
   'confidence': '16.02',
   'label': 'person'}],
 'success': True}
```

See the [Jupyter notebook](https://github.com/robmarkcole/coral-pi-rest-server/blob/master/coral-app-usage.ipynb) for usage with python requests library.

**NOTE:** you need to update the `MODEL` and `LABELS` file paths in `coral-app.py`. For compatability with the way these paths are hard coded in this repo, you can on a pi `cd ~`, `mkdir edgetpu`, `mkdir all_models`, `cd all_models`, `wget https://dl.google.com/coral/canned_models/all_models.tar.gz`, `tar xf all_models.tar.gz`, `rm all_models.tar.gz`

## Pi setup
I am running the server on a pi 3 with the raspi camera below. FYI the camera is mounted on a [pan-tilt stage](https://shop.pimoroni.com/products/pan-tilt-hat).

<p align="center">
<img src="https://github.com/robmarkcole/coral-pi-rest-server/blob/master/images/my_setup.png" width="500">
</p>

## Models
* The official pre-compiled models are at -> https://coral.withgoogle.com/models/
* It is [also possible to train your own models](https://coral.withgoogle.com/tutorials/edgetpu-models-intro/) -> try using Google Colaboratory as the free environment for training or -> https://cloud-annotations.github.io/training/object-detection/cli/index.html

## To do
* Simple front end for uploading images -> https://github.com/gxercavins/image-api 
* Handle traffic -> https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/

## Home Assistant
I have published code for using this app with Home Assistant -> [HASS-Google-Coral](https://github.com/robmarkcole/HASS-Google-Coral)

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
