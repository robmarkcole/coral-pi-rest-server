Expose deep learning models on a Coral usb accelerator via a Flask app. To run the app and expose over a network: 
```
$ python3 coral-app.py
```
Then use curl to query:
```
$curl -X POST -F image=@people_car.jpg 'http://localhost:5000/predict'
 
{'predictions': [{'bounding_box': {'x1': 838.29,
    'x2': 918.53,
    'y1': 135.01,
    'y2': 407.59},
   'confidence': '95.31',
   'label': 'person'},
  {'bounding_box': {'x1': 302.61, 'x2': 613.57, 'y1': 115.94, 'y2': 361.16},
   'confidence': '91.02',
   'label': 'car'},
  {'bounding_box': {'x1': 226.54, 'x2': 350.93, 'y1': 143.46, 'y2': 374.72},
   'confidence': '58.2',
   'label': 'person'},
  {'bounding_box': {'x1': 215.56, 'x2': 346.96, 'y1': 212.26, 'y2': 419.51},
   'confidence': '26.95',
   'label': 'bicycle'},
  {'bounding_box': {'x1': 422.38, 'x2': 454.59, 'y1': 171.35, 'y2': 210.02},
   'confidence': '21.09',
   'label': 'person'},
  {'bounding_box': {'x1': 359.8, 'x2': 389.19, 'y1': 161.48, 'y2': 204.62},
   'confidence': '21.09',
   'label': 'person'},
  {'bounding_box': {'x1': 305.36, 'x2': 346.21, 'y1': 147.99, 'y2': 211.31},
   'confidence': '21.09',
   'label': 'person'},
  {'bounding_box': {'x1': 5.55, 'x2': 21.94, 'y1': 9.36, 'y2': 45.3},
   'confidence': '21.09',
   'label': 'traffic light'},
  {'bounding_box': {'x1': 239.76, 'x2': 324.57, 'y1': 260.07, 'y2': 406.78},
   'confidence': '16.02',
   'label': 'bicycle'},
  {'bounding_box': {'x1': 299.75, 'x2': 358.61, 'y1': 154.64, 'y2': 298.7},
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
