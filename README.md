# coral-pi-rest-server
Expose deep learning models on a Coral usb accelerator via a rest API on a raspberry pi.

## Official code
* The edgetpu code doesn't appear to live on Github, but is downloaded from a google machine. I have placed some of the python files from that library in this repo for reference. In particular `edgetpu/classification/engine.py` contains the `ClassificationEngine` class used to perform inferencing. The good news is that the only python dependencies (apart from some device specific code) are `numpy` and `PIL`.

## Rest API
* As a guide see -> https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
* We will be swapping out the keras:
```python
preds = model.predict(image)
results = imagenet_utils.decode_predictions(preds)
```
with:
```python
# Prepare labels.
labels = ReadLabelFile(args.label)
# Initialize engine.
engine = ClassificationEngine(args.model)
# Run inference.
img = Image.open(args.image)
for result in engine.ClassifyWithImage(img, top_k=3):
  print ('---------------------------')
  print (labels[result[0]])
  print ('Score : ', result[1])
```

* Another possibly useful reference -> https://github.com/tomimick/restpie3

## Models
* The official pre-compiled models are at -> https://coral.withgoogle.com/models/
* It is [also possible to train your own models](https://coral.withgoogle.com/tutorials/edgetpu-models-intro/) -> try using Google Colaboratory as the free environment for training

## Docker
* Want to wrap this project up in a Docker container for easy deployment
* https://www.raspberrypi.org/blog/docker-comes-to-raspberry-pi/

## IDE 
* Tryout cloud9 -> https://www.siaris.net/post/cloud9/

## References
* [Using the official pi camera with Coral](https://github.com/nickoala/edgetpu-on-pi)
* https://github.com/snmsung716/Following-camera-with-Coral-Accelerator-on-Raspberry-Pi
