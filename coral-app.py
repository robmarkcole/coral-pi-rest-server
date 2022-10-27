# Start the server:
# 	python3 coral-app.py
# Submit a request via cURL:
# 	curl -X POST -F image=@images/test-image3.jpg 'http://localhost:5000/v1/vision/detection'

import argparse
import io
import os
import logging

import flask
from PIL import Image
from pycoral.adapters import detect, common
from pycoral.utils import dataset, edgetpu

app = flask.Flask(__name__)

LOGFORMAT = "%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s"
logging.basicConfig(filename="coral.log", level=logging.DEBUG, format=LOGFORMAT)
stderrLogger=logging.StreamHandler()
stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
logging.getLogger().addHandler(stderrLogger)


DEFAULT_MODELS_DIRECTORY = "models"
DEFAULT_MODEL = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
DEFAULT_LABELS = "coco_labels.txt"

ROOT_URL = "/v1/vision/detection"

@app.route("/")
def info():
    info_str = "Flask app exposing tensorflow lite model {}".format(MODEL)
    return info_str


@app.route(ROOT_URL, methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.form.get('min_confidence'):
            threshold=float(flask.request.form['min_confidence'])
        else:
            threshold=float(0.4)
        if flask.request.files.get("image"):
            image_file = flask.request.files["image"]
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))

            size = common.input_size(interpreter)
            image = image.convert("RGB").resize(size, Image.ANTIALIAS)

            _, scale = common.set_resized_input(
                interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))


            # Run an inference
            common.set_input(interpreter, image)
            start = time.perf_counter()
            interpreter.invoke()
            inference_time = time.perf_counter() - start
            objs = detect.get_objects(interpreter, threshold, scale)
            print('%.2f ms' % (inference_time * 1000))
            if objs:
                data["success"] = True
                preds = []

                for obj in objs:
                    if float(obj.score) >= float(threshold):
                        preds.append(
                            {
                                "confidence": float(obj.score),
                                "label": labels[obj.id],
                                "y_min": int(obj.bbox.ymin),
                                "x_min": int(obj.bbox.xmin),
                                "y_max": int(obj.bbox.ymax),
                                "x_max": int(obj.bbox.xmax),
                            }
                        )
                    print(labels.get(obj.id, obj.id))
                    print('  id:    ', obj.id)
                    print('  score: ', obj.score)
                    print('  bbox:  ', obj.bbox)
                data["predictions"] = preds
            else:
                print('No objects detected')

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing coral USB stick")
    parser.add_argument(
        "--models_directory",
        default=DEFAULT_MODELS_DIRECTORY,
        help="the directory containing the model & labels files",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="model file",
    )
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="labels file of model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    global MODEL
    MODEL = args.model
    model_file = os.path.join(args.models_directory, args.model)
    assert os.path.isfile(model_file)

    labels_file = os.path.join(args.models_directory, args.labels)
    assert os.path.isfile(labels_file)

    global labels
    labels = dataset.read_label_file(labels_file)

    global interpreter
    interpreter = edgetpu.make_interpreter(model_file)
    interpreter.allocate_tensors()
    print("\n Initialised interpreter with model : {}".format(model_file))

    app.run(host="0.0.0.0", debug=False, port=args.port)