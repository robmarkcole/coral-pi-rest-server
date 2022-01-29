# Start the server:
# 	python3 coral-app.py
# Submit a request via cURL:
# 	curl -X POST -F image=@face.jpg 'http://localhost:5000/v1/vision/detection'

from edgetpu.detection.engine import DetectionEngine
import argparse
from PIL import Image
import flask
import logging
import io
import time

app = flask.Flask(__name__)

LOGFORMAT = "%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s"
logging.basicConfig(filename="coral.log", level=logging.DEBUG, format=LOGFORMAT)

engine = None
labels = None

DEFAULT_MODELS_DIRECTORY = "~/Documents/GitHub/edgetpu/test_data/"
DEFAULT_MODEL = "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
DEFAULT_LABELS = "coco_labels.txt"

ROOT_URL = "/v1/vision/detection"


# Function to read labels from text files.
def ReadLabelFile(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        ret = {}
        counter = 0
        for label in lines:
            pair = label.strip().split(maxsplit=1)
            if (len(pair) > 1):
                try:
                  counter = int(pair[0])
                  label = pair[1]
                except ValueError:
                  pass
            ret[counter] = label.strip()
            counter += 1
    return ret

# Function to return time in milliseconds
def current_milli_time():
    return round(time.time() * 1000, 4)

@app.route("/")
def info():
    info_str = "Flask app exposing tensorflow lite model {}".format(MODEL)
    return info_str


@app.route(ROOT_URL, methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image_file = flask.request.files["image"]
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))

            # initiate execution timer
            start = current_milli_time()

            # Run inference.
            predictions = engine.detect_with_image(
                image,
                threshold=0.05,
                keep_aspect_ratio=True,
                relative_coord=False,
                top_k=10,
            )

            # calculate detection time
            duration = current_milli_time() - start
            data["duration"] = duration
            app.logger.debug('Detection time %s ms', duration)

            if predictions:
                data["success"] = True
                preds = []
                for prediction in predictions:
                    try:
                        preds.append(
                            {
                                "confidence": float(prediction.score),
                                "label": labels[prediction.label_id],
                                "y_min": int(prediction.bounding_box[0, 1]),
                                "x_min": int(prediction.bounding_box[0, 0]),
                                "y_max": int(prediction.bounding_box[1, 1]),
                                "x_max": int(prediction.bounding_box[1, 0]),
                            }
                        )
                    except KeyError:
                        app.logger.error("Label %s doesn't exist", prediction.label_id)
                data["predictions"] = preds

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
        "--model", default=DEFAULT_MODEL, help="model file",
    )
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="labels file of model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    global MODEL
    MODEL = args.model
    model_file = args.models_directory + args.model
    labels_file = args.models_directory + args.labels

    engine = DetectionEngine(model_file)
    print("\n Loaded engine with model : {}".format(model_file))

    labels = ReadLabelFile(labels_file)
    app.run(host="0.0.0.0", port=args.port)

