# USAGE
# Start the server:
# 	python3 coral-app.py
# Submit a request via cURL:
# 	curl -X POST -F image=@face.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
# 	python simple_request.py

# import the necessary packages
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
engine = None
labels = None

DECIMALS = 2  # The number of decimal places data is returned to

MODEL = "/home/robin/edgetpu/all_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
LABEL_FILE = "/home/robin/edgetpu/all_models/coco_labels.txt"


# Function to read labels from text files.
def ReadLabelFile(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        ret = {}
        for line in lines:
            pair = line.strip().split(maxsplit=1)
            ret[int(pair[0])] = pair[1].strip()
    return ret


def load_model(model_file=None, label_file=None):
    """
    Load model and labels.
    """
    global engine, labels

    if not model_file:
        model_file = MODEL
    if not label_file:
        label_file = LABEL_FILE

    engine = DetectionEngine(model_file)
    print("\n Loaded engine with model : {}".format(model_file))

    labels = ReadLabelFile(label_file)
    print("\n Loaded labels from file : {}".format(label_file))


@app.route("/")
def info():
    info_str = "Flask app exposing tensorflow models via Google Coral.\n"
    return info_str


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image_file = flask.request.files["image"]
            print(image_file)
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))  # PIL img object.

            # Run inference.
            predictions = engine.DetectWithImage(
                image,
                threshold=0.05,
                keep_aspect_ratio=True,
                relative_coord=False,  # True = relative coordinates 0-1 of original image.
                top_k=10,
            )

            if predictions:
                data["success"] = True
                preds = []
                for prediction in predictions:
                    bounding_box = {
                        "x1": round(prediction.bounding_box[0, 0], DECIMALS),
                        "x2": round(prediction.bounding_box[1, 0], DECIMALS),
                        "y1": round(prediction.bounding_box[0, 1], DECIMALS),
                        "y2": round(prediction.bounding_box[1, 1], DECIMALS),
                    }
                    preds.append(
                        {
                            "confidence": str(
                                round(100 * prediction.score, DECIMALS)
                            ),  # A percentage.
                            "label": labels[prediction.label_id],
                            "bounding_box": bounding_box,
                        }
                    )
                data["predictions"] = preds

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Google Coral edgetpu flask daemon")
    parser.add_argument("--quiet", "-q", action='store_true',
                        help="log only warnings, errors")
    parser.add_argument("--port", '-p', default=5000,
                        type=int, choices=range(0, 65536),
                        help="port number")
    parser.add_argument("--model",  default=None, help="model file")
    parser.add_argument("--labels", default=None, help="labels file for model")
    args = parser.parse_args()

    load_model(args.model, args.labels)
    app.run(host="0.0.0.0", port=args.port)
