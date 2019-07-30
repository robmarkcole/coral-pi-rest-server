# Start the server:
# 	python3 coral-app.py
# Submit a request via cURL:
# 	curl -X POST -F image=@face.jpg 'http://localhost:5000/v1/vision/detection'

from edgetpu.detection.engine import DetectionEngine
from PIL import Image
import flask
import io

app = flask.Flask(__name__)
engine = None
labels = None

ROOT_URL = "/v1/vision/detection"
PORT = 5000

MODELS_DIR = "/home/robin/edgetpu/all_models/"
MODEL = "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
LABELS = "coco_labels.txt"

MODEL_FILE = MODELS_DIR + MODEL
LABEL_FILE = MODELS_DIR + LABELS


# Function to read labels from text files.
def ReadLabelFile(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        ret = {}
        for line in lines:
            pair = line.strip().split(maxsplit=1)
            ret[int(pair[0])] = pair[1].strip()
    return ret


@app.route("/")
def info():
    info_str = f"Flask app exposing tensorflow model: {MODEL}\n"
    return info_str


@app.route(ROOT_URL, methods=["POST"])
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
                    preds.append(
                        {
                            "confidence": str(100 * prediction.score),  # A percentage.
                            "label": labels[prediction.label_id],
                            "x_min": int(prediction.bounding_box[0, 0]),
                            "x_max": int(prediction.bounding_box[1, 0]),
                            "y_min": int(prediction.bounding_box[0, 1]),
                            "y_max": int(prediction.bounding_box[1, 1]),
                        }
                    )
                data["predictions"] = preds

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    engine = DetectionEngine(MODEL_FILE)
    print("\n Loaded engine with model : {}".format(MODEL))

    labels = ReadLabelFile(LABEL_FILE)
    app.run(host="0.0.0.0", port=PORT)
