# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@face.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
# 	python simple_request.py

# import the necessary packages
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

MODEL = "/home/robin/Downloads/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite"


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
            engine = DetectionEngine(MODEL)
            predictions = engine.DetectWithImage(
                image,
                threshold=0.05,
                keep_aspect_ratio=True,
                relative_coord=False,
                top_k=10,
            )

 
            if predictions:
                for i, prediction in enumerate(predictions):
                    data[str(i)] = str(prediction.score)
                data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
