import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image

model_file = os.path.join(
    "models", "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
)
print(os.path.isfile(model_file))

label_file = os.path.join("models", "coco_labels.txt")
print(os.path.isfile(label_file))

image_file = os.path.join("images", "parrot.jpg")
print(os.path.isfile(image_file))

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

# Resize the image
size = common.input_size(interpreter)
image = Image.open(image_file).convert("RGB").resize(size, Image.ANTIALIAS)

# Run an inference
common.set_input(interpreter, image)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)

# Print the result
labels = dataset.read_label_file(label_file)
for c in classes:
    print("%s: %.5f" % (labels.get(c.id, c.id), c.score))
