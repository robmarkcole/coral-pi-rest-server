#
#  Louis Mamakos <louie@transsys.com>
#
#  Build a container to run the edgetpu flask daemon
#
#    docker build -t coral .
#
#  Run it something like:
#
#  docker run --restart=always --detach --name coral \
#          -p 5000:5000 --device /dev/bus/usb:/dev/bus/usb   coral:latest
#
#  It's necessary to pass in the /dev/bus/usb device to communicate with the USB stick.
#
#  You can use alternative models by putting them into a directory
#  that's mounted in the container, and then starting the container,
#  passing in environment variables MODEL and LABELS referring to
#  the files.

FROM python:3.6

WORKDIR /tmp

# downloading library file for edgetpu and install it
RUN wget --trust-server-names -O edgetpu_api.tar.gz  https://dl.google.com/coral/edgetpu_api/edgetpu_api_latest.tar.gz && \
    tar xzfz edgetpu_api.tar.gz && rm edgetpu_api.tar.gz && \
    cd edgetpu_api && \
    sed -i.orig  \
    	-e 's/^read USE_MAX_FREQ/USE_MAX_FREQ=y/' \
	-e 's/apt-get install/apt-get install --no-install-recommends/'  \
	-e '/^UDEV_RULE_PATH=/,/udevadm trigger/d'  \
      install.sh && \
    apt-get update && apt-get install sudo && \
    bash ./install.sh


WORKDIR /usr/src/app

# fetch the models.  maybe figure a way to conditionalize this?
# create models subdirectory for volume mount of custom models
RUN  mkdir /models && \
     chdir /models && \
     curl -q -O  https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite  && \
     curl -q -O  https://dl.google.com/coral/canned_models/coco_labels.txt
     

COPY requirements.txt ./
RUN  pip install --no-cache-dir -r requirements.txt 

COPY coral-app.py     ./

ENV MODEL=/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
    LABELS=/models/coco_labels.txt

EXPOSE 5000

CMD  exec python coral-app.py --model  "${MODEL}" --labels "${LABELS}"
