FROM debian:11-slim

ARG SPEED=std
ARG MODEL=tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite
ARG LABELS=coco_labels.txt

EXPOSE 5000

WORKDIR /app

RUN apt-get update && apt-get install -y curl gnupg jq

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN echo "deb https://packages.cloud.google.com/apt coral-cloud-stable main" | tee /etc/apt/sources.list.d/coral-cloud.list

RUN apt-get update && apt-get install -y python3 python3-pip python3-pycoral libedgetpu1-$SPEED

ADD requirements.txt /app/
RUN pip install -r requirements.txt

ADD . /app

ENV MODEL_FILE=$MODEL
ENV LABELS_FILE=$LABELS

CMD /usr/bin/python3 /app/coral-app.py --model $MODEL_FILE --labels $LABELS_FILE --models_directory /app/models/

HEALTHCHECK --interval=60s --timeout=5s CMD ["bash", "-c", "curl -fs http://localhost:5000/v1/vision/detection?healthcheck -F 'image=@/app/car.jpg' | [[ $(jq -r .success) == 'true' ]] || exit 1"]
