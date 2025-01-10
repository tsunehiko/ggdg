FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    wget bzip2 ca-certificates curl git \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion nano vim \
    libosmesa6-dev libgl1-mesa-glx libglfw3 build-essential \
    python3.10-dev python3-pip python3-opencv \
    openjdk-11-jdk

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN pip install --upgrade pip

WORKDIR /ggdg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

COPY . /ggdg
RUN pip install -e .
RUN chmod +x /ggdg/scripts
RUN chmod +x /ggdg/sft/train.sh

# Install Ludii
RUN mkdir -p /ggdg/ludii_java/libs /ggdg/ludii_java/src /ggdg/ludii_java/out /ggdg/ludii_java/jars
RUN wget -P /ggdg/ludii_java/libs https://ludii.games/downloads/Ludii-1.3.13.jar
COPY ludii_java/src/ /ggdg/ludii_java/src/
COPY ludii_java/manifest_eval.mf     /ggdg/ludii_java/
COPY ludii_java/manifest_concept.mf  /ggdg/ludii_java/
COPY ludii_java/manifest_ma.mf       /ggdg/ludii_java/
COPY ludii_java/manifest_expand.mf   /ggdg/ludii_java/
RUN javac \
    -encoding UTF-8 \
    -cp /ggdg/ludii_java/libs/Ludii-1.3.13.jar:/ggdg/ludii_java/src \
    -d /ggdg/ludii_java/out \
    /ggdg/ludii_java/src/EvalLudiiGame.java \
    /ggdg/ludii_java/src/ComputeConcept.java \
    /ggdg/ludii_java/src/ComputeMultiAgents.java \
    /ggdg/ludii_java/src/ExtractExpand.java
RUN jar cfm /ggdg/ludii_java/jars/EvalLudiiGame.jar /ggdg/ludii_java/manifest_eval.mf -C /ggdg/ludii_java/out .
RUN jar cfm /ggdg/ludii_java/jars/ComputeConcept.jar /ggdg/ludii_java/manifest_concept.mf -C /ggdg/ludii_java/out .
RUN jar cfm /ggdg/ludii_java/jars/ComputeMultiAgents.jar /ggdg/ludii_java/manifest_ma.mf -C /ggdg/ludii_java/out .
RUN jar cfm /ggdg/ludii_java/jars/ExtractExpand.jar /ggdg/ludii_java/manifest_expand.mf -C /ggdg/ludii_java/out .

ENV HF_HOME=/ggdg/.cache/huggingface
