FROM kaldiasr/kaldi:gpu-ubuntu20.04-cuda11

# Sistem paketleri
RUN apt-get update && apt-get install -y \
    sox \
    libsox-fmt-all \
    python3-pip \
    espeak-ng \
    cmake \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Python paketleri
RUN pip install phonemizer

RUN ln -s /usr/bin/python3 /usr/bin/python

# KenLM kur
RUN cd /opt/kaldi/tools && \
    git clone https://github.com/kpu/kenlm.git && \
    cd kenlm && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# KenLM PATH
ENV PATH="/opt/kaldi/tools/kenlm/build/bin:${PATH}"

WORKDIR /opt/kaldi/egs/commonvoice/s5

ENTRYPOINT ["bash", "/data/proje/docker_baslat.sh"]
