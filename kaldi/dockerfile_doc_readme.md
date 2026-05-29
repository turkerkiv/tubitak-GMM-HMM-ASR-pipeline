# Dockerfile walkthrough

This document explains each step in Dockerfile in the order it runs. It is written for readers who are new to Docker and Kaldi, so it also explains what each instruction does and why it is needed.

## Quick glossary (very short)
- `image`: a packaged filesystem and settings used to create containers.
- `container`: a running instance of an image.
- `layer`: each Dockerfile instruction creates a new filesystem layer.
- `ENTRYPOINT`: the default command that runs when the container starts.

## 1) Base image
- `FROM kaldiasr/kaldi`
  - What it does: starts from the official Kaldi image, which already contains Kaldi tools and dependencies.
  - Why it matters: you do not need to build Kaldi from source yourself.

## 2) Install system packages
```
RUN apt-get update && apt-get install -y \
    sox \
    libsox-fmt-all \
    python3-pip \
    espeak-ng \
    cmake \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*
```
- What it does: updates package lists, installs needed system tools, then cleans apt cache.
- Why these packages:
  - `sox`, `libsox-fmt-all`: audio utilities and codecs for reading and converting audio.
  - `python3-pip`: install Python packages.
  - `espeak-ng`: text-to-phoneme tool used by some phonemization pipelines.
  - `cmake`, `libboost-all-dev`: required to build KenLM from source.
- Why cleanup matters: removes cached lists to keep the image smaller.

## 3) Install Python package
- `RUN pip install phonemizer --break-system-packages`
  - What it does: installs the `phonemizer` Python library.
  - Why it matters: it can convert text into phonemes for lexicon creation.
  - Note: `--break-system-packages` allows pip to install into a system Python in Debian/Ubuntu images.

## 4) Build KenLM
```
RUN cd /opt/kaldi/tools && \
    git clone https://github.com/kpu/kenlm.git && \
    cd kenlm && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4
```
- What it does: clones KenLM and builds it from source.
- Why it matters: the language model tools (`lmplz`, `build_binary`) are used later in the ASR pipeline.
- `make -j4` uses 4 CPU threads to compile faster.

## 5) Add KenLM to PATH
- `ENV PATH="/opt/kaldi/tools/kenlm/build/bin:${PATH}"`
  - What it does: makes KenLM binaries available in the shell by default.
  - Why it matters: scripts can call `lmplz` without using full paths.

## 6) Set working directory
- `WORKDIR /opt/kaldi/egs/commonvoice/s5`
  - What it does: sets the default directory for any following commands.
  - Why it matters: the Kaldi recipe expects to run from this folder.

## 7) Default container command
- `ENTRYPOINT ["bash", "/data/proje/docker_baslat.sh"]`
  - What it does: when the container starts, it runs the Bash script that launches the full training/decoding pipeline.
  - Why it matters: a single `docker run` will automatically start the Kaldi workflow.
