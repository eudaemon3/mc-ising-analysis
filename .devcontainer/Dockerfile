FROM mcr.microsoft.com/devcontainers/python 

LABEL author="Sebastian Brötz"

RUN apt-get update && apt-get install -y --no-install-recommends \
    openmpi-bin \
    libopenmpi-dev \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-fonts-recommended \
    texlive-latex-extra \
    dvipng \
    cm-super \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace