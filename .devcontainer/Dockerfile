FROM mcr.microsoft.com/devcontainers/python 

LABEL author="Sebastian Brötz"

RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-full \
    dvipng \
    cm-super \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
EXPOSE 8988
WORKDIR /workspace