# FROM ubuntu:22.04
FROM python:3.9

ENV DEBIAN_FRONTEND=noninteractive
# ENV TZ=UTC

# WORKDIR /workspace

# Install basic apt packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    cmake \
    git \
    nano \
    vim \
    wget \
    curl \
    gdb

RUN pip install packaging
RUN pip install mmengine-lite==0.10.4 tqdm cdsapi>=0.7.4
RUN pip install netCDF4 xarray cartopy h5py
RUN pip install matplotlib

ENTRYPOINT /bin/bash
