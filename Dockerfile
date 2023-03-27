FROM ubuntu:20.04

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip xvfb && \
  apt clean && \
  rm -rf /var/libs/apt/lists/* /tmp/* /var/tmp/*

# Install python packages
RUN pip3 install pyopengl glfw pillow numpy pyrr

COPY . /WhipperSnapPy
RUN pip3 install /WhipperSnapPy

ENTRYPOINT ["xvfb-run","whippersnap"]
CMD ["--help"]
