FROM ubuntu:22.04

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3pip xvfb && \
  apt clean && \
  rm -rf /var/libs/apt/lists/* /tmp/* /var/tmp/* \
  pip3 install pyopengl glfw pillow numpy pyrr

COPY ./whippersnapper /whippersnapper

WORKDIR /whippersnapper
ENTRYPOINT ["xvfb-run","python3","whippersnapper.py"]
CMD ["--help"]

