# WhipperSnapper

WhipperSnapper is a small Python OpenGL program to render FreeSurfer and 
FastSurfer surface models and color overlays and generate screen shots.

## Contents:

- Capture 
- Plot: functions for interactive visualization (wrapping plotly)

Note, that currently no off-screen rendering is supported. Even in snap 
mode an invisible window will be created to render the openGL output
and capture the contents to an image. In order to run this on a headless
server, inside Docker, or via ssh we recommend to install xvfb and run

```
apt update && apt install -y python3 python3-pip xvfb
pip3 install pyopengl glfw pillow numpy pyrr
xvfb-run python3 whipersnapper.py ...
```

## ToDo:

- Add unit tests and automated testing 
- Move to true off-screen rendering e.g. via EGL (preferred) or OSMesa

## Usage:



## Links:

We also invite you to check out our lab webpage at https://deep-mi.org
