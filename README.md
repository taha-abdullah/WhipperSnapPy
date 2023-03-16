# WhipperSnapper

WhipperSnapper is a small Python OpenGL program to render FreeSurfer and 
FastSurfer surface models and color overlays and generate screen shots.

## Contents:

- Capture 4x4 surface plots (front & back, left and right)
- OpenGL window for interactive visualization

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

## Installation:

The `whippersnapper` package can be installed from this repository using:
```
python3 -m pip install .
```

## Usage:

### Docker:

The whippersnapper program can be run within a docker container to capture
a snapshot by building the provided Docker image and running a container as
follows:
```
docker build --rm=true -t whippersnapper -f ./Dockerfile .
```
```
docker run --rm --init --name my_whippersnapper -v $SURF_SUBJECT_DIR:/surf_subject_dir \
                                                -v $OVERLAY_DIR:/overlay_dir \
                                                -v $OUTPUT_DIR:/output_dir \
                                                --user $(id -u):$(id -g) whippersnapper:latest \
                                                --lh_overlay /overlay_dir/$LH_OVERLAY_FILE \
                                                --rh_overlay /overlay_dir/$RH_OVERLAY_FILE \
                                                --sdir /surf_subject_dir \
                                                --output_path /output_dir/whippersnapper_image.png
```

In this example: `$SURF_SUBJECT_DIR` contains the surface files, `$OVERLAY_DIR` contains the overlays to be loaded on to the surfaces, `$OUTPUT_DIR` is the local output directory in which the snapshot will be saved, and `${LH/RH}_OVERLAY_FILE` point to the specific overlay files to load.

**Note:** The `--init` flag is needed for the `xvfb-run` tool to be used correctly.

## Links:

We also invite you to check out our lab webpage at https://deep-mi.org
