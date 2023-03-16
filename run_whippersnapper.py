#!/usr/bin/python3

"""Executes the whippersnapper program in an interactive or non-interactive mode.

The non-interactive mode (the default) creates an image that contains four
views of the surface, a color bar, and a configurable caption.
The interactive mode (--interactive) opens a simple GUI with a controllable
view of one of the hemispheres.

Usage:
    $ python3 run_whippersnapper.py -lh $LH_OVERLAY_FILE -rh $RH_OVERLAY_FILE \
                                    -sd $SURF_SUBJECT_DIR -o $OUTPUT_PATH
(See help for full list of arguments.)

@Author     : Martin Reuter
@Created    : 16.03.2022

"""

import argparse

from whippersnapper.core import show_window, snap4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lh', '--lh_overlay', type=str, required=True,
                        help='Absolute path to the lh overlay file.')
    parser.add_argument('-rh', '--rh_overlay', type=str, required=True,
                        help='Absolute path to the rh overlay file.')
    parser.add_argument('-sd', '--sdir', type=str, required=True,
                        help='Absolute path to the subject directory from which surfaces will be loaded. '
                             'This is assumed to contain the surface files in a surf/ sub-directory.')
    parser.add_argument('-s', '--surf_name', type=str, default=None,
                        help='Name of the surface file to load.')
    parser.add_argument('-o', '--output_path', type=str, default='/tmp/whippersnapper_snap.png',
                        help='Absolute path to the output file (snapshot image), '
                             'if not running interactive mode.')
    parser.add_argument('-c', '--caption', type=str, default='Super cool WhipperSnapper 2.0',
                        help='Caption to place on the figure')
    parser.add_argument('--fmax', type=float, default=4.0)
    parser.add_argument('--fthresh', type=float, default=2.0)
    parser.add_argument('-i', '--interactive', dest='interactive', action='store_true',
                        help='Start an interactive session.')
    args = parser.parse_args()

    if args.interactive:
        show_window('lh', args.lh_overlay, sdir=args.sdir, surfname=args.surf_name)
    else:
        snap4(args.lh_overlay, args.rh_overlay, sdir=args.sdir, caption=args.caption, surfname=args.surf_name,
              fthresh=args.fthresh, fmax=args.fmax, invert=False, colorbar=True, outpath=args.output_path)


# headless docker test using xvfb:
# Note, xvfb is a display server implemening the X11 protocol, performing all graphics on memory
# glfw needs a windows to render even if that is invisible, so above code
# will not work via ssh or on a headless server. xvfb can solve this by wrapping:
#docker run --name headless_test -ti -v$(pwd):/test ubuntu /bin/bash
#apt update && apt install -y python3 python3-pip xvfb
#pip3 install pyopengl glfw pillow numpy pyrr
#xvfb-run python3 test4.py

# instead of the above one could really do headless off screen rendering via EGL (preferred)
# or OSMesa. The latter looks doable. EGL looks tricky. 
# EGL is part of any modern NVIDIA driver
# OSMesa needs to be installed, but should work almost everywhere

# using EGL maybe like this:
# https://github.com/eduble/gl
# or via these bindings:
# https://github.com/perey/pegl

# or OSMesa
# https://github.com/AntonOvsyannikov/DockerGL
