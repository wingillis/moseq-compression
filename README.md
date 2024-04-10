# moseq-compression
Helper functions to compress depth and infrared video acquired using the Datta Lab MoSeq platform.

## How to install

Make sure you have a version of `ffmpeg` installed that supports the `libx265` codec.
Recent versions of `ffmpeg` should support this codec, but you can check by running `ffmpeg -hide_banner -codecs | grep libx265` in the terminal and looking for any output containing `libx265`.

Activate the `conda` environment you choose to install this script in.

Then pip install it:
`pip install git+https://github.com/wingillis/moseq-compression.git@v0.1.2`

Change `v0.1.1` to the version you want to install. The current version as of 2024-04-10 is `v0.1.2`.

To upgrade, run the following command with the new version number. For example:
`pip install -U git+https://github.com/wingillis/moseq-compression.git@v0.2`

## How to use

After installation, you can use the `moseq-compress` command in the terminal to select a MoSeq recording folder and compress the depth and infrared (if present) videos within.

Folders **must** be extracted using [moseq2-extract](https://github.com/dattalab/moseq2-extract) before using this tool.

There are two main ways to use this tool:

1. **Interactive mode**: Run `moseq-compress` in the terminal and use the user interface to select a recording folder. The script will look inside to find the depth and infrared videos and compress them. This mode is useful for

2. **Batch mode**: Run `moseq-compress --path /path/to/recording/folder` in the terminal to skip the user interface section and compress the depth and infrared videos within the specified folder. This mode is useful for automated scripts or pipelines.


## Behavior

The script will look for depth and infrared videos in the specified folder.
If it does not find a depth video with the name `depth.avi`, it will throw an error and stop.
However, if it does not find an infrared video with the name `ir.avi`, it will continue, but print a warning that the infrared video was not found.
Then, it will compress them using the `ffmpeg` library and save the compressed videos in the same folder.
The original videos will **not** be deleted.

It will create two versions of the depth videos.
The first, `depth_cropped.avi`, removes the extra surrounding pixels that are not part of the MoSeq arena.
It is saved with a lossless compression, so has the original raw depth values.
The second, `depth_cropped.mp4`, is a lossy compressed version of the first video.
This version is much smaller, so is useful for data storage constraints, but is in the alpha stage at the moment and is not supported by `moseq2-extract`.
Therefore, if you can keep the `depth_cropped.avi` file, it is recommended to do so.

If the infrared video is found, it will be lossy compressed and saved as `ir.mp4`.
Since the infrared video is not used by `moseq2-extract`, it is safe to delete the original `ir.avi` file after compression.
The `ir.mp4` file has enough fidelity to be used effectively for keypoint tracking.
