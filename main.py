import av
import cv2
import h5py
import click
import subprocess
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from contextlib import contextmanager
from tkinter.filedialog import askdirectory

# create context for writing movie
@contextmanager
def write_movie(path: Path, width: int, height: int, min_height: int, max_height: int, scale: float, fps: float) -> callable:
    '''
    Creates a context manager for writing a movie to a file.

    Returns:
        write_frame: a function that writes a frame to the movie.
    '''
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx265', rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'gray12le'
        stream.options = {'crf': '20', 'preset': 'medium', 'vtag': 'hvc1'}
        container.metadata["description"] = f"scale={scale},min_height={min_height},max_height={max_height}"

        def write_frame(frame: np.ndarray):
            new_frame = np.where(frame > max_height, 0, frame.astype('int32') - min_height)
            new_frame = np.clip(new_frame, 0, None) * scale
            # clip to 16-bit range
            new_frame = np.clip(new_frame, 0, 2**16 - 1).astype('uint16')

            av_frame = av.VideoFrame.from_ndarray(new_frame, format='gray16le')
            for packet in stream.encode(av_frame):
                container.mux(packet)

        yield write_frame


def get_bbox(mask):
    y, x = np.where(mask > 0)
    return np.array([[y.min(), x.min()], [y.max(), x.max()]])
    

def get_roi_bounds(h5_path):
    with h5py.File(h5_path, 'r') as f:
        roi = f['/metadata/extraction/roi'][()]
    roi_dilated = cv2.dilate(roi.astype('uint8'), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=5)
    bbox = get_bbox(roi_dilated)
    return {
        'w': bbox[0, 1],
        'h': bbox[0, 0],
        'y_size': bbox[1, 0] - bbox[0, 0],
        'x_size': bbox[1, 1] - bbox[0, 1],
    }


def get_scaling_parameters(path, h5_path):
    with h5py.File(h5_path, 'r') as f:
        roi = f['/metadata/extraction/roi'][()]

    frames = []
    with av.open(str(path), 'r') as container:
        for i, frame in enumerate(container.decode(video=0)):
            frame = frame.to_ndarray(format='gray16').astype('float32')
            frames.append(frame)
            if i > 15:
                break
    frames = np.array(frames)
    
    frames = frames * roi[None]
    frames[frames == 0] = np.nan

    buffer = np.array([-100, 25])
    min_height, max_height = np.nanpercentile(frames.flatten(), [0.1, 99.9]) + buffer

    return max(min_height, 0), max_height


def infer_scaling_parameters(path):
    frames = []
    with av.open(str(path), 'r') as container:
        container.streams.video[0].thread_type = 'AUTO'
        for i, frame in enumerate(container.decode(video=0)):
            frame = frame.to_ndarray(format='gray16').astype('float32')
            frames.append(frame)
            if i > 1_000:
                break
    frames = np.array(frames)
    frames[frames == 0] = np.nan
    center = np.array(frames.shape[1:]) // 2

    buffer = np.array([-150, 30])

    slice_size = 30
    row_slice = slice(center[0]-slice_size, center[0]+slice_size)
    col_slice = slice(center[1]-slice_size, center[1]+slice_size)
    floor = np.nanmedian(frames[:, row_slice, col_slice].flatten())
    min_depth = floor + buffer[0]
    max_depth = floor + buffer[1]

    return max(min_depth, 0), max_depth, frames.shape[1:]


@click.command()
@click.option('--path', type=click.Path(exists=True, file_okay=False), help='Folder that contains videos.')
@click.option('--scale', type=int, default=250, help='Scaling factor for depth video compression.')
@click.option('--crop', is_flag=True, help='Crop the depth video to the ROI.')
def main(path, scale, crop):
    if path is None:
        path = Path(askdirectory(mustexist=True))
    else:
        path = Path(path)
    # set up assertions
    if not (path / "depth.avi").exists():
        raise FileNotFoundError("depth.avi not found in the folder. Folder must be for a recording session.")

    if not (path / "proc").exists() and crop:
        raise FileNotFoundError("proc folder not found in the folder. Recording session must be extracted first.")

    has_ir = (path / "ir.avi").exists()
    if not has_ir:
        print("INFO: ir.avi not found in the folder. Will not run compression routine for ir.avi.")
        print()

    compressed_uncropped_path = path / "depth.mp4"
    min_height, max_height, shape = infer_scaling_parameters(path / "depth.avi")
    print("Compression parameters inferred from the first 200 frames:")
    print("    Min depth (mm)", min_height, "Max depth (mm)", max_height, "Frame shape", shape)
    print()

    print("Creating a viewable version of the depth video...")
    print()

    with write_movie(compressed_uncropped_path, shape[1], shape[0], min_height, max_height, scale, 30) as write_frame:
        with av.open(str(path / "depth.avi"), 'r') as container:
            container.streams.video[0].thread_type = 'AUTO'
            for frame in tqdm(container.decode(video=0), desc="Compressing depth video", unit="frames"):
                frame = frame.to_ndarray(format='gray16').astype('uint16')
                write_frame(frame)

    if crop:
        print("Cropping depth video to ROI and compressing...")
        # first, crop the depth video
        bounds = get_roi_bounds(path / "proc" / "results_00.h5")
        crop_filter = f"crop={bounds['x_size']}:{bounds['y_size']}:{bounds['w']}:{bounds['h']}"

        cropped_path = path / "depth_cropped.avi"
        subprocess.run([
            'ffmpeg', '-y', '-i', str(path / "depth.avi"),
            '-vf', crop_filter,
            '-vcodec', 'ffv1', '-level', '3', '-slicecrc', '1',
            '-coder', '2', '-slices', '24',
            str(cropped_path),
        ])

        # next, compress the depth video
        compressed_path = path / "depth_cropped.mp4"
        min_height, max_height = get_scaling_parameters(path / "depth.avi", path / "proc" / "results_00.h5")
        
        max_scale = int((2**16 / (max_height - min_height)) - 1)
        if max_scale < scale:
            print("WARNING: Scaling factor is too high for the cropped depth video.")
            print("    Current scaling factor:", scale, "Max scaling factor:", max_scale)
            print("    Setting scaling factor to:", max_scale)
            scale = max_scale

        with write_movie(compressed_path, bounds['x_size'], bounds['y_size'], min_height, max_height, scale, 30) as write_frame:
            with av.open(str(cropped_path), 'r') as container:
                container.streams.video[0].thread_type = 'AUTO'
                for frame in tqdm(container.decode(video=0), desc="Compressing cropped depth video", unit="frames"):
                    frame = frame.to_ndarray(format='gray16').astype('uint16')
                    write_frame(frame)
            
    # finally, compress the ir video
    if has_ir:
        print("Compressing ir video...")
        ir_path = path / "ir.avi"
        compressed_ir_path = path / "ir.mp4"
        subprocess.run([
            'ffmpeg', '-y', '-i', str(ir_path),
            '-vcodec', 'libx265', '-crf', '23', '-preset', 'medium',
            str(compressed_ir_path),
        ])

    print("Finished compressing videos.")
    if has_ir:
        print("You may now delete the original depth.avi and ir.avi files.")
    elif crop:
        print("You may now delete the original depth.avi file.")


if __name__ == '__main__':
    main()