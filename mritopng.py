import os
import numpy as np
from PIL import Image
import pydicom as dicom
import argparse


def mri_to_png(mri_path, png_path, save_all=False, gif_path=None):
    """ Function to convert from a DICOM image to png

        @param mri_path: Path to the mri file
        @param png_path: Path to the generated png file (or base for frames)
        @param save_all: If True, save all frames (for multi-frame DICOMs)
    """

    # Extracting data from the mri file
    plan = dicom.dcmread(mri_path)
    arr = np.asarray(plan.pixel_array)

    # Remove singleton dimensions
    arr = np.squeeze(arr)

    # If there are multiple frames (4D) and user requested all frames,
    # iterate and save each frame separately.
    if arr.ndim == 4 and save_all:
        base = os.path.splitext(png_path)[0]
        frames = []
        for i in range(arr.shape[0]):
            frame = np.squeeze(arr[i])

            # If channels are the first axis (3,4,H,W), move them to last
            if frame.ndim == 3 and frame.shape[0] in (3, 4) and (frame.shape[2] not in (3, 4)):
                frame = np.transpose(frame, (1, 2, 0))

            # If still multi-channel and not RGB/RGBA, pick first channel
            if frame.ndim == 3 and frame.shape[2] not in (3, 4):
                frame = frame[..., 0]

            # Normalize
            f = frame.astype(np.float32)
            f -= f.min()
            m = f.max()
            if m > 0:
                f = (f / m) * 255.0
            f_uint8 = f.astype(np.uint8)
            img = Image.fromarray(f_uint8)
            frames.append(img)

            # Also save individual frame PNGs
            out_path = f"{base}_frame{i:03d}.png"
            img.save(out_path, format='PNG')

        # If requested, save as animated GIF as well
        if gif_path:
            # Convert frames to palette mode suitable for GIF
            pal_frames = [f.convert('P', palette=Image.ADAPTIVE) for f in frames]
            pal_frames[0].save(gif_path, save_all=True, append_images=pal_frames[1:], duration=100, loop=0)
        return

    # If there are multiple frames (4D), pick the first frame
    if arr.ndim == 4:
        arr = arr[0]

    # If channels are the first axis (3,4,H,W), move them to last
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and (arr.shape[2] not in (3, 4)):
        arr = np.transpose(arr, (1, 2, 0))

    # If still multi-channel/frame and not a common 3/4-channel image,
    # pick the first channel/frame
    if arr.ndim == 3 and arr.shape[2] not in (3, 4):
        arr = arr[..., 0]

    # Convert to float and normalize to 0-255 uint8
    arr = arr.astype(np.float32)
    arr -= arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr = (arr / max_val) * 255.0
    arr_uint8 = arr.astype(np.uint8)

    # Create PIL image and save as PNG to the provided path
    img = Image.fromarray(arr_uint8)
    img.save(png_path, format='PNG')


def convert_file(mri_file_path, png_file_path, save_all=False, gif_path=None):
    """ Function to convert an MRI binary file to a
        PNG image file.

        @param mri_file_path: Full path to the mri file
        @param png_file_path: Fill path to the png file
    """

    # Making sure that the mri file exists
    if not os.path.exists(mri_file_path):
        raise Exception('File "%s" does not exists' % mri_file_path)

    # Making sure the png file does not exist (unless saving all frames)
    if os.path.exists(png_file_path) and not save_all:
        raise Exception('File "%s" already exists' % png_file_path)

    mri_to_png(mri_file_path, png_file_path, save_all=save_all, gif_path=gif_path)


def convert_folder(mri_folder, png_folder, save_all=False, gif_path=None):
    """ Convert all MRI files in a folder to png files
        in a destination folder
    """

    # Create the folder for the png directory structure (safe if exists)
    os.makedirs(png_folder, exist_ok=True)

    # Recursively traverse all sub-folders in the path
    for mri_sub_folder, subdirs, files in os.walk(mri_folder):
        for mri_file in os.listdir(mri_sub_folder):
            mri_file_path = os.path.join(mri_sub_folder, mri_file)

            # Make sure path is an actual file
            if os.path.isfile(mri_file_path):

                # Replicate the original file structure
                rel_path = os.path.relpath(mri_sub_folder, mri_folder)
                png_folder_path = os.path.join(png_folder, rel_path)
                if not os.path.exists(png_folder_path):
                    os.makedirs(png_folder_path)
                png_file_path = os.path.join(png_folder_path, '%s.png' % mri_file)

                try:
                    # Convert the actual file
                    convert_file(mri_file_path, png_file_path, save_all=save_all, gif_path=gif_path)
                    print(f'SUCCESS>', mri_file_path, '-->', png_file_path)
                except Exception as e:
                    print(f'FAIL>', mri_file_path, '-->', png_file_path, ':', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a dicom MRI file to png")
    parser.add_argument('-f', action='store_true')
    parser.add_argument('-a', '--all-frames', action='store_true', help='Save all frames as separate PNGs')
    parser.add_argument('-g', '--gif', dest='gif_path', help='Path to save animated GIF (for multi-frame DICOM)')
    parser.add_argument('dicom_path', help='Full path to the mri file')
    parser.add_argument('png_path', help='Full path to the generated png file')

    args = parser.parse_args()
    print(args)
    if args.f:
        convert_folder(args.dicom_path, args.png_path, save_all=args.all_frames, gif_path=args.gif_path)
    else:
        convert_file(args.dicom_path, args.png_path, save_all=args.all_frames, gif_path=args.gif_path)