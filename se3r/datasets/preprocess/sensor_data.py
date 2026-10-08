"""
ScanNet .sens file parser - Python 3 compatible version.

Adapted from ScanNet/SensReader/python/SensorData.py (originally Python 2).
Reads compressed RGB-D sensor streams and exports color images, depth maps,
camera poses, and intrinsics in the format expected by AMB3R's Scannet dataset.

Reference format (ScanNet .sens v4):
  - Per-frame: 4x4 camera_to_world, timestamps, compressed color/depth data
  - Global: color/depth intrinsics & extrinsics, image dimensions, depth_shift
"""

import os
import struct
import numpy as np
import zlib
import cv2


COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


class RGBDFrame:
    """A single RGB-D frame from a .sens file."""

    def load(self, file_handle):
        """Load frame data from a binary file handle."""
        # 4x4 camera-to-world transformation matrix
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        # In Python 3, struct.unpack('c', ...) returns bytes objects
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

    def decompress_depth(self, compression_type):
        """Decompress depth data based on the compression type."""
        if compression_type == "zlib_ushort":
            return self.decompress_depth_zlib()
        else:
            raise ValueError(f"Unknown depth compression type: {compression_type}")

    def decompress_depth_zlib(self):
        """Decompress zlib-compressed depth data."""
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        """Decompress color data based on the compression type."""
        if compression_type == "jpeg":
            return self.decompress_color_jpeg()
        else:
            raise ValueError(f"Unknown color compression type: {compression_type}")

    def decompress_color_jpeg(self):
        """Decode JPEG-compressed color data using OpenCV."""
        # Convert bytes to numpy array and decode JPEG
        color_array = np.frombuffer(self.color_data, dtype=np.uint8)
        # cv2.imdecode returns BGR, convert to RGB
        color_image = cv2.imdecode(color_array, cv2.IMREAD_COLOR)
        if color_image is not None:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        return color_image


class SensorData:
    """
    Parser for ScanNet .sens binary format (version 4).

    Loads the entire sensor stream into memory and provides export
    methods for color images, depth maps, camera poses, and intrinsics.
    """

    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        """Load and parse the .sens binary file."""
        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            assert self.version == version, (
                f"Expected .sens version {self.version}, got {version}"
            )
            strlen = struct.unpack("Q", f.read(8))[0]
            # In Python 3, read bytes and decode to string
            self.sensor_name = f.read(strlen).decode("utf-8", errors="replace")

            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[
                struct.unpack("i", f.read(4))[0]
            ]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[
                struct.unpack("i", f.read(4))[0]
            ]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            self.num_frames = num_frames
            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export_depth_images(self, output_path, image_size=None, frame_skip=1):
        """
        Export depth frames as 16-bit PNG images.

        Args:
            output_path: Directory to save depth images.
            image_size: Optional (height, width) tuple to resize depth maps.
            frame_skip: Export every N-th frame. Default 1 (all frames).
        """
        os.makedirs(output_path, exist_ok=True)
        num_export = len(range(0, len(self.frames), frame_skip))
        print(f"  Exporting {num_export} depth frames to {output_path}")

        for idx, f_idx in enumerate(range(0, len(self.frames), frame_skip)):
            depth_data = self.frames[f_idx].decompress_depth(self.depth_compression_type)
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                self.depth_height, self.depth_width
            )
            if image_size is not None:
                depth = cv2.resize(
                    depth,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            # Write 16-bit PNG using OpenCV
            out_file = os.path.join(output_path, f"{idx}.png")
            cv2.imwrite(out_file, depth)

    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        """
        Export color frames as JPEG images.

        Args:
            output_path: Directory to save color images.
            image_size: Optional (height, width) tuple to resize images.
            frame_skip: Export every N-th frame. Default 1 (all frames).
        """
        os.makedirs(output_path, exist_ok=True)
        num_export = len(range(0, len(self.frames), frame_skip))
        print(f"  Exporting {num_export} color frames to {output_path}")

        for idx, f_idx in enumerate(range(0, len(self.frames), frame_skip)):
            color = self.frames[f_idx].decompress_color(self.color_compression_type)
            if color is None:
                print(f"  Warning: Failed to decode color frame {f_idx}, skipping")
                # Write a black image as placeholder
                if image_size is not None:
                    color = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
                else:
                    color = np.zeros(
                        (self.color_height, self.color_width, 3), dtype=np.uint8
                    )
            if image_size is not None:
                color = cv2.resize(
                    color,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            out_file = os.path.join(output_path, f"{idx}.jpg")
            # Convert RGB to BGR for cv2.imwrite
            cv2.imwrite(out_file, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

    def save_mat_to_file(self, matrix, filename):
        """Save a matrix to a text file in the ScanNet format."""
        with open(filename, "w") as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt="%f")

    def export_poses(self, output_path, frame_skip=1):
        """
        Export camera poses as 4x4 text files (camera-to-world).

        Args:
            output_path: Directory to save pose files.
            frame_skip: Export every N-th frame. Default 1 (all frames).
        """
        os.makedirs(output_path, exist_ok=True)
        num_export = len(range(0, len(self.frames), frame_skip))
        print(f"  Exporting {num_export} camera poses to {output_path}")

        for idx, f_idx in enumerate(range(0, len(self.frames), frame_skip)):
            self.save_mat_to_file(
                self.frames[f_idx].camera_to_world,
                os.path.join(output_path, f"{idx}.txt"),
            )

    def export_intrinsics(self, output_path):
        """
        Export camera intrinsic and extrinsic matrices.

        Exports 4 files:
          - intrinsic_color.txt: 4x4 color camera intrinsic matrix
          - extrinsic_color.txt: 4x4 color camera extrinsic matrix
          - intrinsic_depth.txt: 4x4 depth camera intrinsic matrix
          - extrinsic_depth.txt: 4x4 depth camera extrinsic matrix
        """
        os.makedirs(output_path, exist_ok=True)
        print(f"  Exporting camera intrinsics to {output_path}")
        self.save_mat_to_file(
            self.intrinsic_color, os.path.join(output_path, "intrinsic_color.txt")
        )
        self.save_mat_to_file(
            self.extrinsic_color, os.path.join(output_path, "extrinsic_color.txt")
        )
        self.save_mat_to_file(
            self.intrinsic_depth, os.path.join(output_path, "intrinsic_depth.txt")
        )
        self.save_mat_to_file(
            self.extrinsic_depth, os.path.join(output_path, "extrinsic_depth.txt")
        )
