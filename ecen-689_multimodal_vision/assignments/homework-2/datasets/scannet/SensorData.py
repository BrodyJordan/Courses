import os
import struct
import zlib
import numpy as np
import imageio.v3 as iio

COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}

class RGBDFrame:
    def __init__(self):
        self.camera_to_world = np.zeros((4, 4))
        self.timestamp = 0
        self.depth_size = 0
        self.depth_zlib = b''
        self.color_size = 0
        self.color_jpg = b''

class SensorData:
    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = f.read(strlen).decode('utf-8')
            self.intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(64)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(64)), dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(64)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(64)), dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]
            num_frames = struct.unpack('Q', f.read(8))[0]
            
            self.frames = []
            for _ in range(num_frames):
                frame = RGBDFrame()
                frame.camera_to_world = np.asarray(struct.unpack('f'*16, f.read(64)), dtype=np.float32).reshape(4, 4)
                frame.timestamp = struct.unpack('Q', f.read(8))[0]
                frame.depth_size = struct.unpack('Q', f.read(8))[0]
                frame.depth_zlib = f.read(frame.depth_size)
                frame.color_size = struct.unpack('Q', f.read(8))[0]
                frame.color_jpg = f.read(frame.color_size)
                self.frames.append(frame)

    def export_depth_images(self, output_path, frame_skip=1):
        os.makedirs(output_path, exist_ok=True)
        print(f"Exporting depth images to {output_path}")
        for i in range(0, len(self.frames), frame_skip):
            depth_data = self.frames[i].depth_zlib
            if self.depth_compression_type == 'zlib_ushort':
                depth_data = zlib.decompress(depth_data)
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
            iio.imwrite(os.path.join(output_path, f'{i}.png'), depth)

    def export_color_images(self, output_path, frame_skip=1):
        os.makedirs(output_path, exist_ok=True)
        print(f"Exporting color images to {output_path}")
        for i in range(0, len(self.frames), frame_skip):
            color_data = self.frames[i].color_jpg
            iio.imwrite(os.path.join(output_path, f'{i}.jpg'), color_data)

    def export_poses(self, output_path, frame_skip=1):
        os.makedirs(output_path, exist_ok=True)
        print(f"Exporting poses to {output_path}")
        for i in range(0, len(self.frames), frame_skip):
            np.savetxt(os.path.join(output_path, f'{i}.txt'), self.frames[i].camera_to_world, fmt='%.8f')

    def export_intrinsics(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        print(f"Exporting intrinsics to {output_path}")
        np.savetxt(os.path.join(output_path, 'intrinsic_color.txt'), self.intrinsic_color, fmt='%.8f')
        np.savetxt(os.path.join(output_path, 'intrinsic_depth.txt'), self.intrinsic_depth, fmt='%.8f')