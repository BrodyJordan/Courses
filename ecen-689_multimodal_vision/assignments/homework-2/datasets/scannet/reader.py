import argparse
import os
from SensorData import SensorData

def main():
    parser = argparse.ArgumentParser(description='Read ScanNet .sens files')
    parser.add_argument('--filename', required=True, help='path to sens file to read')
    parser.add_argument('--output_path', required=True, help='path to output folder')
    parser.add_argument('--export_depth_images', action='store_true', help='export depth images')
    parser.add_argument('--export_color_images', action='store_true', help='export color images')
    parser.add_argument('--export_poses', action='store_true', help='export poses')
    parser.add_argument('--export_intrinsics', action='store_true', help='export intrinsics')
    parser.add_argument('--frame_skip', type=int, default=1, help='Extract every Nth frame (e.g., 10 to skip 90% of frames)')
    opt = parser.parse_args()

    print(f"Arguments: {opt}")

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path, exist_ok=True)

    print(f"Loading {opt.filename}...")
    sd = SensorData(opt.filename)
    print("Loaded successfully!")

    if opt.export_depth_images:
        sd.export_depth_images(os.path.join(opt.output_path, 'depth'), opt.frame_skip)
    if opt.export_color_images:
        sd.export_color_images(os.path.join(opt.output_path, 'color'), opt.frame_skip)
    if opt.export_poses:
        sd.export_poses(os.path.join(opt.output_path, 'pose'), opt.frame_skip)
    if opt.export_intrinsics:
        sd.export_intrinsics(os.path.join(opt.output_path, 'intrinsic'))

if __name__ == '__main__':
    main()