import os
import subprocess
import sys
import random

curr_dir = os.getcwd()
print(f"Current directory: {curr_dir}")

file_paths = sys.argv[1:]
# Check if the file paths are valid
file_flag = False
dir_flag = False
for file_path in file_paths:
	if not os.path.exists(file_path):
		print(f"Invalid file path: {file_path}")
		exit()
	if os.path.isfile(file_path):
		file_flag = True
	if os.path.isdir(file_path):
		dir_flag = True

if file_flag and dir_flag:
	print("Please provide either a directory or a list of files, not both.")
	exit()
elif not file_flag and not dir_flag:
	print("Please provide a directory or a list of files.")
	exit()
elif dir_flag:
	file_paths = [os.path.join(file_paths[0], f) for f in os.listdir(file_paths[0]) if f.endswith(".png")]


file_paths.sort(key=lambda x: int(x.split(os.sep)[-1].split(".")[0]))
print(f"No. of file paths: {len(file_paths)}")

input_path = file_paths[0]
input_dir = os.path.dirname(input_path)
print(f"Input directory: {input_dir}")
input_dir2 = os.path.dirname(input_dir)
print(f"Input directory2: {input_dir2}")
input_dir3 = os.path.dirname(input_dir2)
print(f"Input directory3: {input_dir3}")

input_base = os.path.basename(input_dir)
print(f"Input base: {input_base}")


video_length = 30  # Length of the video in seconds
fps = len(file_paths) / video_length  # Frames per second


# Create temporary directory to store the video frames
temp_dir = os.path.join("tmp", "ffmpeg_frames"+str(random.randint(1, 1000)))
print(f"Creating temporary directory at: {temp_dir}")
os.makedirs(temp_dir, exist_ok=True)

# Create a list of frames for ffmpeg to take as input
for i, file_path in enumerate(file_paths):
	new_file_path = os.path.join(temp_dir, f"{i}.png")
	os.system(f"cp {file_path} {new_file_path}")

# Create the video
video_path = os.path.join(input_dir3, f"{input_base}_{len(file_paths)}.mp4")
print(f"Creating video at: {video_path}")

# Find the path to ffmpeg
ffmpeg_path = "/opt/homebrew/bin/ffmpeg"
if not ffmpeg_path:
	print("ffmpeg not found. Please install ffmpeg.")
	exit()

# Run ffmpeg to create the video
subprocess.run([ffmpeg_path, "-y", "-framerate", str(fps), "-i", f"{temp_dir}/%d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path])