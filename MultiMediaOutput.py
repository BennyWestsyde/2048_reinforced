import os
import subprocess

import pandas

from KEYS import *
import matplotlib.pyplot as plt
import numpy as np


def save_video(model_id, metric, curr_phase):
    output_dir = f"Outputs/{model_id}/images/{metric}"
    num_files = curr_phase // save_plot_interval
    desired_fps = num_files // video_length

    # Calculate previous phase based on current phase and interval
    prev_phase = curr_phase - save_video_interval

    prev_video_path = f"Outputs/{model_id}/{metric}_{prev_phase}.mp4"
    new_video_temp_path = f"Outputs/{model_id}/images/{metric}_temp.mp4"
    final_video_path = f"Outputs/{model_id}/{metric}_{curr_phase}.mp4"

    # Convert images to video
    cmd_convert_images = [
        "ffmpeg", "-y",
        "-r", str(max(1, num_files // video_length)),  # Avoid division by zero for frame rate
        "-i", os.path.join(output_dir, "%d.png"),
        "-vf", "scale=1280:-1",
        "-preset", "veryslow",
        "-crf", "24",
        "-pix_fmt", "yuv420p",  # For compatibility
        new_video_temp_path
        ]
    subprocess.run(cmd_convert_images, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Check if a previous video exists and concatenate
    if os.path.exists(prev_video_path):
        with open("concat_list.txt", "w") as f:
            f.write(f"file '{prev_video_path}'\n")
            f.write(f"file '{new_video_temp_path}'\n")

        cmd_concat = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", "concat_list.txt",
            "-c", "copy",
            os.path.splitext(final_video_path)[0] + "_temp.mp4"  # Output concatenated video
            ]
        subprocess.run(cmd_concat, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        # No previous video, just rename temp to final
        os.rename(new_video_temp_path, final_video_path)

    # Adjust frame rate of final video, if necessary
    cmd_adjust_fps = [
        "ffmpeg", "-y",
        "-i", os.path.splitext(final_video_path)[0] + "_temp.mp4",
        "-r", str(desired_fps),
        "-preset", "veryslow",
        "-crf", "24",
        "-pix_fmt", "yuv420p",
        final_video_path  # Output adjusted video
        ]

    # Optionally replace original final video with the adjusted one
    os.rename(os.path.splitext(final_video_path)[0] + "_adjusted.mp4", final_video_path)

    # Clean up temp files if needed
    os.remove(new_video_temp_path)
    os.remove("concat_list.txt")
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))


def save_plot(base_name, file, phase):
    # Assuming 'curr_dt' and 'phase' are defined outside this function and passed as arguments
    temp_df = pandas.read_csv(f"Outputs/{base_name}/{file}.csv", index_col=0)
    temp_df.columns = ['value']
    plt.figure(figsize=(10, 6))
    plt.scatter(temp_df.index, temp_df['value'])

    # Determine the degree for the polynomial fit
    # Minimum degree is 1 (linear), and maximum is set to 5 for practicality
    num_points = len(temp_df.index)
    degree = 1
    while num_points // 4 ** degree > 0:
        degree = min(degree + 1, 10)  # Example dynamic degree adjustment
    if num_points > 1:
        z = np.polyfit(temp_df.index.astype(float), temp_df['value'], degree)
        p = np.poly1d(z)
        # Add trendline to plot, ensuring index is handled as float for large datasets
        plt.plot(temp_df.index, p(temp_df.index.astype(float)), color='red')
        if degree >= 2:
            z2 = np.polyfit(temp_df.index.astype(float), temp_df['value'], 1)
            p2 = np.poly1d(z2)
            plt.plot(temp_df.index, p2(temp_df.index.astype(float)), color='green', linestyle='--')

    # Define title and y-label dynamically
    title_map = {
        'average_scores': ('Average Scores Over Phases', 'Average Score'),
        'high_tiles': ('High Tile Over Phases', 'High Tile'),
        'moves_before_break': ('Moves Before Break Over Phases', 'Moves Before Break')
        }
    output_path = None
    for key, (title, ylabel) in title_map.items():
        if key in file:
            plt.title(title)
            plt.ylabel(ylabel)
            output_path = f"Outputs/{base_name}/images/{key}/{phase + 1}.png"
            break
    if not output_path:
        output_path = f"Outputs/{base_name}/images/{file}/{phase + 1}.png"

    plt.xlabel('Episode')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
