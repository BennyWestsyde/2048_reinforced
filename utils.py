import os

import torch


def get_device():
	if torch.cuda.is_available():
		print("Cuda Device Count: ", torch.cuda.device_count())  # Number of GPUs on the machine
		print("Cuda Current Device: ", torch.cuda.current_device())  # Device number of the active GPU (e.g., 0)
		device = torch.device("cuda")
	elif torch.backends.mps.is_available():
		print("Metal Device Count: 1")  # Number of GPUs on the machine
		print("Metal Current Device: 0")  # Device number of the active GPU (e.g., 0)
		device = torch.device("mps")
	else:
		print("No GPU available, using CPU")
		device = torch.device("cpu")

	torch.set_default_device(device)
	return device


def init_filesystem(run_identifier):
	if not os.path.exists("Outputs"):
		os.makedirs("Outputs")

	if not os.path.exists("Models"):
		os.makedirs("Models")

	if not os.path.exists(f"Outputs/{run_identifier}"):
		os.makedirs(f"Outputs/{run_identifier}")

	if not os.path.exists(f"Models/{run_identifier}"):
		os.makedirs(f"Models/{run_identifier}")

	if not os.path.exists(f"Outputs/{run_identifier}/images"):
		os.makedirs(f"Outputs/{run_identifier}/images")

	if not os.path.exists(f"Outputs/{run_identifier}/images/average_scores"):
		os.makedirs(f"Outputs/{run_identifier}/images/average_scores")

	if not os.path.exists(f"Outputs/{run_identifier}/images/high_tiles"):
		os.makedirs(f"Outputs/{run_identifier}/images/high_tiles")

	if not os.path.exists(f"Outputs/{run_identifier}/images/moves_before_break"):
		os.makedirs(f"Outputs/{run_identifier}/images/moves_before_break")

	if not os.path.exists(f"Outputs/{run_identifier}/average_scores.csv"):
		with open(f"Outputs/{run_identifier}/average_scores.csv", "w") as f:
			f.write("phase,average_score\n")

	if not os.path.exists(f"Outputs/{run_identifier}/high_tiles.csv"):
		with open(f"Outputs/{run_identifier}/high_tiles.csv", "w") as f:
			f.write("phase,high_tile\n")

	if not os.path.exists(f"Outputs/{run_identifier}/moves_before_break.csv"):
		with open(f"Outputs/{run_identifier}/moves_before_break.csv", "w") as f:
			f.write("phase,moves_before_break\n")
