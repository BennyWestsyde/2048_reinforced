import multiprocessing

from utils import get_device

state_size = 16  # 4x4 grid flattened
action_size = 4  # left, right, up, down
batch_size = 64


repetition_allowance = 10
video_length = 15  #seconds

num_phases = 100000
num_episodes_per_phase = 40
num_outputs_per_episode = 10
num_actions_per_episode = 5
num_tests = 5

save_model_interval = 1000
save_plot_interval = 10
save_video_interval = num_phases // 100
performance_threshold = 5
cross_phase_total_score = 0

slice_size = (num_episodes_per_phase // num_outputs_per_episode)
percentage_kept= 100
cpu_cores = multiprocessing.cpu_count()

device = get_device()