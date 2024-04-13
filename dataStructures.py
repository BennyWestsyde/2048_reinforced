from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class GameState:
    action: int
    state: Dict
    reward: int
    done: bool

@dataclass
class Episode:
    states: List[GameState] = field(default_factory=list)

@dataclass
class EpisodeDetails:
    episode_score: int
    episode_buffer: Episode

@dataclass
class EpochTrainingDetails:
    epoch_training_score: int
    epoch_training_high_tile: int
    epoch_training_moves_before_break: int
    epoch_training_game_break: int
    episode_details: List[EpisodeDetails]

@dataclass
class EpochTestingDetails:
    epoch_testing_score: int
    epoch_testing_high_tile: int
    epoch_testing_moves_before_break: int
    epoch_testing_game_break: int
    epoch_testing_game_loss: int
    epoch_testing_game_win: int
    episode_details: List[EpisodeDetails]

@dataclass
class TrainingOutput:
    epoch_range: str
    epoch_average_score: int
    epsilon: float
    learning_rate: float

@dataclass
class TestingOutput:
    game: int
    score: int
    high_tile: int
    moves_before_break: int
    game_output: str

@dataclass
class PhaseDetails:
    phase_total_score: int
    phase_high_tile: int
    phase_moves_before_break: int
    phase_game_break: int
    phase_training_details: TrainingOutput
    phase_testing_details: TestingOutput

@dataclass
class PhaseBuffer:
    phase_average_score: int
    phase_average_high_tile: int
    phase_average_moves_before_break: int
    phase_average_game_break: int
    phase_training_details: TrainingOutput
    phase_testing_details: TestingOutput
