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
    def __iter__(self):
        """ Allows the TrainingOutput dataclass to be iterable, which can be used with functions like `tabulate`."""
        yield self.epoch_range
        yield self.epoch_average_score
        yield self.epsilon
        yield self.learning_rate

@dataclass
class TestingOutput:
    game: int
    score: int
    high_tile: int
    moves_before_break: int
    game_output: int
    def __iter__(self):
        """ Allows the TrainingOutput dataclass to be iterable, which can be used with functions like `tabulate`."""
        yield self.game
        yield self.score
        yield self.high_tile
        yield self.moves_before_break
        yield self.game_output
@dataclass
class PhaseDetails:
    phase_total_score: int
    phase_high_tile: int
    phase_moves_before_break: int
    phase_game_break: int
    phase_training_details: EpochTrainingDetails
    phase_testing_details: EpochTestingDetails

@dataclass
class PhaseBuffer:
    phase_average_score: int
    phase_average_high_tile: int
    phase_average_moves_before_break: int
    phase_average_game_break: int
    phase_training_output: TrainingOutput
    phase_testing_output: TestingOutput
