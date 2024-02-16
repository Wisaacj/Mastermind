import numpy as np
import gymnasium as gym

from typing_extensions import Self
from typing import Tuple, Dict, Deque
from collections import deque

# TODO: Implement `Mastermind` as a subclass of `gym.Env`


class Code:

    code_length: int = 4
    num_colours: int = 6
    num_codes: int = num_colours ** code_length
    # vector_shape: int = num_colours * code_length
    vector_shape: int = code_length

    def __init__(self, code: str):
        self.code = code

    @classmethod
    def set_code_space(cls: Self, code_length: int, num_colours: int):
        cls.code_length = code_length
        cls.num_colours = num_colours
        cls.num_codes = num_colours ** code_length
        # cls.vector_shape = num_colours * code_length
        cls.vector_shape = code_length

    @classmethod
    def from_index(cls: Self, index: int) -> Self:
        digits = []

        while index > 0:
            digits.append(str(index % cls.num_colours))
            index //= cls.num_colours

        code_str = ''.join(reversed(digits)).zfill(cls.code_length)

        return cls(code_str)
    
    @classmethod
    def sample(cls: Self) -> Self:
        """
        Randomly samples a code from the code space.
        """
        return cls.from_index(np.random.randint(cls.num_codes))

    def to_index(self) -> int:
        return int(self.code, base=self.num_colours)
    
    # def as_vector(self) -> np.ndarray:
    #     """
    #     Returns a 1-hot encoded vector representation of the code.
    #     """
    #     code_vector = np.zeros(self.num_colours*self.code_length, dtype=np.float32)

    #     for pos, val in enumerate(self.code):
    #         colour_index = int(val)
    #         code_vector[pos*self.num_colours + colour_index] = 1

    #     return code_vector

    def as_vector(self) -> np.ndarray:
        """
        Returns a normalized numerical vector representation of the code.
        """
        code_vector = np.zeros(self.code_length, dtype=np.float32)

        for pos, val in enumerate(self.code):
            code_vector[pos] = int(val) / (self.num_colours - 1)
            
        return code_vector


class Feedback:

    vector_shape: int = 2
    
    def __init__(self, secret: Code, guess: Code):
        self.black_pegs = 0
        self.white_pegs = 0

        self._prepare_feedback(secret.code, guess.code)

    def _prepare_feedback(self, secret: str, guess: str):
        secret_counts = [0] * Code.num_colours
        guess_counts = [0] * Code.num_colours

        # Count black pegs and prepare data for counting white pegs.
        for (s, g) in zip(secret, guess):
            if s == g:
                self.black_pegs += 1
            else:
                secret_counts[int(s)] += 1
                guess_counts[int(g)] += 1

        # Count white pegs.
        for i in range(Code.num_colours):
            self.white_pegs += min(secret_counts[i], guess_counts[i])

    def as_vector(self) -> np.ndarray:
        """
        Returns a 2D vector (black pegs, white pegs).
        """
        return np.array([self.black_pegs, self.white_pegs], dtype=np.float32)


class Observation:

    vector_shape: int = Code.vector_shape + Feedback.vector_shape

    def __init__(self, guess: Code, feedback: Feedback):
        self.guess = guess
        self.feedback = feedback

    @classmethod
    def refresh_vector_shape(cls: Self):
        cls.vector_shape = Code.vector_shape + Feedback.vector_shape

    def as_vector(self) -> np.ndarray:
        """
        Returns a vector representation of the observation where the first 
        `num_colours * code_length` elements describe a 1-hot encoded vector representation
        of the guess and the final two elements detail the feedback given (black pegs, 
        white pegs).
        """
        return np.concatenate([self.guess.as_vector(), self.feedback.as_vector()], dtype=np.float32)


class State:

    def __init__(self, history_length: int):
        self.history_length = history_length
        self.history: Deque[Observation] = deque(maxlen=history_length)

    @property
    def vector_shape(self) -> Tuple[int]:
        return (self.history_length, Observation.vector_shape)

    def add_observation(self, guess: Code, feedback: Feedback):
        self.history.append(Observation(guess, feedback))

    def encode(self) -> np.ndarray:
        """
        Encodes the current state into a fixed-sized numerical vector.
        """
        state_vector = np.zeros(self.vector_shape, dtype=np.float32)

        for i, obs in enumerate(self.history):
            state_vector[i] = obs.as_vector()

        return state_vector

    def render(self):
        print("History of Guesses and Feedback")

        for i, obs in enumerate(self.history):
            guess = obs.guess.code
            feedback = f"Black pegs: {obs.feedback.black_pegs}, " + \
                f"White pegs: {obs.feedback.white_pegs}"
            print(f"Attempt {i}: Guess = {guess}, Feedback = ({feedback})")


class Mastermind:

    def __init__(
        self, 
        code_length: int = 4,
        num_colours: int = 6,
        max_attempts: int = 20,
        history_length: int = 20,
        seed: int = 42,
    ) -> Self:
        assert(0 < num_colours <= 9)

        self.max_attempts = max_attempts
        self.history_length = history_length
        self.configure_codes(code_length, num_colours)

        self.reward_range = [-max_attempts, 1]
        self.reset(seed)

    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict]:
        self.attempts = 0
        self.state = State(self.history_length)

        if seed is not None:
            self.set_seed(seed)
            self.reset_spaces(seed)

        self.secret_code: Code = Code.sample()

        return self.state.encode(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.attempts == self.max_attempts:
            return self.state.encode(), 0, True, True, {}
        
        guess = Code.from_index(action)
        feedback = Feedback(self.secret_code, guess)

        self.state.add_observation(guess, feedback)
        self.attempts += 1

        reward = self.reward(feedback)
        done = feedback.black_pegs == self.code_length or self.attempts >= self.max_attempts

        return self.state.encode(), reward, done, done, {}

    def reward(self, feedback: Feedback) -> float:
        if feedback.black_pegs == self.code_length:
            return 1 # Win reward
        else:
            return -1 # Penalty for incorrect guess
        
    def render(self, reveal_secret: bool = True):
        print(f"Current Attempts: {self.attempts}")

        if reveal_secret:
            print(f"Secret Code: {self.secret_code.code}")

        self.state.render()

    def configure_codes(self, code_length: int, num_colours: int):
        self.code_length = code_length
        self.num_colours = num_colours

        Code.set_code_space(code_length, num_colours)
        Observation.refresh_vector_shape()

    def reset_spaces(self, seed: int):
        self.action_space = gym.spaces.Discrete(Code.num_codes, seed=seed)
        self.observation_space = gym.spaces.Box(
            0, Code.code_length, self.state.vector_shape, seed=seed)

    def set_seed(self, seed: int):
        """
        Reseeds all random number generators.
        """
        self.seed = seed
        np.random.seed(seed)

    def close(self):
        """
        To satisfy the pseudo abc.
        """
        pass


if __name__ == "__main__":
    env = Mastermind()

    for i in range(10):
        guess = env.action_space.sample()
        env.step(guess)
        
    pass