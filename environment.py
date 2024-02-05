import numpy as np

from typing_extensions import Self
from typing import List, Tuple, Dict


class Code:

    code_length: int = 4
    num_colours: int = 6
    code_space: int = (num_colours ** code_length) - 1

    def __init__(self, code: str):
        self.code = code

    @classmethod
    def set_code_space(cls: Self, code_length: int, num_colours: int):
        cls.code_length = code_length
        cls.num_colours = num_colours
        cls.code_space = (num_colours ** code_length) - 1

    @classmethod
    def from_index(cls: Self, index: int) -> Self:
        digits = []

        while index > 0:
            digits.append(str(index % cls.num_colours))
            index //= cls.num_colours

        code_str = ''.join(reversed(digits)).zfill(cls.code_length)

        return cls(code_str)

    def to_index(self) -> int:
        return int(self.code, base=self.num_colours)


class Feedback:
    
    def __init__(self, secret: Code, guess: Code):
        self.black_pegs = 0
        self.white_pegs = 0

        self._prepare_feedback(secret.code, guess.code)

    def _prepare_feedback(self, secret: str, guess: str) -> Self:
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


class Observation:

    def __init__(self, guess: Code, feedback: Feedback):
        self.guess = guess
        self.feedback = feedback


class State:

    def __init__(self):
        self.history: List[Observation] = []

    def add_observation(self, guess: Code, feedback: Feedback):
        self.history.append(Observation(guess, feedback))


class Mastermind:

    def __init__(
        self, 
        code_length: int = 4,
        num_colours: int = 6,
        max_attempts: int = float('inf'),
        seed: int = 42,
    ) -> Self:
        assert(0 < num_colours <= 9)

        self.code_length = code_length
        self.num_colours = num_colours
        self.max_attempts = max_attempts
        self.action_space = (num_colours ** code_length) - 1
        Code.set_code_space(code_length, num_colours)

        self._reseed(seed)
        self.reset()

    def reset(self) -> State:
        self.attempts = 0
        self.state = State()
        self.secret_code = self._generate_secret_code()

        return self.state
    
    def step(self, action: int) -> Tuple[State, float, bool, Dict]:
        if self.attempts == self.max_attempts:
            return self.state, 0, True, {}
        
        guess = Code.from_index(action)
        feedback = Feedback(self.secret_code, guess)

        self.state.add_observation(guess, feedback)
        self.attempts += 1

        reward = self.reward(feedback)
        done = feedback.black_pegs == self.code_length or self.attempts >= self.max_attempts

        return self.state, reward, done, {}

    def reward(self, feedback: Feedback) -> float:
        if feedback.black_pegs == self.code_length:
            return 100 # Win reward
        else:
            return -1 # Penalty for incorrect guess

    def _generate_secret_code(self) -> Code:
        """
        Randomly selects a secret code.
        """
        return Code.from_index(np.random.randint(self.action_space+1))

    def _reseed(self, seed: int):
        """
        Reseeds all random number generators.
        """
        self.seed = seed
        np.random.seed(seed)


if __name__ == "__main__":
    wait = 2