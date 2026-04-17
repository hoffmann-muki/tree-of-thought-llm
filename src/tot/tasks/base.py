import os
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

class Task:
    def __init__(self):
        self.evaluator_model = None
        self.evaluator_temperature = 0.0
        self.propose_max_tokens = 1000
        self.value_max_tokens = 1000
        self.vote_max_tokens = 1000
        self.sample_max_tokens = 1000

    def __len__(self) -> int:
        pass

    def get_input(self, idx: int) -> str:
        pass

    def test_output(self, idx: int, output: str):
        pass

    def configure_evaluator(self, model=None, temperature=0.0):
        self.evaluator_model = model
        self.evaluator_temperature = temperature