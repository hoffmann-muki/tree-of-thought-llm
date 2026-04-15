import os
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

class Task:
    def __init__(self):
        self.evaluator_model = None
        self.evaluator_temperature = 0.0

    def __len__(self) -> int:
        pass

    def get_input(self, idx: int) -> str:
        pass

    def test_output(self, idx: int, output: str):
        pass

    def configure_evaluator(self, model=None, temperature=0.0):
        self.evaluator_model = model
        self.evaluator_temperature = temperature