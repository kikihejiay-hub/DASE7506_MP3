"""One optional own-LoRA completion, only after a genuine failed final check."""
import importlib.util
from pathlib import Path


def sibling(name):
    spec = importlib.util.spec_from_file_location('sparse_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cpu = sibling('terminal_agent')
runtime = sibling('model_agent')


class SparseAgent:
    def __init__(self, checkpoint=None):
        self.base = cpu.create_agent(checkpoint)
        self.attempted = False
        self.generated = None

    def act(self, observation):
        if self.generated:
            return {'tool':'submit', 'sql':self.generated}
        action = self.base.act(observation)
        if action['tool'] != 'submit' or self.attempted:
            return action
        history = observation['history']
        checks = [e for e in history if e['tool'] == 'check']
        if any(e['observation'].get('matches_example') for e in checks):
            return action
        receipt = next((e['observation'] for e in checks if e['sql'] == action['sql']), None)
        if receipt is None or receipt.get('matches_example') is not False:
            return action
        if not any(e['tool'] == 'schema' for e in history):
            return action
        self.attempted = True
        sql = runtime.generate(observation, action['sql'], 1)
        tried = {cpu.rules.key(e['sql']) for e in checks}
        if sql and cpu.rules.key(sql) not in tried:
            self.generated = sql
            return {'tool':'check' if observation['remaining'] else 'submit', 'sql':sql}
        return action


def create_agent(checkpoint=None):
    return SparseAgent(checkpoint)
