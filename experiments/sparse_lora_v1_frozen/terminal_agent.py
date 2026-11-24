"""Failure-only text-anchor rescue on top of the frozen CPU search design.

Inspired by the general failure-gating architecture in the reviewed public
repository. No external prediction, model, benchmark file, or task ID is read.
"""
import importlib.util
from pathlib import Path
import re
from sqlglot import exp

spec = importlib.util.spec_from_file_location('terminal_cpu_parent', Path(__file__).with_name('planned_agent.py'))
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
rules = parent.rules


def normalized(tree, schema, question):
    if not isinstance(tree, exp.Select) or tree.find(exp.Subquery) or tree.find(exp.CTE):
        return None
    tree = rules.qualify(tree, schema)
    owners = rules.bindings(tree, schema)
    for column in tree.find_all(exp.Column):
        if column.is_star:
            continue
        owner = owners.get(column.table.lower())
        if owner is None or not any(c['name'].lower() == column.name.lower() for c in owner['columns']):
            return None
    where = tree.args.get('where')
    if where is None:
        return None
    changed = False
    for predicate in list(where.find_all(exp.EQ)):
        column, literal = predicate.this, predicate.expression
        if not isinstance(column, exp.Column) or not isinstance(literal, exp.Literal) or not literal.is_string:
            continue
        text = literal.this.strip()
        if len(text) < 2 or not re.search(r'(?<!\w)' + re.escape(text) + r'(?!\w)', question, re.I):
            continue
        owner = owners[column.table.lower()]
        metadata = next(c for c in owner['columns'] if c['name'].lower() == column.name.lower())
        if not re.search(r'text|char', metadata['type'], re.I):
            continue
        predicate.replace(exp.EQ(this=exp.Lower(this=exp.Trim(this=column.copy())), expression=exp.Literal.string(text.lower())))
        changed = True
    return tree if changed else None


class TerminalAgent:
    def __init__(self, checkpoint=None):
        self.base = parent.create_agent(checkpoint)
        self.rescues = set()

    def act(self, observation):
        checks = [e for e in observation['history'] if e['tool'] == 'check']
        for event in checks:
            if rules.key(event['sql']) in self.rescues and event['observation'].get('matches_example'):
                return {'tool': 'submit', 'sql': event['sql']}
        action = self.base.act(observation)
        if action['tool'] != 'submit':
            return action
        receipt = next((e['observation'] for e in checks if e['sql'] == action['sql']), None)
        if receipt is None or receipt.get('matches_example') is not False:
            return action
        error = receipt.get('error') or {}
        deterministic = receipt.get('ok') is True or error.get('type') == 'OperationalError' and any(t in error.get('message', '').lower() for t in ('no such column', 'no such table', 'ambiguous column', 'syntax error'))
        if not deterministic:
            return action
        schema = next((e['observation'] for e in observation['history'] if e['tool'] == 'schema'), [])
        tried = {rules.key(e['sql']) for e in checks} | self.rescues
        candidates = []
        for sql, score, reason in self.base.pool.values():
            tree = normalized(rules.parse(sql), schema, observation['task']['question'])
            if tree is None:
                continue
            candidate = tree.sql(dialect='sqlite')
            if rules.key(candidate) not in tried:
                candidates.append((score, candidate))
        if not candidates:
            return action
        _, candidate = max(candidates, key=lambda c: (c[0], -len(c[1])))
        self.rescues.add(rules.key(candidate))
        return {'tool': 'check' if observation['remaining'] else 'submit', 'sql': candidate}


def create_agent(checkpoint=None):
    return TerminalAgent(checkpoint)
