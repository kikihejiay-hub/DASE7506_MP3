"""Bounded best-first CPU search over independently implemented SQL operators."""
import importlib.util
import json
from pathlib import Path
import re
from sqlglot import exp


def local(name):
    spec = importlib.util.spec_from_file_location('planned_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = local('submission')
rules = local('relational_rules')


def relation_repairs(sql, schema):
    try:
        return core.schema_repairs(sql, schema)
    except (rules.sqlglot.errors.SqlglotError, RecursionError):
        return []


class SearchAgent:
    def __init__(self, config):
        self.config = config
        self.pool = {}
        self.initialized = False
        self.values_loaded = False

    def rank(self, sql, observation, schema):
        tree = rules.parse(sql)
        if tree is None:
            return -1000
        top = tree
        while isinstance(top, exp.SetOperation):
            top = top.this
        score = core.quality(top.sql(dialect='sqlite'), observation, schema)
        if isinstance(top, exp.Select):
            expressions = [e.this if isinstance(e, exp.Alias) else e for e in top.expressions]
            score -= 4 * (len(expressions) - len({e.sql(normalize=True) for e in expressions}))
        q = observation['task']['question'].lower()
        literal_words = set().union(*(rules.terms(str(v.this)) for v in tree.find_all(exp.Literal) if v.is_string))
        score += .4 * len(literal_words & rules.terms(q))
        if re.search(r'\bunique\b|\bdistinct\b|\bdifferent\b', q):
            score += 1.5 if tree.find(exp.Distinct) else -.5
        for select in tree.find_all(exp.Select):
            owners = rules.bindings(select, schema)
            projected_aliases = {e.alias.lower() for e in select.expressions if isinstance(e, exp.Alias)}
            # Ignore correlated columns owned by an enclosing SELECT.
            for column in select.find_all(exp.Column):
                if column.find_ancestor(exp.Select) is not select or column.is_star:
                    continue
                if not column.table and column.name.lower() in projected_aliases:
                    continue
                if column.table and column.table.lower() not in owners:
                    continue
                choices = [a for a, table in owners.items() if any(c['name'].lower() == column.name.lower() for c in table['columns'])]
                if not choices or column.table and column.table.lower() not in choices:
                    score -= 6
                elif not column.table and len(choices) > 1:
                    score -= 2
        return score

    def add(self, tree, origin, bonus, observation, schema):
        if tree is None:
            return
        tree = rules.qualify(tree, schema)
        sql = tree.sql(dialect='sqlite')
        if len(sql.encode('utf-8')) > 16000:
            return
        canonical = rules.key(sql)
        score = self.rank(sql, observation, schema) + bonus
        previous = self.pool.get(canonical)
        if previous is None or score > previous[1]:
            self.pool[canonical] = (sql, score, origin)

    def initialize(self, observation, schema, hints):
        unique = {}
        for sql, prior in [(observation['issue_sql'], 0)] + [(s, 1) for s in hints]:
            tree = rules.parse(sql)
            if tree is not None:
                canonical = rules.key(sql)
                unique[canonical] = (tree, max(prior, unique.get(canonical, (None, -1))[1]))
        for tree, prior in unique.values():
            self.add(tree, 'seed', prior, observation, schema)
            base = rules.qualify(tree, schema)
            forms = [base]
            if self.config['shape']:
                forms.extend(rules.projections(base, schema, observation['example']['columns']))
            if self.config['relations']:
                forms.extend(rules.parse(s) for s in relation_repairs(base.sql(dialect='sqlite'), schema))
            seen = set()
            for form in forms:
                if form is None or rules.key(form.sql()) in seen:
                    continue
                seen.add(rules.key(form.sql()))
                self.add(form, 'schema', prior + .15, observation, schema)
                edits = rules.predicate_edits(form, schema, observation['task']['question'])
                if self.config['entities']:
                    edits += rules.entity_edits(form, schema, observation['task']['question'])
                for edited, reason, confidence in edits:
                    self.add(edited, reason, prior + confidence, observation, schema)
                    if reason == 'alphabetical_attribute' and self.config['relations']:
                        for sql in relation_repairs(edited.sql(dialect='sqlite'), schema):
                            self.add(rules.parse(sql), reason, prior + confidence + .2, observation, schema)
                    if self.config['entities'] and reason in ('unused_join', 'exact_condition', 'unsupported_condition', 'minimal_conditions', 'entity_name_condition'):
                        for combined, second_reason, second_confidence in rules.entity_edits(edited, schema, observation['task']['question']):
                            self.add(combined, reason + '+' + second_reason, prior + confidence + .7 * second_confidence, observation, schema)
                for sql in core.semantic_variants(form.sql(dialect='sqlite'), observation):
                    self.add(rules.parse(sql), 'legacy_semantic', prior + .35, observation, schema)
            if self.config['variants']:
                for sql in core.variants(base.sql(dialect='sqlite'), observation):
                    self.add(rules.parse(sql), 'small_edit', prior - 1, observation, schema)
        self.initialized = True

    def act(self, observation):
        history = observation['history']
        if not history:
            return {'tool': 'check', 'sql': observation['issue_sql']}
        hint = next((e for e in history if e['tool'] == 'repair_hint'), None)
        schema_event = next((e for e in history if e['tool'] == 'schema'), None)
        if observation['remaining']:
            if hint is None:
                return {'tool': 'repair_hint'}
            if schema_event is None:
                return {'tool': 'schema'}
        schema = schema_event['observation'] if schema_event else []
        if not self.initialized:
            self.initialize(observation, schema, hint['observation'] if hint else [])
        if not self.values_loaded and rules.discovered(history):
            self.values_loaded = True
            for sql, parent_score, _ in list(self.pool.values()):
                inherited = parent_score - self.rank(sql, observation, schema)
                consistent = rules.consistent_literals(rules.parse(sql), schema, history)
                if consistent is not None:
                    self.add(consistent, 'consistent_spelling', 4.7 + inherited, observation, schema)
                for tree, reason, confidence in rules.literal_edits(rules.parse(sql), schema, history, observation['task']['question']):
                    self.add(tree, reason, confidence + inherited, observation, schema)
                    if self.config['entities']:
                        for combined, second_reason, second_confidence in rules.entity_edits(tree, schema, observation['task']['question']):
                            if second_reason == 'unused_join':
                                self.add(combined, reason + '+' + second_reason, confidence + inherited + .7 * second_confidence, observation, schema)
        checked = {rules.key(e['sql']): e for e in history if e['tool'] == 'check'}
        def preference(event):
            return self.pool.get(rules.key(event['sql']), ('', self.rank(event['sql'], observation, schema), ''))[1]
        matched = [e for e in checked.values() if e['observation'].get('matches_example')]
        best = max(matched, key=preference) if matched else None
        pending = sorted((candidate for canonical, candidate in self.pool.items() if canonical not in checked), key=lambda c: (-c[1], len(c[0])))
        next_candidate = pending[0] if pending else None
        can_probe = (self.config['values'] and observation['remaining'] >= 2 and len(checked) >= 2 and
                     not any(e['tool'] == 'execute' for e in history))
        if can_probe and (not matched or observation['example'].get('row_count') == 0):
            seeds = [rules.parse(sql) for sql, _, _ in sorted(self.pool.values(), key=lambda c: -c[1])[:6]]
            query = rules.probe(seeds, schema)
            if query:
                return {'tool': 'execute', 'sql': query}
        if best and (not next_candidate or next_candidate[1] <= preference(best)):
            return {'tool': 'submit', 'sql': best['sql']}
        if observation['remaining'] and next_candidate:
            return {'tool': 'check', 'sql': next_candidate[0]}
        if best:
            return {'tool': 'submit', 'sql': best['sql']}
        executable = [e for e in checked.values() if e['observation'].get('ok')]
        fallback = max(executable or list(checked.values()), key=lambda e: (preference(e), -abs(e['observation'].get('row_count', 0) - observation['example'].get('row_count', 0))))
        return {'tool': 'submit', 'sql': fallback['sql']}


def create_agent(checkpoint=None):
    config = dict(entities=True, relations=True, shape=True, values=True, variants=True)
    if checkpoint:
        supplied = json.loads(Path(checkpoint).read_text(encoding='utf-8'))
        if set(supplied) - set(config) or any(type(v) is not bool for v in supplied.values()):
            raise ValueError('Unsupported planner configuration')
        config.update(supplied)
    return SearchAgent(config)
