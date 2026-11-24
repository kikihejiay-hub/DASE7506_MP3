"""Offline, feedback-driven SQL repair controller; no benchmark file access."""
import json
import itertools
import re
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope


def variants(sql, observation=None):
    """Small generic edits, checked with real tools before final selection."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except (sqlglot.errors.SqlglotError, RecursionError):
        return []
    if not isinstance(tree, exp.Select):
        return []
    results = []
    if observation:
        width = len(observation['example']['columns'])
        expressions = tree.expressions
        if width and width <= 3 and len(expressions) <= 5:
            for selected in itertools.permutations(expressions, width):
                if list(selected) == expressions:
                    continue
                edited = tree.copy()
                edited.set('expressions', [e.copy() for e in selected])
                results.append(edited.sql(dialect='sqlite'))
        if tree.args.get('group') and width == 1 and len(expressions) > 1:
            for e in expressions:
                if not e.find(exp.AggFunc):
                    edited = tree.copy()
                    edited.set('expressions', [e.copy()])
                    edited.set('group', None)
                    results.append(edited.sql(dialect='sqlite'))
    if not any(tree.find_all(exp.AggFunc)):
        edited = tree.copy()
        edited.set("distinct", None if tree.args.get("distinct") else exp.Distinct())
        results.append(edited.sql(dialect="sqlite"))
    for count in tree.find_all(exp.Count):
        if not isinstance(count.this, exp.Star):
            edited = tree.copy()
            for node in edited.find_all(exp.Count):
                node.set('this', exp.Star())
            results.append(edited.sql(dialect='sqlite'))
        if isinstance(count.this, (exp.Star, exp.Distinct)):
            continue
        edited = tree.copy()
        for node in edited.find_all(exp.Count):
            if node.sql() == count.sql():
                node.set("this", exp.Distinct(expressions=[node.this.copy()]))
                break
        results.append(edited.sql(dialect="sqlite"))
    if tree.args.get("order"):
        edited = tree.copy()
        for item in edited.args["order"].expressions:
            item.set("desc", not item.args.get("desc", False))
        results.append(edited.sql(dialect="sqlite"))
    return results


def key(sql):
    try:
        return sqlglot.parse_one(sql, read='sqlite').sql(dialect='sqlite').lower()
    except (sqlglot.errors.SqlglotError, RecursionError):
        return sql.strip().lower()


def schema_repairs(sql, schema):
    try:
        tree = sqlglot.parse_one(sql, read='sqlite')
    except (sqlglot.errors.SqlglotError, RecursionError):
        return []
    tables = {t['name'].lower(): {c['name'].lower(): c['name'] for c in t['columns']} for t in schema}
    aliases = {t.alias_or_name.lower(): t.name.lower() for t in tree.find_all(exp.Table)}
    changed = False
    for scope in traverse_scope(tree):
        local = {alias.lower(): source.name.lower() for alias, source in scope.sources.items() if isinstance(source, exp.Table)}
        for col in scope.columns:
            if col.table:
                owner = local.get(col.table.lower())
                if owner is None or col.name.lower() in tables.get(owner, {}):
                    continue
            owners = [alias for alias, name in local.items() if col.name.lower() in tables.get(name, {})]
            if owners and (col.table or len(owners) > 1):
                col.set('table', exp.to_identifier(owners[0])); changed = True
    results = [tree.sql(dialect='sqlite')] if changed else []
    # Reconstruct invalid join edges using declared foreign keys, including bridges.
    graph = {}
    for t in schema:
        a = t['name'].lower()
        for fk in t['foreign_keys']:
            b = fk['table'].lower()
            graph.setdefault(a, []).append((b, fk['from'], fk['to']))
            graph.setdefault(b, []).append((a, fk['to'], fk['from']))
    def path(starts, target):
        queue = deque((s, []) for s in starts)
        seen = set(starts)
        while queue:
            current, steps = queue.popleft()
            if current == target:
                return steps
            for dest, src_col, dst_col in graph.get(current, []):
                if dest not in seen:
                    seen.add(dest); queue.append((dest, steps + [(current, dest, src_col, dst_col)]))
        return None
    rebuilt = tree.copy()
    for select in rebuilt.find_all(exp.Select):
        base = select.args.get('from')
        if not base or not isinstance(base.this, exp.Table):
            continue
        active = {base.this.name.lower(): base.this.alias_or_name}
        joins = []
        for join in select.args.get('joins', []):
            if not isinstance(join.this, exp.Table):
                joins.append(join); continue
            target = join.this.name.lower()
            steps = path(active, target)
            if not steps:
                joins.append(join); active[target] = join.this.alias_or_name; continue
            for src, dst, sc, dc in steps:
                alias = join.this.alias_or_name if dst == target else dst
                table = join.this.copy() if dst == target else exp.to_table(dst)
                on = exp.EQ(this=exp.column(sc, table=active[src]), expression=exp.column(dc, table=alias))
                new = exp.Join(this=table, on=on)
                if dst == target and join.args.get('side'):
                    new.set('side', join.args['side'])
                joins.append(new); active[dst] = alias
        select.set('joins', joins)
    results.append(rebuilt.sql(dialect='sqlite'))
    return results


def semantic_variants(sql, observation):
    try:
        tree = sqlglot.parse_one(sql, read='sqlite')
    except (sqlglot.errors.SqlglotError, RecursionError):
        return []
    if not isinstance(tree, exp.Select):
        return []
    q = observation['task']['question'].lower()
    out = []
    base = tree.args.get('from')
    threshold = re.search(r'at least (\d+|two|three|four|five)', q)
    if threshold and len(tree.expressions) == 1 and base and isinstance(base.this, exp.Table):
        expression = tree.expressions[0]
        if isinstance(expression, exp.Column) and tree.args.get('where') and tree.find(exp.Subquery):
            value = {'two':2,'three':3,'four':4,'five':5}.get(threshold[1],threshold[1])
            t=exp.select(expression.copy()).from_(base.this.copy()).group_by(expression.copy())
            t=t.having(exp.GTE(this=exp.Count(this=exp.Star()),expression=exp.Literal.number(value)))
            out.append(t.sql(dialect='sqlite'))
    if re.search(r'(older|higher|greater|larger|more) than (the )?average', q) and base and isinstance(base.this, exp.Table):
        average = next(tree.find_all(exp.Avg), None)
        column = average.this.copy() if average else exp.column('Age' if 'older' in q else 'value')
        sub=exp.select(exp.Avg(this=exp.column(column.name))).from_(base.this.name)
        t=tree.copy();t.set('where',exp.Where(this=exp.GT(this=column,expression=exp.Subquery(this=sub))))
        out.append(t.sql(dialect='sqlite'))
        try:
            t.set('expressions',[sqlglot.parse_one(h,read='sqlite') for h in observation['example']['columns']])
            out.append(t.sql(dialect='sqlite'))
        except sqlglot.errors.SqlglotError: pass
    if 'never' in q and tree.args.get('joins') and tree.args.get('where') and base and isinstance(base.this,exp.Table):
        joins=tree.args['joins']
        if len(joins)==1 and isinstance(joins[0].this,exp.Table):
            join=joins[0];condition=join.args.get('on');predicate=tree.args['where'].this
            if isinstance(condition,exp.EQ) and isinstance(predicate,exp.Not):
                a,b=condition.this,condition.expression
                if isinstance(a,exp.Column) and isinstance(b,exp.Column):
                    outer,inner=(a,b) if a.table.lower()==base.this.alias_or_name.lower() else (b,a)
                    sub=exp.select(inner.copy()).from_(join.this.copy()).where(predicate.this.copy())
                    t=tree.copy();t.set('joins',None)
                    t.set('where',exp.Where(this=exp.Not(this=exp.In(this=outer.copy(),query=exp.Subquery(this=sub)))))
                    out.append(t.sql(dialect='sqlite'))
    # Multiplicity is preserved unless uniqueness is explicitly requested.
    if tree.args.get('distinct') and not re.search(r'\bdistinct\b|\bdifferent\b|\bunique\b', q):
        t = tree.copy(); t.set('distinct', None); out.append(t.sql(dialect='sqlite'))
    # Repeated counts refer to groups, not to a single extremal row.
    for node in tree.find_all(exp.GTE):
        if 'at least' not in q and re.search(r'\b(two|three|four|[2-9])\b', q) and isinstance(node.this, exp.Count):
            t=tree.copy()
            for n in t.find_all(exp.GTE):
                if n.sql() == node.sql(): n.replace(exp.EQ(this=n.this.copy(), expression=n.expression.copy()))
            out.append(t.sql(dialect='sqlite'))
    for node in tree.find_all(exp.EQ):
        if isinstance(node.expression, exp.Literal) and node.expression.is_string:
            value = node.expression.this
            if value and '%' not in value:
                t = tree.copy()
                for n in t.find_all(exp.EQ):
                    if n.sql() == node.sql():
                        n.replace(exp.Like(this=n.this.copy(), expression=exp.Literal.string('%'+value+'%')))
                out.append(t.sql(dialect='sqlite'))
                if value.isdigit() and re.search(r'duration|minute|hour',q):
                    for pattern in [value+':%',value+'%']:
                        t=tree.copy()
                        for n in t.find_all(exp.EQ):
                            if n.sql()==node.sql(): n.replace(exp.Like(this=n.this.copy(),expression=exp.Literal.string(pattern)))
                        out.insert(0,t.sql(dialect='sqlite'))
    # Extra joins can multiply a global aggregate or an unconstrained listing.
    if not tree.args.get('where') and not tree.args.get('group') and tree.args.get('joins'):
        base=tree.args.get('from')
        if base and isinstance(base.this, exp.Table):
            alias=base.this.alias_or_name.lower()
            if all(not c.table or c.table.lower()==alias for e in tree.expressions for c in e.find_all(exp.Column)):
                t=tree.copy(); t.set('joins', None); out.append(t.sql(dialect='sqlite'))
    # Use the public output schema to construct the expected projection.
    headers=observation['example']['columns']
    try:
        t=tree.copy()
        expressions=[]
        for header in headers:
            e=sqlglot.parse_one(header, read='sqlite')
            if not isinstance(e, (exp.Column, exp.AggFunc)):
                break
            expressions.append(e)
        if len(expressions)==len(headers) and expressions:
            t.set('expressions', expressions); out.append(t.sql(dialect='sqlite'))
    except sqlglot.errors.SqlglotError:
        pass
    return out


def quality(sql, observation, schema):
    try:
        tree = sqlglot.parse_one(sql, read='sqlite')
    except (sqlglot.errors.SqlglotError, RecursionError):
        return -1000
    if not isinstance(tree, exp.Select):
        return 0
    headers = observation['example']['columns']
    score = -5 * abs(len(tree.expressions) - len(headers))
    normalize = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
    for expr, header in zip(tree.expressions, headers):
        name = expr.this.sql() if isinstance(expr, exp.Alias) else expr.sql()
        name = re.sub(r'\b\w+\.', '', name)
        score += 4 * SequenceMatcher(None, normalize(name), normalize(header)).ratio()
    q = observation['task']['question'].lower()
    if tree.args.get('distinct') and not re.search(r'\bdistinct\b|\bdifferent\b|\bunique\b', q):
        score -= 1
    for n in tree.find_all(exp.GTE):
        if isinstance(n.this, exp.Count) and 'at least' not in q and re.search(r'\b(two|three|four|[2-9])\b', q):
            score -= 2
    aliases = {t.alias_or_name.lower(): t.name.lower() for t in tree.find_all(exp.Table)}
    edges = set()
    for t in schema or []:
        for fk in t['foreign_keys']:
            a=(t['name'].lower(),fk['from'].lower()); b=(fk['table'].lower(),fk['to'].lower())
            edges.add((a,b)); edges.add((b,a))
    for join in tree.find_all(exp.Join):
        for n in (join.args.get('on') or exp.Null()).find_all(exp.EQ):
            if isinstance(n.this, exp.Column) and isinstance(n.expression, exp.Column):
                a,b=n.this,n.expression
                if a.sql().lower()==b.sql().lower(): score-=15
                pair=((aliases.get(a.table.lower(),a.table.lower()),a.name.lower()),(aliases.get(b.table.lower(),b.table.lower()),b.name.lower()))
                if edges and pair not in edges: score-=3
    return score


def features(sql, observation, schema, hints):
    """Fixed, task-independent features for a training-split linear ranker."""
    try:
        tree = sqlglot.parse_one(sql, read='sqlite')
    except (sqlglot.errors.SqlglotError, RecursionError):
        return [0.0] * 22
    q = observation['task']['question'].lower()
    words = set(re.findall(r'[a-z]+', q))
    tokens = set(re.findall(r'[a-z]+', sql.lower()))
    canonical = key(sql)
    original = canonical == key(observation['issue_sql'])
    is_hint = canonical in {key(s) for s in hints or []}
    joins = list(tree.find_all(exp.Join))
    aggs = list(tree.find_all(exp.AggFunc))
    count = bool(list(tree.find_all(exp.Count)))
    distinct = bool(tree.args.get('distinct'))
    group = bool(tree.args.get('group'))
    order = bool(tree.args.get('order'))
    limit = bool(tree.args.get('limit'))
    cols = list(tree.find_all(exp.Column))
    unqualified = sum(not c.table for c in cols) / max(1, len(cols))
    match_width = len(tree.expressions) == len(observation['example']['columns']) if isinstance(tree, exp.Select) else False
    return [1., quality(sql, observation, schema)/10, float(original), float(is_hint),
            len(joins)/5, len(aggs)/5, float(distinct), float(group), float(order), float(limit),
            len(list(tree.find_all(exp.Select)))/5, unqualified,
            len(tokens & words)/max(1, len(words)), float(match_width),
            float(distinct and bool(words & {'distinct','different','unique'})),
            float(count and bool(words & {'many','number','count'})),
            float(order and bool(words & {'order','ordered','alphabetical','sorted'})),
            float(limit and bool(words & {'most','least','maximum','minimum','highest','lowest','top','first','last'})),
            float(group and bool(words & {'each','every','per'})),
            float(bool(tree.args.get('where'))),
            len(list(tree.find_all(exp.Like)))/3,
            float(bool(aggs) and bool(words & {'average','total','sum','maximum','minimum','many','number'}))]


class RepairAgent:
    def __init__(self, config):
        self.config = config
        path = Path(__file__).with_name('ranker.json')
        self.weights = json.loads(path.read_text(encoding='utf-8'))['weights'] if config['use_ranker'] and path.exists() else None

    def act(self, observation):
        history = observation["history"]
        original = observation["issue_sql"]
        if not history:
            return {"tool": "check", "sql": original}
        checks = [event for event in history if event["tool"] == "check"]
        matched = [event for event in checks
                   if event["observation"].get("matches_example")]
        hints = next((event["observation"] for event in history
                      if event["tool"] == "repair_hint"), None)
        if self.config["use_hints"] and hints is None and observation["remaining"]:
            return {"tool": "repair_hint"}
        hint_checks = [event for event in matched if event["sql"] in (hints or [])]
        schema = next((e['observation'] for e in history if e['tool'] == 'schema'), None)
        if schema is None and observation['remaining'] and hints is not None:
            return {'tool': 'schema'}
        candidates = []
        bases = list(dict.fromkeys(list(hints or []) or [original]))
        preferred = []
        for sql in bases:
            preferred.extend(semantic_variants(sql, observation))
        candidates.extend(hints or [])
        if schema:
            for event in checks[1:]:
                candidates.extend(schema_repairs(event['sql'], schema))
        candidates.extend(preferred)
        if self.config["use_variants"]:
            for sql in bases:
                candidates.extend(variants(sql, observation))
        tried = {key(event["sql"]) for event in checks}
        preferred_keys = {key(s) for s in preferred}
        preferred_matches = [e for e in matched if key(e['sql']) in preferred_keys]
        if not self.config['collect'] and self.weights is None and preferred_matches and hint_checks:
            return {'tool': 'submit', 'sql': preferred_matches[0]['sql']}
        if not self.config['collect'] and self.weights is None and hint_checks and all(key(s) in tried for s in preferred[:2]):
            return {'tool': 'submit', 'sql': hint_checks[0]['sql']}
        if observation["remaining"]:
            for sql in dict.fromkeys(candidates):
                if isinstance(sql, str) and key(sql) not in tried and len(sql.encode()) <= 16000:
                    return {"tool": "check", "sql": sql}
        if matched:
            if self.weights:
                return {'tool': 'submit', 'sql': max(matched, key=lambda e: sum(w*x for w,x in zip(self.weights, features(e['sql'], observation, schema, hints))))['sql']}
            return {"tool": "submit", "sql": max([e for e in matched if e["sql"] != original] or matched, key=lambda e: quality(e["sql"], observation, schema))["sql"]}
        # No public match: prefer an executable hint, then an executable draft.
        successful = {event["sql"] for event in checks if event["observation"]["ok"]}
        fallback = next((sql for sql in (hints or []) if sql in successful), None)
        if fallback is None:
            fallback = next((event["sql"] for event in checks
                             if event["observation"]["ok"]), original)
        return {"tool": "submit", "sql": fallback}


def create_agent(checkpoint=None):
    config = {"use_hints": True, "use_variants": True, "collect": False, "use_ranker": False}
    if checkpoint is not None:
        supplied = json.loads(Path(checkpoint).read_text(encoding="utf-8"))
        if set(supplied) - set(config) or any(type(v) is not bool for v in supplied.values()):
            raise ValueError("Configuration fields must be supported booleans")
        config.update(supplied)
    return RepairAgent(config)
