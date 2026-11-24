"""Schema-based repair operators. No benchmark files or task identifiers are used.

Entity/set repair and value discovery were inspired by the public submission
reviewed in issue 397. This module is an independent implementation.
"""
import itertools
import re
from difflib import SequenceMatcher
from functools import lru_cache

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope


@lru_cache(maxsize=1024)
def _parsed(sql):
    try:
        trees = sqlglot.parse(sql, read='sqlite')
        if len(trees) == 1 and isinstance(trees[0], (exp.Select, exp.SetOperation)):
            return trees[0]
    except (sqlglot.errors.SqlglotError, RecursionError):
        pass
    return None


def parse(sql):
    tree = _parsed(sql)
    return tree.copy() if tree is not None else None


def key(sql):
    tree = parse(sql)
    if isinstance(tree, exp.Select) and len(list(tree.find_all(exp.Select))) == 1:
        sources = list(tree.find_all(exp.Table))
        if len({t.name.lower() for t in sources}) == len(sources):
            replacements = {t.alias_or_name.lower(): t.name.lower() for t in sources}
            for column in tree.find_all(exp.Column):
                if column.table.lower() in replacements:
                    column.set('table', exp.to_identifier(replacements[column.table.lower()]))
            for table in sources:
                table.set('alias', None)
    return tree.sql(dialect='sqlite', normalize=True) if tree is not None else sql.strip()


def terms(text):
    parts = re.findall(r'[a-z]+', re.sub(r'([a-z])([A-Z])', r'\1 \2', text).lower())
    normalized = set()
    for p in parts:
        if p.endswith('ies') and len(p) > 4:
            p = p[:-3] + 'y'
        elif p.endswith('sses'):
            p = p[:-2]
        elif p.endswith('s') and not p.endswith(('ss', 'us')) and len(p) > 3:
            p = p[:-1]
        normalized.add({'employee': 'person', 'people': 'person'}.get(p, p))
    return normalized


def bindings(select, schema):
    catalog = {t['name'].lower(): t for t in schema}
    source = select.args.get('from')
    tables = ([source.this] if source else []) + [j.this for j in select.args.get('joins', [])]
    return {t.alias_or_name.lower(): catalog[t.name.lower()] for t in tables
            if isinstance(t, exp.Table) and t.name.lower() in catalog}


def _qualify(tree, schema):
    tree = tree.copy()
    catalog = {t['name'].lower(): t for t in schema}
    for scope in traverse_scope(tree):
        owners = {alias.lower(): catalog[source.name.lower()] for alias, source in scope.sources.items()
                  if isinstance(source, exp.Table) and source.name.lower() in catalog}
        for column in scope.columns:
            choices = [alias for alias, table in owners.items()
                       if any(c['name'].lower() == column.name.lower() for c in table['columns'])]
            if column.table.lower() in choices or (column.table and column.table.lower() not in owners):
                continue
            if not column.table and len(choices) > 1:
                evidence = [c.table.lower() for c in scope.columns if c.name.lower() == column.name.lower() and c.table.lower() in choices]
                if evidence:
                    choices = [evidence[0]]
                else:
                    source = scope.expression.args.get('from')
                    if source and isinstance(source.this, exp.Table) and source.this.alias_or_name.lower() in choices:
                        choices = [source.this.alias_or_name.lower()]
            if len(choices) == 1:
                column.set('table', exp.to_identifier(choices[0]))
    return tree


def qualify(tree, schema):
    try:
        return _qualify(tree, schema)
    except (sqlglot.errors.SqlglotError, RecursionError):
        # Broken drafts can contain duplicate aliases. Keep their error for
        # the official checker rather than letting a repair operator crash.
        return tree.copy()


def projections(tree, schema, headers):
    if not isinstance(tree, exp.Select):
        return []
    target = []
    tables = bindings(tree, schema)
    used = set()
    aliases = {e.alias.lower(): e.this for e in tree.expressions if isinstance(e, exp.Alias)}
    for index, header in enumerate(headers):
        expression = parse('SELECT ' + header)
        if expression is None or len(expression.expressions) != 1:
            return []
        expression = expression.expressions[0]
        if not isinstance(expression, (exp.Column, exp.AggFunc)):
            return []
        old = tree.expressions[index] if index < len(tree.expressions) else None
        old = old.this if isinstance(old, exp.Alias) else old
        if isinstance(expression, exp.Column) and isinstance(old, exp.Column) and old.name.lower() == expression.name.lower() and old.table.lower() in tables:
            if old.table.lower() not in used or len(headers) == 1:
                target.append(old.copy())
                used.add(old.table.lower())
                continue
        for column in expression.find_all(exp.Column):
            owners = [a for a, t in tables.items() if any(c['name'].lower() == column.name.lower() for c in t['columns'])]
            if not owners:
                return []
            owner = next((a for a in owners if a not in used), owners[0])
            column.set('table', exp.to_identifier(owner))
            used.add(owner)
        target.append(expression)
    result = tree.copy()
    result.set('expressions', target)
    order = result.args.get('order')
    if order:
        for item in order.expressions:
            if isinstance(item.this, exp.Column) and not item.this.table and item.this.name.lower() in aliases:
                item.set('this', aliases[item.this.name.lower()].copy())
    if any(e.find(exp.AggFunc) for e in target) and not tree.find(exp.AggFunc):
        grouped = [e.copy() for e in target if not e.find(exp.AggFunc)]
        if grouped:
            result.set('group', exp.Group(expressions=grouped))
        if order:
            for item in order.expressions:
                if isinstance(item.this, exp.Column):
                    replacement = next((a for e in target for a in e.find_all(exp.AggFunc) if isinstance(a.this, exp.Column) and a.this.name.lower() == item.this.name.lower()), None)
                    if replacement is not None:
                        item.set('this', replacement.copy())
    return [result]


def root_expressions(tree, root):
    return all(not c.table or c.table.lower() == root for expression in tree.expressions for c in expression.find_all(exp.Column))


def entity_edits(tree, schema, question):
    if not isinstance(tree, exp.Select):
        return []
    out = []
    q = question.lower()
    tables = bindings(tree, schema)
    source = tree.args.get('from')
    if not source or not isinstance(source.this, exp.Table):
        return out
    root = source.this.alias_or_name.lower()
    metadata = tables.get(root)
    joins = tree.args.get('joins', [])
    flat = len(list(tree.find_all(exp.Select))) == 1
    # Preserve the identity of counted entities when detail rows multiply them.
    noun = re.search(r'\b(?:how many|number of)\s+(?:(?:distinct|different|unique)\s+)?([a-z_]+)', q)
    if metadata and joins and flat and noun and terms(noun[1]) <= terms(metadata['name']):
        keys = [c['name'] for c in metadata['columns'] if c.get('pk')]
        if keys and not tree.args.get('group') and len(tree.expressions) == 1 and isinstance(tree.expressions[0], exp.Count):
            inner = tree.copy()
            inner.set('expressions', [exp.column(k, table=root) for k in keys])
            inner.set('distinct', exp.Distinct())
            for clause in ('order', 'limit', 'offset'):
                inner.set(clause, None)
            result = exp.select(exp.Count(this=exp.Star())).from_(inner.subquery('_entities'))
            related = any(terms(j.this.name) <= terms(q) for j in joins if isinstance(j.this, exp.Table))
            constrained = tree.args.get('where') and any(c.table.lower() != root for c in tree.args['where'].find_all(exp.Column))
            out.append((result, 'entity_count', 3.0 if related or constrained else 2.0))
    # Remove only leaves that no output, condition, or other JOIN needs.
    if flat and joins:
        changed = tree.copy()
        kept = list(changed.args['joins'])
        external = [e for k, v in tree.args.items() if k not in ('joins', 'from')
                    for e in (v if isinstance(v, list) else [v]) if isinstance(e, exp.Expression)]
        required = {c.table.lower() for e in external for c in e.find_all(exp.Column) if c.table}
        for join in reversed(list(kept)):
            if not isinstance(join.this, exp.Table):
                continue
            alias = join.this.alias_or_name.lower()
            named = bool(terms(join.this.name)) and terms(join.this.name) <= terms(q)
            dependent = any(c.table.lower() == alias for other in kept if other is not join for c in other.find_all(exp.Column))
            if alias not in required and not named and not dependent:
                kept.remove(join)
        if len(kept) != len(joins):
            changed.set('joins', kept)
            out.append((changed, 'unused_join', 2.6))
    if flat and joins and root_expressions(tree, root) and tree.args.get('where'):
        condition = tree.args['where'].this
        atoms = list(condition.flatten()) if isinstance(condition, exp.And) else [condition]
        negative = next((a for a in atoms if isinstance(a, exp.NEQ) or isinstance(a, exp.Not) and isinstance(a.this, (exp.EQ, exp.Like, exp.In))), None)
        if negative is not None and re.search(r'\bnever\b|\bwithout\b|\bnot have\b|\bdo not\b|\bnot (?:living|located|based)\b', q):
            positive = exp.EQ(this=negative.this.copy(), expression=negative.expression.copy()) if isinstance(negative, exp.NEQ) else negative.this.copy()
            root_filters = [a.copy() for a in atoms if a is not negative and all(c.table.lower() == root for c in a.find_all(exp.Column))]
            universe = tree.copy()
            universe.set('joins', [])
            universe.set('where', exp.Where(this=exp.and_(*root_filters)) if root_filters else None)
            excluded = tree.copy()
            excluded.set('where', exp.Where(this=exp.and_(*[a.copy() for a in atoms if a is not negative], positive)))
            for branch in (universe, excluded):
                for clause in ('group', 'having', 'order', 'limit', 'distinct'):
                    branch.set(clause, None)
            if not any(tree.find_all(exp.AggFunc)):
                out.append((exp.Except(this=universe, expression=excluded, distinct=True), 'entity_absence', 4.0))
    # A nullable NOT IN input must not suppress every outer entity.
    if re.search(r'\bnever\b|\bwithout\b|\bnot in\b|\bnot have\b|\bdo not\b', q):
        for position, node in enumerate(tree.find_all(exp.Not)):
            if not isinstance(node.this, exp.In) or not isinstance(node.this.this, exp.Column):
                continue
            outer = node.this.this
            query = node.this.args.get('query')
            inner = query.this if isinstance(query, exp.Subquery) else query
            if not isinstance(inner, exp.Select) or len(inner.expressions) != 1 or not isinstance(inner.expressions[0], exp.Column):
                continue
            if any(inner.args.get(c) for c in ('group', 'having', 'limit', 'offset')) or outer.table.lower() in bindings(inner, schema):
                continue
            owner = tables.get(outer.table.lower())
            if not owner or not any(c['name'].lower() == outer.name.lower() and (c.get('pk') or c.get('not_null')) for c in owner['columns']):
                continue
            inner = inner.copy()
            where = inner.args.get('where')
            relation = exp.EQ(this=inner.expressions[0].copy(), expression=outer.copy())
            inner.set('where', exp.Where(this=exp.and_(where.this.copy(), relation) if where else relation))
            inner.set('expressions', [exp.Literal.number(1)])
            inner.set('order', None)
            result = tree.copy()
            list(result.find_all(exp.Not))[position].replace(exp.Not(this=exp.Exists(this=inner)))
            out.append((result, 'null_safe_absence', 3.5))
    # Translate a conjunction about an entity across two records into a set.
    if flat and 'both' in terms(q) and not any(e.find(exp.AggFunc) for e in tree.expressions):
        for position, node in enumerate(tree.find_all(exp.In)):
            if len(node.expressions) == 2 and all(isinstance(v, exp.Literal) for v in node.expressions):
                branches = []
                for value in node.expressions:
                    branch = tree.copy()
                    target = list(branch.find_all(exp.In))[position]
                    target.replace(exp.EQ(this=target.this.copy(), expression=value.copy()))
                    for clause in ('order', 'limit', 'group', 'having'):
                        branch.set(clause, None)
                    branches.append(branch)
                out.append((exp.Intersect(this=branches[0], expression=branches[1], distinct=True), 'both_records', 3.5))
        where = tree.args.get('where')
        atoms = list(where.this.flatten()) if where and isinstance(where.this, (exp.And, exp.Or)) else []
        for a, b in itertools.combinations(atoms, 2):
            if isinstance(a, exp.EQ) and isinstance(b, exp.EQ) and isinstance(a.this, exp.Column) and a.this == b.this and a.expression != b.expression:
                branches = []
                for predicate in (a, b):
                    branch = tree.copy()
                    others = [v.copy() for v in atoms if v is not a and v is not b]
                    branch.set('where', exp.Where(this=exp.and_(*others, predicate.copy())))
                    branch.set('order', None)
                    branch.set('limit', None)
                    branches.append(branch)
                out.append((exp.Intersect(this=branches[0], expression=branches[1], distinct=True), 'both_records', 3.5))
    if flat and metadata and tree.args.get('group') and tree.args.get('limit') and tree.find(exp.Count):
        grouped = tree.args['group'].expressions
        owner = tables.get(grouped[0].table.lower()) if len(grouped) == 1 and isinstance(grouped[0], exp.Column) else None
        keys = [c['name'] for c in (owner or {}).get('columns', []) if c.get('pk')]
        related = owner and bool(terms(grouped[0].name) & terms(owner['name']) - {'name'})
        if keys and related and len(grouped) == 1 and 'name' in grouped[0].name.lower():
            result = tree.copy()
            result.set('group', exp.Group(expressions=[exp.column(k, table=grouped[0].table) for k in keys]))
            out.append((result, 'identity_group', 2.5))
    if flat and len(joins) == 1 and tree.find(exp.Avg) and root_expressions(tree, root):
        join = joins[0]
        on = join.args.get('on')
        if isinstance(join.this, exp.Table) and isinstance(on, exp.EQ):
            child = join.this.alias_or_name.lower()
            where = tree.args.get('where')
            atoms = list(where.this.flatten()) if where and isinstance(where.this, exp.And) else [where.this] if where else []
            root_filters = [a.copy() for a in atoms if all(c.table.lower() == root for c in a.find_all(exp.Column))]
            child_filters = [a.copy() for a in atoms if any(c.table.lower() == child for c in a.find_all(exp.Column))]
            grounded_filter = any((terms(c.name) - {'id', 'code', 'name'}) & terms(q)
                                  for a in child_filters for c in a.find_all(exp.Column) if c.table.lower() == child)
            if grounded_filter or terms(join.this.name) <= terms(q):
                inner = exp.select(exp.Literal.number(1)).from_(join.this.copy()).where(exp.and_(on.copy(), *child_filters))
                result = tree.copy()
                result.set('joins', [])
                result.set('where', exp.Where(this=exp.and_(*root_filters, exp.Exists(this=inner))))
                out.append((result, 'population_average', 3.2))
    if flat and metadata and re.search(r'\bnever\b|\bnot have\b', q) and root_expressions(tree, root) and tree.args.get('where') and not any(tree.find_all(exp.AggFunc)):
        atoms = list(tree.args['where'].this.flatten()) if isinstance(tree.args['where'].this, exp.And) else [tree.args['where'].this]
        positive = [a for a in atoms if isinstance(a, exp.EQ) and isinstance(a.this, exp.Column) and isinstance(a.expression, exp.Literal) and terms(str(a.expression.this)) <= terms(q)]
        if positive and all(a.this.table.lower() == root for a in positive):
            universe = tree.copy()
            universe.set('where', None)
            universe.set('joins', [])
            excluded = universe.copy()
            excluded.set('where', exp.Where(this=exp.and_(*[a.copy() for a in positive])))
            for branch in (universe, excluded):
                for name in ('group', 'having', 'order', 'limit', 'distinct'):
                    branch.set(name, None)
            out.append((exp.Except(this=universe, expression=excluded, distinct=True), 'attribute_absence', 4.6))
    return out


def predicate_edits(tree, schema, question):
    if not isinstance(tree, exp.Select):
        return []
    out = []
    q = question.lower()
    if not re.search(r'\bcontain\w*\b|\bsubstring\b|\blike\b|\bstart\w*\b|\bend\w*\b|\bmention\w*\b', q):
        for index, node in enumerate(tree.find_all(exp.Like)):
            if isinstance(node.expression, exp.Literal) and node.expression.is_string and node.expression.this.startswith('%') and node.expression.this.endswith('%'):
                result = tree.copy()
                target = list(result.find_all(exp.Like))[index]
                target.replace(exp.EQ(this=target.this.copy(), expression=exp.Literal.string(target.expression.this.strip('%'))))
                out.append((result, 'exact_condition', 1.8))
    where = tree.args.get('where')
    tables = bindings(tree, schema)
    # Use the requested measurement rather than a similarly named metric.
    for index, aggregate in enumerate(tree.find_all(exp.Avg, exp.Sum, exp.Min, exp.Max)):
        columns = list(aggregate.find_all(exp.Column))
        if len(columns) != 1:
            continue
        column = columns[0]
        owner = tables.get(column.table.lower())
        if owner is None:
            continue
        old_terms = terms(column.name)
        overlap = len(old_terms & terms(q))
        choices = [c for c in owner['columns'] if len(terms(c['name']) & terms(q)) > overlap and
                   len(terms(c['name']) & old_terms) >= 1]
        if len(choices) == 1:
            result = tree.copy()
            for target in result.find_all(type(aggregate)):
                for reference in target.find_all(exp.Column):
                    if reference.name.lower() == column.name.lower() and reference.table.lower() == column.table.lower():
                        reference.set('this', exp.to_identifier(choices[0]['name']))
            out.append((result, 'requested_measurement', 4.1))
    if where and isinstance(where.this, exp.And):
        atoms = list(where.this.flatten())
        rejected = []
        for atom in atoms:
            if isinstance(atom, exp.EQ) and isinstance(atom.expression, exp.Literal) and atom.expression.is_string:
                literal_words = terms(atom.expression.this)
                if len(literal_words) >= 2 and not literal_words <= terms(q):
                    rejected.append(atom)
                    result = tree.copy()
                    result.set('where', exp.Where(this=exp.and_(*[a.copy() for a in atoms if a is not atom])))
                    out.append((result, 'unsupported_condition', 2.0))
                if isinstance(atom.this, exp.Column) and re.search(r'description|type|detail', atom.this.name, re.I):
                    duplicate = any(other is not atom and isinstance(other, exp.EQ) and isinstance(other.this, exp.Column) and
                                    other.expression == atom.expression and re.search(r'name$', other.this.name, re.I)
                                    for other in atoms)
                    if duplicate and not re.search(r'\bdescription\b|\btype\b|\bdetail\b', q):
                        rejected.append(atom)
        if rejected and len(rejected) < len(atoms):
            result = tree.copy()
            result.set('where', exp.Where(this=exp.and_(*[a.copy() for a in atoms if not any(a is r for r in rejected)])))
            out.append((result, 'minimal_conditions', 3.0))
    # A named root entity is identified by its name, not a joined type label.
    source = tree.args.get('from')
    if source and isinstance(source.this, exp.Table):
        root = source.this.alias_or_name.lower()
        metadata = tables.get(root)
        if metadata and terms(metadata['name']) <= terms(q):
            names = [c['name'] for c in metadata['columns'] if re.search(r'name$', c['name'], re.I)]
            if len(names) == 1:
                for index, node in enumerate(tree.find_all(exp.EQ)):
                    if isinstance(node.this, exp.Column) and re.search(r'name|description|type', node.this.name, re.I) and node.this.table.lower() != root and isinstance(node.expression, exp.Literal) and node.expression.is_string and terms(node.expression.this) <= terms(q):
                        result = tree.copy()
                        list(result.find_all(exp.EQ))[index].set('this', exp.column(names[0], table=root))
                        out.append((result, 'entity_name_condition', 2.8))
    per = re.search(r'\b(?:each|per)\s+([a-z_]+)', q)
    if per and tree.args.get('group') and tree.find(exp.AggFunc):
        options = []
        tail = q[per.start():]
        for alias, metadata in tables.items():
            for c in metadata['columns']:
                meaning = terms(c['name']) - {'id', 'identifier', 'code'}
                if meaning and terms(per[1]) <= meaning and meaning <= terms(tail):
                    options.append((len(meaning), alias, c['name']))
        if options:
            length = max(o[0] for o in options)
            for _, alias, name in [o for o in options if o[0] == length][:2]:
                result = tree.copy()
                desired = exp.column(name, table=alias)
                result.set('group', exp.Group(expressions=[desired]))
                out.append((result, 'requested_group', 2.2))
                old_group = tree.args['group'].expressions
                if len(old_group) == 1 and isinstance(old_group[0], exp.Column) and old_group[0] != desired and terms(old_group[0].name) - {'id'} <= terms(q):
                    result = tree.copy()
                    result.set('group', exp.Group(expressions=[old_group[0].copy(), desired.copy()]))
                    result.set('distinct', None)
                    out.append((result, 'requested_group_pair', 2.4))
    if tree.find(exp.Count) and not re.search(r'\bincluding\b|\beven\b|\bzero\b|\bwithout\b', q):
        result = tree.copy()
        changed = False
        for join in result.args.get('joins', []):
            if join.args.get('side') == 'LEFT':
                join.set('side', None)
                changed = True
        if changed:
            out.append((result, 'matched_groups', 1.5))
    if re.search(r'\balphabetical\b', q) and tree.args.get('order'):
        for metadata in schema:
            if not terms(metadata['name']) <= terms(q):
                continue
            for c in metadata['columns']:
                if re.search(r'text|char', c['type'], re.I) and terms(c['name']) & terms(q):
                    result = tree.copy()
                    alias = next((a for a, t in tables.items() if t['name'] == metadata['name']), metadata['name'])
                    if alias not in tables:
                        result.set('joins', list(result.args.get('joins', [])) + [exp.Join(this=exp.to_table(metadata['name']))])
                    result.set('order', exp.Order(expressions=[exp.Ordered(this=exp.column(c['name'], table=alias), desc=False)]))
                    out.append((result, 'alphabetical_attribute', 2.8))
    return out


def probe(seeds, schema):
    seeds = [select for tree in seeds if tree is not None for select in tree.find_all(exp.Select)]
    fields = []
    for tree in seeds:
        if not isinstance(tree, exp.Select):
            continue
        tables = bindings(tree, schema)
        for predicate in tree.find_all(exp.EQ, exp.NEQ, exp.Like):
            if not isinstance(predicate.this, exp.Column) or not isinstance(predicate.expression, exp.Literal) or not predicate.expression.is_string:
                continue
            column = predicate.this
            for alias, table in tables.items():
                if column.table and column.table.lower() != alias:
                    continue
                columns = sorted(table['columns'], key=lambda c: c['name'].lower() != column.name.lower())
                for c in columns:
                    if c['name'].lower() == column.name.lower() or re.search(r'text|char', c['type'], re.I):
                        pair = (table['name'], c['name'])
                        if pair not in fields:
                            fields.append(pair)
        if fields:
            break
    branches = []
    # Include other participating text fields after predicate-owner fields.
    if fields:
        for tree in seeds:
            if isinstance(tree, exp.Select):
                for table in bindings(tree, schema).values():
                    for c in table['columns']:
                        pair = (table['name'], c['name'])
                        if re.search(r'text|char', c['type'], re.I) and pair not in fields:
                            fields.append(pair)
    for table, column in fields[:6]:
        t = exp.to_identifier(table, quoted=True).sql(dialect='sqlite')
        c = exp.to_identifier(column, quoted=True).sql(dialect='sqlite')
        label = exp.Literal.string(table + '.' + column).sql(dialect='sqlite')
        branches.append(f'SELECT {label} AS field, GROUP_CONCAT(v, CHAR(31)) AS samples FROM '
                        f'(SELECT DISTINCT CAST({c} AS TEXT) AS v FROM {t} WHERE {c} IS NOT NULL ORDER BY 1 LIMIT 32)')
    return ' UNION ALL '.join(branches) if branches else None


def discovered(history):
    values = {}
    for event in history:
        feedback = event.get('observation', {})
        if event['tool'] == 'execute' and feedback.get('ok') and feedback.get('columns') == ['field', 'samples']:
            for field, samples in feedback['rows']:
                if samples is not None:
                    values[field.casefold()] = str(samples).split(chr(31))
    return values


def literal_edits(tree, schema, history, question):
    values = discovered(history)
    if not values or not isinstance(tree, (exp.Select, exp.SetOperation)):
        return []
    out = []
    for position, predicate in enumerate(tree.find_all(exp.EQ, exp.NEQ, exp.Like)):
        if not isinstance(predicate.this, exp.Column) or not isinstance(predicate.expression, exp.Literal) or not predicate.expression.is_string:
            continue
        column = predicate.this
        text = predicate.expression.this.strip('%')
        select = predicate.find_ancestor(exp.Select)
        tables = bindings(select, schema) if select is not None else {}
        for alias, table in tables.items():
            cross_table = bool(column.table and column.table.lower() != alias)
            if cross_table:
                owner = tables.get(column.table.lower())
                old_values = values.get((owner['name'] + '.' + column.name).casefold(), []) if owner else []
                if any(v.casefold() == text.casefold() for v in old_values) or not terms(table['name']) & terms(question):
                    continue
            options = []
            for c in table['columns']:
                observed = values.get((table['name'] + '.' + c['name']).casefold(), [])
                exact = [v for v in observed if v.casefold() == text.casefold()]
                contained = [v for v in observed if len(text) >= 3 and re.search(r'(?<!\w)' + re.escape(text) + r'(?!\w)', v, re.I) and terms(v) <= terms(question)]
                near = [v for v in observed if len(text) >= 4 and SequenceMatcher(None, text.casefold(), v.casefold()).ratio() >= .82]
                choices = exact or contained or near
                if len(choices) == 1:
                    named = bool(terms(c['name']) & terms(question))
                    options.append((named, bool(exact), c['name'].lower() == column.name.lower(), c['name'], choices[0]))
            if options:
                options.sort(reverse=True)
                if len(options) == 1 or options[0][:3] != options[1][:3]:
                    _, _, _, name, value = options[0]
                    result = tree.copy()
                    target = list(result.find_all(exp.EQ, exp.NEQ, exp.Like))[position]
                    target.set('this', exp.column(name, table=alias))
                    if isinstance(target, exp.Like):
                        value = ('%' if predicate.expression.this.startswith('%') else '') + value + ('%' if predicate.expression.this.endswith('%') else '')
                    target.set('expression', exp.Literal.string(value))
                    if key(result.sql()) != key(tree.sql()):
                        out.append((result, 'observed_value', 3.0 if cross_table else 4.5))
            if re.search(r'name|fname|lname', column.name, re.I) and len(text.split()) >= 2:
                components = []
                for c in table['columns']:
                    if not re.search(r'name|fname|lname|given|family', c['name'], re.I):
                        continue
                    observed = values.get((table['name'] + '.' + c['name']).casefold(), [])
                    matches = [v for v in observed if len(v) >= 2 and re.search(r'(?<!\w)' + re.escape(v) + r'(?!\w)', text, re.I)]
                    if len(matches) == 1:
                        components.append((c['name'], matches[0]))
                if len(components) == 2 and terms(' '.join(v for _, v in components)) == terms(text):
                    result = tree.copy()
                    target = list(result.find_all(exp.EQ, exp.NEQ, exp.Like))[position]
                    target.replace(exp.and_(*[exp.EQ(this=exp.column(c, table=alias), expression=exp.Literal.string(v)) for c, v in components]))
                    out.append((result, 'split_person_name', 4.5))
    return out


def consistent_literals(tree, schema, history):
    """Apply a discovered categorical spelling consistently across all branches."""
    values = discovered(history)
    if not values or tree is None:
        return None
    result = tree.copy()
    changed = False
    for predicate in result.find_all(exp.EQ, exp.NEQ, exp.Like, exp.In):
        if not isinstance(predicate.this, exp.Column):
            continue
        column = predicate.this
        scope = predicate.find_ancestor(exp.Select)
        owners = bindings(scope, schema) if scope is not None else {}
        owner = owners.get(column.table.lower())
        if owner is None:
            continue
        samples = values.get((owner['name'] + '.' + column.name).casefold(), [])
        if not samples:
            continue
        literals = predicate.expressions if isinstance(predicate, exp.In) else [predicate.expression]
        for literal in literals:
            if not isinstance(literal, exp.Literal) or not literal.is_string:
                continue
            word = literal.this.strip('%') if isinstance(predicate, exp.Like) else literal.this
            exact = [v for v in samples if v.casefold() == word.casefold()]
            replacement = exact[0] if len(exact) == 1 else None
            # Infer a spelling convention, not a new condition or example answer.
            if replacement is None and word.islower() and re.fullmatch(r'[a-z]+(?: [a-z]+)*', word) and all(len(v) >= 2 and v.istitle() for v in samples):
                replacement = word.title()
            if replacement is not None and replacement != word:
                value = replacement
                if isinstance(predicate, exp.Like):
                    value = ('%' if literal.this.startswith('%') else '') + value + ('%' if literal.this.endswith('%') else '')
                literal.set('this', value)
                changed = True
    return result if changed else None
