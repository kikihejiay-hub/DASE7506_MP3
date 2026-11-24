"""Semantic regression tests on synthetic databases, never benchmark instances."""
import importlib.util
from pathlib import Path
import sqlite3
import unittest

spec = importlib.util.spec_from_file_location('new_relational_rules_tests', Path(__file__).resolve().parent.parent / 'submission/relational_rules.py')
rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rules)


class EntityRules(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.db.executescript("CREATE TABLE people(id INTEGER PRIMARY KEY,name TEXT); CREATE TABLE visits(person_id INTEGER,color TEXT); INSERT INTO people VALUES(1,'A'),(2,'B'),(3,'C'); INSERT INTO visits VALUES(1,'red'),(1,'blue'),(2,'red'),(NULL,'green');")
        self.schema = [dict(name='people', columns=[dict(name='id', type='INTEGER', pk=1), dict(name='name', type='TEXT', pk=0)], foreign_keys=[]),
                       dict(name='visits', columns=[dict(name='person_id', type='INTEGER', pk=0), dict(name='color', type='TEXT', pk=0)], foreign_keys=[dict(table='people', **{'from':'person_id','to':'id'})])]

    def tearDown(self):
        self.db.close()

    def repaired(self, sql, question, reason):
        edits = rules.entity_edits(rules.qualify(rules.parse(sql), self.schema), self.schema, question)
        query = next(tree.sql(dialect='sqlite') for tree, tag, confidence in edits if tag == reason)
        return self.db.execute(query).fetchall()

    def test_entity_count_ignores_duplicate_detail_rows(self):
        self.assertEqual(self.repaired('SELECT COUNT(*) FROM people p JOIN visits v ON p.id=v.person_id', 'How many people have visits?', 'entity_count'), [(2,)])

    def test_average_is_not_weighted_by_child_count(self):
        self.assertEqual(self.repaired('SELECT AVG(p.id) FROM people p JOIN visits v ON p.id=v.person_id', 'What is the average id of people with visits?', 'population_average'), [(1.5,)])

    def test_both_with_a_having_clause_still_creates_intersection(self):
        self.assertEqual(self.repaired("SELECT p.id FROM people p JOIN visits v ON p.id=v.person_id WHERE v.color IN ('red','blue') GROUP BY p.id HAVING COUNT(DISTINCT v.color)=2", 'Which people have both red and blue visits?', 'both_records'), [(1,)])

    def test_both_is_satisfied_by_separate_rows(self):
        self.assertEqual(self.repaired("SELECT p.id FROM people p JOIN visits v ON p.id=v.person_id WHERE v.color IN ('red','blue')", 'Which people have both red and blue visits?', 'both_records'), [(1,)])

    def test_absence_includes_people_without_children(self):
        self.assertEqual(self.repaired("SELECT p.id FROM people p JOIN visits v ON p.id=v.person_id WHERE v.color != 'red'", 'Which people never have red visits?', 'entity_absence'), [(3,)])

    def test_nullable_inner_key_does_not_poison_absence(self):
        self.assertEqual(self.repaired("SELECT p.id FROM people p WHERE p.id NOT IN (SELECT person_id FROM visits WHERE color='green')", 'Which people do not have green visits?', 'null_safe_absence'), [(1,), (2,), (3,)])

    def test_dedup_preserves_literal_case(self):
        self.assertNotEqual(rules.key("SELECT 'Red'"), rules.key("SELECT 'red'"))

    def test_duplicate_alias_draft_does_not_crash_qualification(self):
        tree = rules.parse('SELECT a.id FROM people a JOIN visits a ON a.id=a.person_id')
        self.assertIsNotNone(rules.qualify(tree, self.schema))

    def test_probe_reads_only_declared_columns(self):
        query = rules.probe([rules.qualify(rules.parse("SELECT p.id FROM people p WHERE p.name='a'"), self.schema)], self.schema)
        rows = self.db.execute(query).fetchall()
        self.assertIn(('people.name', 'A\x1fB\x1fC'), rows)

    def test_discovered_spelling_is_applied_to_every_set_branch(self):
        self.db.execute("UPDATE visits SET color=UPPER(SUBSTR(color,1,1)) || SUBSTR(color,2)")
        history = [{'tool':'execute','observation':{'ok':True,'columns':['field','samples'], 'rows':[['visits.color','Red\x1fBlue\x1fGreen']]}}]
        tree = rules.qualify(rules.parse("SELECT p.id FROM people p JOIN visits v ON p.id=v.person_id WHERE v.color='red' INTERSECT SELECT p.id FROM people p JOIN visits v ON p.id=v.person_id WHERE v.color='blue'"), self.schema)
        fixed = rules.consistent_literals(tree, self.schema, history)
        self.assertEqual(self.db.execute(fixed.sql(dialect='sqlite')).fetchall(), [(1,)])


if __name__ == '__main__':
    unittest.main()
