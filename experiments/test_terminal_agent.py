import importlib.util
from pathlib import Path
import sqlite3
import unittest

spec = importlib.util.spec_from_file_location('terminal_tests', Path(__file__).resolve().parent.parent / 'submission/terminal_agent.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TerminalTests(unittest.TestCase):
    def schema(self):
        return [{'name':'people','columns':[{'name':'name','type':'TEXT','pk':0}], 'foreign_keys':[]}]

    def test_text_rescue_handles_case_and_boundary_spaces(self):
        connection = sqlite3.connect(':memory:')
        connection.executescript("CREATE TABLE people(name TEXT); INSERT INTO people VALUES('ALICE ')")
        query = module.normalized(module.rules.parse("SELECT p.name FROM people p WHERE p.name='Alice'"), self.schema(), 'Find Alice')
        self.assertEqual(connection.execute(query.sql(dialect='sqlite')).fetchall(), [('ALICE ',)])
        connection.close()

    def test_public_pass_is_never_overridden(self):
        sql = "SELECT p.name FROM people p WHERE p.name='Alice'"
        class Parent:
            pool = {}
            def act(self, observation):
                return {'tool':'submit','sql':sql}
        agent = module.TerminalAgent()
        agent.base = Parent()
        observation = {'history':[{'tool':'check','sql':sql,'observation':{'ok':True,'matches_example':True}}]}
        self.assertEqual(agent.act(observation), {'tool':'submit','sql':sql})


if __name__ == '__main__':
    unittest.main()
