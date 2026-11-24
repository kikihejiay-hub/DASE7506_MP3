"""Task-independent selection safeguards using public schema and question cues."""
import itertools
import re
from difflib import SequenceMatcher
import sqlglot
from sqlglot import exp


def parse(sql):
    try:return sqlglot.parse_one(sql,read='sqlite')
    except (sqlglot.errors.SqlglotError,RecursionError):return None


def align_projection(sql,headers):
    tree=parse(sql)
    if not isinstance(tree,exp.Select) or not 1<len(headers)<=5 or len(tree.expressions)!=len(headers):return sql
    def normalize(value):return re.sub(r'[^a-z0-9]','',value.lower())
    names=[]
    for expression in tree.expressions:
        body=expression.this.copy() if isinstance(expression,exp.Alias) else expression.copy()
        for c in body.find_all(exp.Column):c.set('table',None)
        names.append([normalize(body.sql()),normalize(expression.output_name)])
    scores=[[max(SequenceMatcher(None,n,normalize(h)).ratio() for n in row) for h in headers] for row in names]
    permutations=list(itertools.permutations(range(len(headers))))
    best=max(permutations,key=lambda p:sum(scores[p[j]][j] for j in range(len(headers))))
    old=sum(scores[j][j] for j in range(len(headers)))
    if sum(scores[best[j]][j] for j in range(len(headers)))>old+.5 and min(scores[best[j]][j] for j in range(len(headers)))>=.6:
        tree.set('expressions',[tree.expressions[i].copy() for i in best])
        return tree.sql(dialect='sqlite')
    return sql


def prefer_baseline(new_sql,old_sql,observation,schema,quality):
    new,old=parse(new_sql),parse(old_sql)
    if not isinstance(new,exp.Select) or not isinstance(old,exp.Select):return False
    q=observation['task']['question'].lower()
    words=set(re.findall(r'[a-z0-9]+',q))
    def coverage(tree):
        return {word for literal in tree.find_all(exp.Literal) for word in re.findall(r'[a-z0-9]+',str(literal.this).lower()) if word in words}
    if coverage(old)-coverage(new):return True
    if re.search(r'\bdistinct\b|\bunique\b|\bdifferent\b',q) and old.args.get('distinct') and not new.args.get('distinct'):return True
    if quality(old_sql,observation,schema)>quality(new_sql,observation,schema)+.5:return True
    if re.search(r'\bmost\b|\bleast\b|\bcommon\b|\bfrequent\b|\bfewest\b',q):
        if old.find(exp.Count) and not new.find(exp.Count):return True
    for cue,wrong,right in [('at most',(exp.GT,exp.GTE),(exp.LT,exp.LTE)),('at least',(exp.LT,exp.LTE),(exp.GT,exp.GTE))]:
        if cue in q and any(isinstance(n,wrong) and n.find(exp.Count) for n in new.walk()) and any(isinstance(n,right) and n.find(exp.Count) for n in old.walk()):return True
    return False
