"""Statically recover one read-only query from model text; no SQL execution."""
import re
import sqlglot
from sqlglot.tokens import TokenType
from mp3_sql.sqlite_tool import validate_sql


def extract_sql(raw):
    text=re.sub(r'<think>.*?</think>','',raw,flags=re.S).strip()
    fenced=re.findall(r'```(?:sql|sqlite)?\s*([\s\S]*?)```',text,flags=re.I)
    texts=fenced+[text]
    for body in texts:
        for match in re.finditer(r'\b(?:SELECT|WITH)\b',body,re.I):
            candidate=body[match.start():].strip()
            variants=[candidate]
            try:
                tokens=sqlglot.tokenize(candidate,read='sqlite')
                first=next((token for token in tokens if token.token_type==TokenType.SEMICOLON),None)
                if first:variants.append(candidate[:first.start])
            except sqlglot.errors.SqlglotError:pass
            for sql in variants:
                try:
                    validate_sql(sql)
                    return sql
                except ValueError:pass
                try:
                    tokens=sqlglot.tokenize(sql.rstrip(';'),read='sqlite')
                    balance=sum(t.token_type==TokenType.L_PAREN for t in tokens)-sum(t.token_type==TokenType.R_PAREN for t in tokens)
                    if 0<balance<=4:
                        fixed=sql.rstrip(';')+')'*balance
                        validate_sql(fixed)
                        return fixed
                except (ValueError,sqlglot.errors.SqlglotError):pass
    return None
