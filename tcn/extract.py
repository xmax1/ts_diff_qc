from pathlib import Path
from pprint import pprint
import ast

from pydantic import BaseModel
import rope 

import ast
import re

repo = Path(".")
paths = list(repo.rglob("*.py"))
paths = [p for p in paths if p.name.endswith("_test.py")]

for p in paths:

    code = p.read_text()

    # functions: list[ast.FunctionDef] = [n for n in node.body if isinstance(n, ast.FunctionDef)]
    # classes: list[ast.ClassDef] = [n for n in node.body if isinstance(n, ast.ClassDef)]

    def get_class(node: ast.AST):
        tree = ast.parse(code, mode= 'exec')
        for node in ast.walk(tree):   
            if isinstance(node, ast.ClassDef):
                text = ast.get_source_segment(code, node)
                print(text)

    def fix_syntax(code: str, p: Path):
        p = p.with_suffix('')
        p = p.name + '_fixed.py'
        with open(p, 'w') as f:
            for line in code.splitlines():
                try:
                    ast.parse(line)
                except SyntaxError:
                    line = '# syntax_error: ' + line
                f.write(line  + '\n')
    
    fix_syntax(code, p)