from pydantic import BaseModel
from pathlib import Path
from typing import Any
import subprocess
from rich import print, inspect as rich_inspect

def inspect_if(obj: Any, **kwargs):
	try:
		rich_inspect(obj, methods=True, **kwargs)
	except Exception as e:
		print(':inspect:error:', e)
		rich_inspect(e)

def filter_none(data: BaseModel):
    """ helper function to remove None values from a pydantic model and make dict """
    return dict(filter(lambda x: x[1] is not None, data.dict().items()))

def run_cmds(cmd: str|list, cwd:str | Path='.', silent=True, _res=[]):
	for cmd_1 in (cmd if isinstance(cmd, list) else [cmd,]): 
		try:
			cmd_1 = [c.strip() for c in cmd_1.split(' ')]
			_res = subprocess.run(cmd_1, cwd=str(cwd), capture_output=True, text=True)
			if not silent:
				print(f'Run: {cmd_1} at {cwd}')
				print('stdout:', _res.stdout, 'stderr:', _res.stderr, sep='\n')
		except Exception as e:
			if not silent:
				print(':run_cmds:', cmd_1, e)
			return ('Fail', '')
	return _res.stdout.rstrip('\n')

def mkdir(path: Path) -> Path:
	path = Path(path)
	if path.suffix != '':
		path = path.parent
	try:
		if not path.exists() or not path.is_dir():
			path.mkdir(parents=True)
	except Exception as e:
		print(':mkdir:', e)
	return path
