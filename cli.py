#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import os
from typing import Optional, Any
from pydantic import BaseModel, Field

import argparse, argcomplete
from pydantic import BaseModel, Field
import rich_argparse

def add_model(model: BaseModel):
    "Add Pydantic model to an ArgumentParser"
    parser = argparse.ArgumentParser(
        description= 'Pydantic model CLI',
        formatter_class=rich_argparse.RichHelpFormatter,
    )

    fields = model.__fields__
    for name, field in fields.items():
        parser.add_argument(
            f"--{name}", 
            dest=name, 
            type=field.type_, 
            default=field.default,
            help=field.field_info.description,
        )
    args = parser.parse_args()
    argcomplete.autocomplete(parser)
    return args
