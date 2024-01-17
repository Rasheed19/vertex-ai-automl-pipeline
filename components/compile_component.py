from typing import Callable
from kfp import compiler



def component_compiler(
    component: Callable,
    config_path_name: str,

) -> None:
    
    compiler.Compiler().compile(
       component,
       config_path_name,
    )