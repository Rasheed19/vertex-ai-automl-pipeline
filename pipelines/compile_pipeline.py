from typing import Callable
from kfp import compiler



def pipeline_compiler(
    pipeline: Callable,
    config_path_name: str,

) -> None:
    
   compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path=config_path_name,
)


