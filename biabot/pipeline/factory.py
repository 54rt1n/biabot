# biabot/pipeline/factory.py

from typing import Callable

from ..constants import PIPELINE_ANALYSIS, PIPELINE_JOURNAL, PIPELINE_CHORE, PIPELINE_REPORT

# This import is not getting used here, but can be imported from here by other modules
from .base import BasePipeline
from .analyze import analysis_pipeline
from .journal import journal_pipeline
from .chore import chore_pipeline
from .report import report_pipeline

def pipeline_factory(pipeline_type: str) -> Callable:
    """
    Provides a factory function to create pipeline instances based on the specified pipeline type.
    
    Args:
        pipeline_type (str): The type of pipeline to create, which must be one of the following constants:
            - PIPELINE_ANALYSIS
            - PIPELINE_JOURNAL
            - PIPELINE_CHORE
            - PIPELINE_REPORT
    
    Returns:
        Callable: A function that creates an instance of the specified pipeline type.
    
    Raises:
        ValueError: If the provided `pipeline_type` is not a valid pipeline type.
    """
        
    if pipeline_type == PIPELINE_ANALYSIS:
        return analysis_pipeline
    elif pipeline_type == PIPELINE_JOURNAL:
        return journal_pipeline
    elif pipeline_type == PIPELINE_CHORE:
        return chore_pipeline
    elif pipeline_type == PIPELINE_REPORT:
        return report_pipeline
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
