"""
Cross-project operations utility for CDAS.

This module provides utilities for executing operations across multiple 
project databases, including parallel execution and result aggregation.
"""

import logging
import concurrent.futures
from typing import List, Dict, Any, Callable, Optional, TypeVar, Generic, Union

from cdas.db.project_manager import get_project_db_manager, session_scope
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Type variables for better type hinting
T = TypeVar('T')  # Operation result type
E = TypeVar('E')  # Error result type


class ProjectOperationResult(Generic[T, E]):
    """
    Container for the result of an operation executed on a project.
    
    Attributes:
        project_id: The ID of the project the operation was executed on
        success: Whether the operation succeeded
        result: The result of the operation (if successful)
        error: The error that occurred (if unsuccessful)
    """
    
    def __init__(self, project_id: str, success: bool, result: Optional[T] = None, error: Optional[E] = None):
        """
        Initialize the operation result.
        
        Args:
            project_id: The ID of the project the operation was executed on
            success: Whether the operation succeeded
            result: The result of the operation (if successful)
            error: The error that occurred (if unsuccessful)
        """
        self.project_id = project_id
        self.success = success
        self.result = result
        self.error = error
    
    def __repr__(self) -> str:
        """String representation of the operation result."""
        if self.success:
            return f"ProjectOperationResult(project_id='{self.project_id}', success=True, result={repr(self.result)})"
        else:
            return f"ProjectOperationResult(project_id='{self.project_id}', success=False, error={repr(self.error)})"


def run_across_projects(
    operation: Callable[[Session, ...], T],
    projects: Optional[List[str]] = None,
    max_workers: int = 4,
    include_errors: bool = False,
    **operation_kwargs
) -> Dict[str, Union[T, ProjectOperationResult[T, Exception]]]:
    """
    Run an operation across multiple projects.
    
    Args:
        operation: Function to run for each project. Should accept a session as first arg.
        projects: List of project IDs to run on. If None, runs on all projects.
        max_workers: Maximum number of concurrent workers.
        include_errors: If True, return ProjectOperationResult objects instead of just results.
        operation_kwargs: Additional kwargs to pass to the operation function.
        
    Returns:
        Dict mapping project IDs to operation results (or ProjectOperationResult objects if
        include_errors is True).
    """
    project_manager = get_project_db_manager()
    
    # Get list of projects to operate on
    if projects is None:
        projects = project_manager.get_project_list()
    
    if not projects:
        logger.warning("No projects found to run operation on.")
        return {}
    
    logger.info(f"Running operation across {len(projects)} projects: {', '.join(projects)}")
    
    results: Dict[str, Union[T, ProjectOperationResult[T, Exception]]] = {}
    
    # For small numbers of projects, run sequentially
    if len(projects) <= 2 or max_workers <= 1:
        for project_id in projects:
            try:
                with session_scope(project_id) as session:
                    result = operation(session, **operation_kwargs)
                    if include_errors:
                        results[project_id] = ProjectOperationResult(project_id, True, result)
                    else:
                        results[project_id] = result
                    logger.info(f"Successfully executed operation on project '{project_id}'")
            except Exception as e:
                logger.exception(f"Error executing operation on project '{project_id}': {str(e)}")
                if include_errors:
                    results[project_id] = ProjectOperationResult(project_id, False, error=e)
    else:
        # For larger numbers, use concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Define a worker function that runs the operation for a project
            def worker(project_id):
                try:
                    with session_scope(project_id) as session:
                        result = operation(session, **operation_kwargs)
                        logger.info(f"Successfully executed operation on project '{project_id}'")
                        return ProjectOperationResult(project_id, True, result)
                except Exception as e:
                    logger.exception(f"Error executing operation on project '{project_id}': {str(e)}")
                    return ProjectOperationResult(project_id, False, error=e)
            
            # Submit all tasks
            future_to_project = {executor.submit(worker, p): p for p in projects}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_project):
                project_id = future_to_project[future]
                try:
                    operation_result = future.result()
                    if include_errors:
                        results[project_id] = operation_result
                    elif operation_result.success:
                        results[project_id] = operation_result.result
                except Exception as e:
                    logger.exception(f"Unexpected error in worker for project '{project_id}': {str(e)}")
                    if include_errors:
                        results[project_id] = ProjectOperationResult(project_id, False, error=e)
    
    logger.info(f"Completed operation across {len(projects)} projects. "
               f"Successful: {sum(1 for r in results.values() if (isinstance(r, ProjectOperationResult) and r.success) or not isinstance(r, ProjectOperationResult))}")
    
    return results


def search_across_projects(
    search_function: Callable[[Session, ...], List[Any]],
    projects: Optional[List[str]] = None,
    max_workers: int = 4,
    **search_kwargs
) -> Dict[str, List[Any]]:
    """
    Run a search operation across multiple projects and return the results.
    
    This is a specialized version of run_across_projects optimized for search operations
    that return lists of results.
    
    Args:
        search_function: Search function to run for each project. Should accept a session as first arg.
        projects: List of project IDs to run on. If None, runs on all projects.
        max_workers: Maximum number of concurrent workers.
        search_kwargs: Additional kwargs to pass to the search function.
        
    Returns:
        Dict mapping project IDs to search results.
    """
    return run_across_projects(
        search_function,
        projects=projects,
        max_workers=max_workers,
        include_errors=False,
        **search_kwargs
    )


def aggregate_search_results(search_results: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate search results from multiple projects, adding project_id to each item.
    
    Args:
        search_results: Dict mapping project IDs to search results from each project.
        
    Returns:
        List of result items with project_id added to each item.
    """
    aggregated = []
    
    for project_id, results in search_results.items():
        for item in results:
            # If item is a dict, add project_id
            if isinstance(item, dict):
                item_copy = item.copy()
                item_copy['project_id'] = project_id
                aggregated.append(item_copy)
            # If item is a database model, convert to dict and add project_id
            elif hasattr(item, '__dict__'):
                item_dict = {k: v for k, v in vars(item).items() if not k.startswith('_')}
                item_dict['project_id'] = project_id
                aggregated.append(item_dict)
            # Otherwise, wrap in a dict with project_id
            else:
                aggregated.append({'item': item, 'project_id': project_id})
    
    return aggregated


def find_in_all_projects(
    search_function: Callable[[Session, ...], List[Any]],
    aggregate: bool = True,
    max_workers: int = 4,
    **search_kwargs
) -> Union[Dict[str, List[Any]], List[Dict[str, Any]]]:
    """
    Find items across all projects using the provided search function.
    
    Args:
        search_function: Search function to run for each project. Should accept a session as first arg.
        aggregate: If True, aggregate results into a single list with project_id added.
        max_workers: Maximum number of concurrent workers.
        search_kwargs: Additional kwargs to pass to the search function.
        
    Returns:
        If aggregate is True, returns a list of items with project_id added to each.
        If aggregate is False, returns a dict mapping project IDs to search results.
    """
    results = search_across_projects(
        search_function,
        projects=None,  # All projects
        max_workers=max_workers,
        **search_kwargs
    )
    
    if aggregate:
        return aggregate_search_results(results)
    else:
        return results


def copy_between_projects(
    export_operation: Callable[[Session], List[Dict[str, Any]]],
    import_operation: Callable[[Session, List[Dict[str, Any]]], int],
    source_project: str,
    target_project: str
) -> Dict[str, Any]:
    """
    Copy data from one project to another using the provided export and import operations.
    
    Args:
        export_operation: Function to export data from the source project.
        import_operation: Function to import data into the target project.
        source_project: Source project ID to copy from.
        target_project: Target project ID to copy to.
        
    Returns:
        Dict with information about the operation:
        {
            'source_project': source_project,
            'target_project': target_project,
            'items_exported': Number of items exported,
            'items_imported': Number of items imported
        }
    """
    # Export data from source project
    with session_scope(source_project) as source_session:
        exported_data = export_operation(source_session)
    
    # Import data into target project
    with session_scope(target_project) as target_session:
        items_imported = import_operation(target_session, exported_data)
    
    return {
        'source_project': source_project,
        'target_project': target_project,
        'items_exported': len(exported_data),
        'items_imported': items_imported
    }