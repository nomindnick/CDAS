"""Vector Index management for semantic search.

This module provides functionality for managing vector indices for efficient
semantic search operations across document embeddings.
"""

import logging
import pickle
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import os
import time

logger = logging.getLogger(__name__)


class VectorIndex:
    """Manages a vector index for efficient semantic search."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the vector index.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Index storage settings
        self.index_dir = self.config.get('index_dir', 'data/vector_indices')
        self.index_name = self.config.get('index_name', 'default_index')
        
        # Vector dimensions
        self.dimensions = self.config.get('dimensions', 1536)  # Default for embedding-3-small
        
        # Indexing parameters
        self.batch_size = self.config.get('batch_size', 1000)
        self.rebuild_threshold = self.config.get('rebuild_threshold', 0.2)  # Rebuild if 20% new entries
        
        # Initialize index structures
        self.vectors = []  # List of embedding vectors
        self.metadata = []  # List of metadata objects
        self.id_to_index = {}  # Mapping from ID to index in vectors list
        
        # Index stats
        self.last_modified = None
        self.total_vectors = 0
        self.is_dirty = False  # Flag for changes since last save
        
        # Load the index if it exists
        self._ensure_index_dir()
        self._load_or_initialize()
    
    def add_vector(self, vector_id: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """Add a vector to the index.
        
        Args:
            vector_id: Unique identifier for the vector
            vector: The embedding vector
            metadata: Metadata associated with the vector
            
        Returns:
            Success flag
        """
        try:
            # Validate vector
            if len(vector) != self.dimensions:
                logger.error(f"Vector dimension mismatch: expected {self.dimensions}, got {len(vector)}")
                return False
            
            # Check if vector exists and needs update
            if vector_id in self.id_to_index:
                idx = self.id_to_index[vector_id]
                self.vectors[idx] = vector
                self.metadata[idx] = metadata
                logger.debug(f"Updated existing vector for {vector_id}")
            else:
                # Add new vector
                self.vectors.append(vector)
                self.metadata.append(metadata)
                self.id_to_index[vector_id] = len(self.vectors) - 1
                self.total_vectors += 1
                logger.debug(f"Added new vector for {vector_id}")
            
            # Mark index as dirty
            self.is_dirty = True
            self.last_modified = time.time()
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding vector to index: {str(e)}")
            return False
    
    def batch_add_vectors(self, vector_data: List[Tuple[str, List[float], Dict[str, Any]]]) -> Dict[str, Any]:
        """Add multiple vectors to the index.
        
        Args:
            vector_data: List of (vector_id, vector, metadata) tuples
            
        Returns:
            Summary of operation
        """
        if not vector_data:
            return {'added': 0, 'updated': 0, 'failed': 0, 'success': True}
        
        added = 0
        updated = 0
        failed = 0
        
        try:
            for vector_id, vector, metadata in vector_data:
                # Validate vector
                if len(vector) != self.dimensions:
                    logger.error(f"Vector dimension mismatch for {vector_id}: expected {self.dimensions}, got {len(vector)}")
                    failed += 1
                    continue
                
                # Check if vector exists and needs update
                if vector_id in self.id_to_index:
                    idx = self.id_to_index[vector_id]
                    self.vectors[idx] = vector
                    self.metadata[idx] = metadata
                    updated += 1
                else:
                    # Add new vector
                    self.vectors.append(vector)
                    self.metadata.append(metadata)
                    self.id_to_index[vector_id] = len(self.vectors) - 1
                    self.total_vectors += 1
                    added += 1
            
            # Mark index as dirty
            self.is_dirty = True
            self.last_modified = time.time()
            
            # Check if index should be saved
            if added + updated >= self.batch_size:
                self.save_index()
            
            return {
                'added': added,
                'updated': updated,
                'failed': failed,
                'total': len(vector_data),
                'success': True
            }
        
        except Exception as e:
            logger.error(f"Error batch adding vectors to index: {str(e)}")
            return {
                'added': added,
                'updated': updated,
                'failed': failed + (len(vector_data) - (added + updated)),
                'total': len(vector_data),
                'success': False,
                'error': str(e)
            }
    
    def remove_vector(self, vector_id: str) -> bool:
        """Remove a vector from the index.
        
        Args:
            vector_id: ID of vector to remove
            
        Returns:
            Success flag
        """
        if vector_id not in self.id_to_index:
            logger.warning(f"Vector {vector_id} not found in index")
            return False
        
        try:
            # Get index of vector
            idx = self.id_to_index[vector_id]
            
            # Remove vector and metadata
            del self.vectors[idx]
            del self.metadata[idx]
            
            # Update id_to_index mapping
            del self.id_to_index[vector_id]
            
            # Update mappings for all vectors after the removed one
            for vid, index in list(self.id_to_index.items()):
                if index > idx:
                    self.id_to_index[vid] = index - 1
            
            self.total_vectors -= 1
            self.is_dirty = True
            self.last_modified = time.time()
            
            return True
        
        except Exception as e:
            logger.error(f"Error removing vector from index: {str(e)}")
            return False
    
    def search(self, query_vector: List[float], limit: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search the index for similar vectors.
        
        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List of search results with metadata
        """
        if not self.vectors:
            logger.warning("Empty vector index, no results")
            return []
        
        try:
            # Validate query vector
            if len(query_vector) != self.dimensions:
                logger.error(f"Query vector dimension mismatch: expected {self.dimensions}, got {len(query_vector)}")
                return []
            
            # Convert inputs to NumPy arrays for faster processing
            query_np = np.array(query_vector)
            vectors_np = np.array(self.vectors)
            
            # Compute cosine similarity
            # (dot product of normalized vectors is equivalent to cosine similarity)
            # First normalize vectors
            query_norm = query_np / np.linalg.norm(query_np)
            vectors_norm = vectors_np / np.linalg.norm(vectors_np, axis=1, keepdims=True)
            
            # Compute dot product (cosine similarity)
            similarities = np.dot(vectors_norm, query_norm)
            
            # Apply threshold filter
            above_threshold = similarities >= threshold
            
            # Get indices of results sorted by similarity (descending)
            result_indices = np.argsort(-similarities[above_threshold])
            
            # Limit results
            result_indices = result_indices[:limit]
            
            # Format results
            results = []
            for idx in result_indices:
                true_idx = np.where(above_threshold)[0][idx]
                similarity = float(similarities[true_idx])
                
                # Find the vector_id for this index
                vector_id = None
                for vid, index in self.id_to_index.items():
                    if index == true_idx:
                        vector_id = vid
                        break
                
                results.append({
                    'vector_id': vector_id,
                    'similarity': similarity,
                    'metadata': self.metadata[true_idx]
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching vector index: {str(e)}")
            return []
    
    def filtered_search(self, query_vector: List[float], metadata_filters: Dict[str, Any],
                      limit: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search the index with metadata filters.
        
        Args:
            query_vector: The query embedding vector
            metadata_filters: Filters to apply to metadata
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List of search results with metadata
        """
        if not self.vectors:
            logger.warning("Empty vector index, no results")
            return []
        
        try:
            # Validate query vector
            if len(query_vector) != self.dimensions:
                logger.error(f"Query vector dimension mismatch: expected {self.dimensions}, got {len(query_vector)}")
                return []
            
            # Apply metadata filtering first to reduce computation
            filtered_indices = []
            for idx, meta in enumerate(self.metadata):
                match = True
                for key, value in metadata_filters.items():
                    # Skip filtering if the key doesn't exist in metadata
                    if key not in meta:
                        match = False
                        break
                    
                    # Handle different filter types
                    if isinstance(value, list):
                        # List values: check if metadata value is in the filter list
                        if meta[key] not in value:
                            match = False
                            break
                    elif isinstance(value, dict):
                        # Dict values: support range queries with min/max
                        if 'min' in value and meta[key] < value['min']:
                            match = False
                            break
                        if 'max' in value and meta[key] > value['max']:
                            match = False
                            break
                    else:
                        # Direct comparison
                        if meta[key] != value:
                            match = False
                            break
                
                if match:
                    filtered_indices.append(idx)
            
            if not filtered_indices:
                logger.debug("No vectors match the metadata filters")
                return []
            
            # Extract filtered vectors
            filtered_vectors = [self.vectors[idx] for idx in filtered_indices]
            
            # Convert to NumPy arrays
            query_np = np.array(query_vector)
            vectors_np = np.array(filtered_vectors)
            
            # Compute cosine similarity
            query_norm = query_np / np.linalg.norm(query_np)
            vectors_norm = vectors_np / np.linalg.norm(vectors_np, axis=1, keepdims=True)
            similarities = np.dot(vectors_norm, query_norm)
            
            # Apply threshold filter
            above_threshold = similarities >= threshold
            
            # Sort results by similarity
            result_indices = np.argsort(-similarities[above_threshold])
            result_indices = result_indices[:limit]
            
            # Format results
            results = []
            for idx in result_indices:
                true_idx = filtered_indices[np.where(above_threshold)[0][idx]]
                similarity = float(similarities[np.where(above_threshold)[0][idx]])
                
                # Find the vector_id for this index
                vector_id = None
                for vid, index in self.id_to_index.items():
                    if index == true_idx:
                        vector_id = vid
                        break
                
                results.append({
                    'vector_id': vector_id,
                    'similarity': similarity,
                    'metadata': self.metadata[true_idx]
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Error performing filtered search: {str(e)}")
            return []
    
    def get_vector(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Get a vector and its metadata by ID.
        
        Args:
            vector_id: ID of the vector
            
        Returns:
            Vector data or None if not found
        """
        if vector_id not in self.id_to_index:
            return None
        
        try:
            idx = self.id_to_index[vector_id]
            
            return {
                'vector_id': vector_id,
                'vector': self.vectors[idx],
                'metadata': self.metadata[idx]
            }
        
        except Exception as e:
            logger.error(f"Error getting vector {vector_id}: {str(e)}")
            return None
    
    def save_index(self) -> bool:
        """Save the index to disk.
        
        Returns:
            Success flag
        """
        if not self.is_dirty:
            logger.debug("Index not dirty, skipping save")
            return True
        
        try:
            self._ensure_index_dir()
            
            # Prepare index data
            index_data = {
                'dimensions': self.dimensions,
                'vectors': self.vectors,
                'metadata': self.metadata,
                'id_to_index': self.id_to_index,
                'total_vectors': self.total_vectors,
                'last_modified': time.time()
            }
            
            # Save to file
            index_path = os.path.join(self.index_dir, f"{self.index_name}.pkl")
            
            # Create a temporary file first
            temp_path = f"{index_path}.tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(index_data, f)
            
            # Rename to final file
            os.replace(temp_path, index_path)
            
            # Save a metadata file with basic index info
            metadata_path = os.path.join(self.index_dir, f"{self.index_name}.meta")
            with open(metadata_path, 'w') as f:
                f.write(f"Index: {self.index_name}\n")
                f.write(f"Dimensions: {self.dimensions}\n")
                f.write(f"Total vectors: {self.total_vectors}\n")
                f.write(f"Last modified: {time.ctime(time.time())}\n")
            
            self.is_dirty = False
            logger.info(f"Saved index '{self.index_name}' with {self.total_vectors} vectors")
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            return False
    
    def load_index(self, index_name: Optional[str] = None) -> bool:
        """Load an index from disk.
        
        Args:
            index_name: Optional name of index to load (uses the default if None)
            
        Returns:
            Success flag
        """
        if index_name:
            self.index_name = index_name
        
        try:
            index_path = os.path.join(self.index_dir, f"{self.index_name}.pkl")
            
            if not os.path.exists(index_path):
                logger.warning(f"Index file '{index_path}' not found")
                return False
            
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            # Validate index structure
            required_keys = ['dimensions', 'vectors', 'metadata', 'id_to_index', 'total_vectors']
            if not all(key in index_data for key in required_keys):
                logger.error(f"Invalid index file: missing required keys")
                return False
            
            # Update index attributes
            self.dimensions = index_data['dimensions']
            self.vectors = index_data['vectors']
            self.metadata = index_data['metadata']
            self.id_to_index = index_data['id_to_index']
            self.total_vectors = index_data['total_vectors']
            self.last_modified = index_data.get('last_modified', time.time())
            
            # Reset dirty flag
            self.is_dirty = False
            
            logger.info(f"Loaded index '{self.index_name}' with {self.total_vectors} vectors")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def list_indices(self) -> List[Dict[str, Any]]:
        """List all available indices.
        
        Returns:
            List of index information
        """
        try:
            self._ensure_index_dir()
            
            index_files = []
            for file in os.listdir(self.index_dir):
                if file.endswith('.meta'):
                    index_name = file[:-5]  # Remove .meta extension
                    
                    # Read metadata file
                    metadata_path = os.path.join(self.index_dir, file)
                    with open(metadata_path, 'r') as f:
                        metadata_lines = f.readlines()
                    
                    # Parse metadata
                    metadata = {}
                    for line in metadata_lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            metadata[key.strip()] = value.strip()
                    
                    # Get index file size
                    index_path = os.path.join(self.index_dir, f"{index_name}.pkl")
                    file_size = os.path.getsize(index_path) if os.path.exists(index_path) else 0
                    
                    index_files.append({
                        'name': index_name,
                        'total_vectors': int(metadata.get('Total vectors', 0)),
                        'dimensions': int(metadata.get('Dimensions', 0)),
                        'last_modified': metadata.get('Last modified', ''),
                        'file_size': file_size
                    })
            
            return index_files
        
        except Exception as e:
            logger.error(f"Error listing indices: {str(e)}")
            return []
    
    def clear_index(self) -> bool:
        """Clear the current index.
        
        Returns:
            Success flag
        """
        try:
            self.vectors = []
            self.metadata = []
            self.id_to_index = {}
            self.total_vectors = 0
            self.is_dirty = True
            self.last_modified = time.time()
            
            logger.info(f"Cleared index '{self.index_name}'")
            
            return True
        
        except Exception as e:
            logger.error(f"Error clearing index: {str(e)}")
            return False
    
    def delete_index(self, index_name: Optional[str] = None) -> bool:
        """Delete an index from disk.
        
        Args:
            index_name: Optional name of index to delete (uses the default if None)
            
        Returns:
            Success flag
        """
        name_to_delete = index_name or self.index_name
        
        try:
            index_path = os.path.join(self.index_dir, f"{name_to_delete}.pkl")
            meta_path = os.path.join(self.index_dir, f"{name_to_delete}.meta")
            
            # Check if files exist
            if not os.path.exists(index_path) and not os.path.exists(meta_path):
                logger.warning(f"Index '{name_to_delete}' not found")
                return False
            
            # Delete files
            if os.path.exists(index_path):
                os.remove(index_path)
            
            if os.path.exists(meta_path):
                os.remove(meta_path)
            
            # If we deleted the current index, reset it
            if name_to_delete == self.index_name:
                self.clear_index()
            
            logger.info(f"Deleted index '{name_to_delete}'")
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'index_name': self.index_name,
            'dimensions': self.dimensions,
            'total_vectors': self.total_vectors,
            'last_modified': self.last_modified,
            'is_dirty': self.is_dirty,
            'batch_size': self.batch_size
        }
    
    def _ensure_index_dir(self) -> None:
        """Ensure the index directory exists."""
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_or_initialize(self) -> None:
        """Load index if it exists, otherwise initialize a new one."""
        index_path = os.path.join(self.index_dir, f"{self.index_name}.pkl")
        
        if os.path.exists(index_path):
            self.load_index()
        else:
            logger.info(f"No existing index found, initializing new index '{self.index_name}'")
            # Index is already initialized with empty structures