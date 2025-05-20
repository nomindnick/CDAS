"""
Network analysis for relationships between financial entities.

This module provides tools for analyzing networks of relationships between
financial entities such as documents, line items, and parties.
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Union
from sqlalchemy.orm import Session

from cdas.db.models import Document, LineItem, DocumentRelationship, AmountMatch
from cdas.financial_analysis.relationships.document import DocumentRelationshipAnalyzer
from cdas.financial_analysis.relationships.item import ItemRelationshipAnalyzer


class NetworkAnalyzer:
    """Analyzer for network relationships between financial entities."""

    def __init__(self, session: Session, config: Optional[Dict[str, Any]] = None):
        """
        Initialize NetworkAnalyzer.

        Args:
            session: SQLAlchemy session for database access
            config: Optional configuration dictionary
        """
        self.session = session
        self.config = config or {}
        self.doc_analyzer = DocumentRelationshipAnalyzer(session, self.config)
        self.item_analyzer = ItemRelationshipAnalyzer(session, self.config)
        self.graph = nx.MultiDiGraph()
        
    def build_document_network(self):
        """
        Build a network of document relationships.
        
        Returns:
            NetworkX DiGraph object representing document relationships
        """
        # Create a directed graph
        graph = nx.DiGraph()
        
        # Get all documents
        documents = self.session.query(Document).all()
        
        # Add document nodes
        for doc in documents:
            graph.add_node(doc.doc_id, 
                           type="document", 
                           doc_type=doc.doc_type,
                           party=doc.party,
                           meta_data=doc.meta_data)
        
        # Get document relationships
        relationships = self.session.query(DocumentRelationship).all()
        
        # Add relationship edges
        for rel in relationships:
            graph.add_edge(
                rel.source_doc_id,
                rel.target_doc_id,
                type=rel.relationship_type,
                confidence=float(rel.confidence) if rel.confidence else None,
                meta_data=rel.meta_data
            )
        
        return graph
    
    def build_item_network(self):
        """
        Build a network of line item relationships.
        
        Returns:
            NetworkX DiGraph object representing line item relationships
        """
        # Create a directed graph
        graph = nx.DiGraph()
        
        # Get all line items
        items = self.session.query(LineItem).all()
        
        # Add line item nodes
        for item in items:
            graph.add_node(item.item_id, 
                           type="line_item", 
                           doc_id=item.doc_id,
                           description=item.description,
                           amount=float(item.amount) if item.amount else None,
                           category=item.category,
                           status=item.status,
                           meta_data=item.meta_data)
        
        # Get amount matches
        matches = self.session.query(AmountMatch).all()
        
        # Add match edges
        for match in matches:
            graph.add_edge(
                match.source_item_id,
                match.target_item_id,
                type=match.match_type,
                confidence=float(match.confidence) if match.confidence else None,
                difference=float(match.difference) if match.difference else None,
                meta_data=match.meta_data
            )
        
        return graph
    
    def build_complete_network(self):
        """
        Build a complete network of all financial entities and their relationships.
        
        Returns:
            NetworkX MultiDiGraph object representing the complete financial network
        """
        # Create a multi-directed graph (can have multiple edges between nodes)
        graph = nx.MultiDiGraph()
        
        # Get all documents and line items
        documents = self.session.query(Document).all()
        items = self.session.query(LineItem).all()
        
        # Add document nodes
        for doc in documents:
            graph.add_node(f"doc_{doc.doc_id}", 
                           node_type="document", 
                           doc_type=doc.doc_type,
                           party=doc.party,
                           meta_data=doc.meta_data)
        
        # Add line item nodes
        for item in items:
            graph.add_node(f"item_{item.item_id}", 
                           node_type="line_item", 
                           doc_id=item.doc_id,
                           description=item.description,
                           amount=float(item.amount) if item.amount else None,
                           category=item.category,
                           status=item.status,
                           meta_data=item.meta_data)
            
            # Add edge from document to line item
            graph.add_edge(
                f"doc_{item.doc_id}",
                f"item_{item.item_id}",
                edge_type="contains"
            )
        
        # Get document relationships
        doc_relationships = self.session.query(DocumentRelationship).all()
        
        # Add document relationship edges
        for rel in doc_relationships:
            graph.add_edge(
                f"doc_{rel.source_doc_id}",
                f"doc_{rel.target_doc_id}",
                edge_type=rel.relationship_type,
                confidence=float(rel.confidence) if rel.confidence else None,
                meta_data=rel.meta_data
            )
        
        # Get amount matches
        item_matches = self.session.query(AmountMatch).all()
        
        # Add line item match edges
        for match in item_matches:
            graph.add_edge(
                f"item_{match.source_item_id}",
                f"item_{match.target_item_id}",
                edge_type=match.match_type,
                confidence=float(match.confidence) if match.confidence else None,
                difference=float(match.difference) if match.difference else None,
                meta_data=match.meta_data
            )
        
        return graph
        
    def calculate_centrality(self, centrality_type="degree"):
        """
        Calculate centrality measures for entities in the network.
        
        Args:
            centrality_type: Type of centrality to calculate
                             ('degree', 'betweenness', 'closeness', 'eigenvector')
            
        Returns:
            Dictionary mapping entity IDs to centrality scores
        """
        # Build the complete network
        graph = self.build_complete_network()
        
        # Calculate the requested centrality
        if centrality_type == "degree":
            centrality = nx.degree_centrality(graph)
        elif centrality_type == "betweenness":
            centrality = nx.betweenness_centrality(graph)
        elif centrality_type == "closeness":
            centrality = nx.closeness_centrality(graph)
        elif centrality_type == "eigenvector":
            centrality = nx.eigenvector_centrality(graph, max_iter=1000)
        else:
            raise ValueError(f"Unsupported centrality type: {centrality_type}")
        
        # Sort by centrality score (descending)
        sorted_centrality = sorted(
            centrality.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Format the results
        result = []
        for node_id, score in sorted_centrality:
            result.append({
                'id': node_id,
                'centrality_score': score,
                'data': graph.nodes[node_id]
            })
        
        return result
        
    def generate_document_graph(self):
        """Generate a graph of document relationships.
        
        Returns:
            Document relationship graph
        """
        return self.build_document_network()
        
    def generate_financial_graph(self, min_confidence: float = 0.7):
        """Generate a graph of financial relationships.
        
        Args:
            min_confidence: Minimum confidence for relationships
            
        Returns:
            Financial relationship graph
        """
        return self.build_complete_network()
    
    def get_entity_neighbors(self, entity_id, entity_type="line_item", depth=1):
        """
        Get neighbors of a specific entity in the network.
        
        Args:
            entity_id: ID of the entity
            entity_type: Type of entity ('document' or 'line_item')
            depth: Depth of neighborhood to traverse
            
        Returns:
            Dictionary of neighboring entities and their relationships
        """
        # Build the appropriate network
        if entity_type == "document":
            graph = self.build_document_network()
            node_id = entity_id
        elif entity_type == "line_item":
            graph = self.build_item_network()
            node_id = entity_id
        else:
            graph = self.build_complete_network()
            node_id = f"{entity_type}_{entity_id}"
        
        # Get the ego network (subgraph of node and its neighbors)
        ego_network = nx.ego_graph(graph, node_id, radius=depth)
        
        # Extract node and edge information
        result = {
            "entity": {
                "id": node_id,
                "type": entity_type,
                "data": graph.nodes[node_id] if node_id in graph.nodes else {}
            },
            "neighbors": [],
            "relationships": []
        }
        
        # Add neighbors
        for node in ego_network.nodes():
            if node != node_id:
                result["neighbors"].append({
                    "id": node,
                    "data": graph.nodes[node]
                })
        
        # Add edges
        for u, v, data in ego_network.edges(data=True):
            result["relationships"].append({
                "source": u,
                "target": v,
                "data": data
            })
        
        return result
    
    def find_paths(self, source_id, target_id, max_length=3):
        """
        Find all paths between two entities in the network.
        
        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            max_length: Maximum path length to consider
            
        Returns:
            List of paths between the entities
        """
        # Build the complete network
        graph = self.build_complete_network()
        
        # Find all simple paths
        try:
            paths = list(nx.all_simple_paths(
                graph, 
                source=source_id, 
                target=target_id, 
                cutoff=max_length
            ))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        
        # Format the results
        result_paths = []
        for path in paths:
            path_info = []
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                edge_data = graph.get_edge_data(source, target)
                
                path_info.append({
                    "source": source,
                    "source_data": graph.nodes[source],
                    "target": target,
                    "target_data": graph.nodes[target],
                    "edge_data": edge_data
                })
            
            result_paths.append(path_info)
        
        return result_paths
    
    def identify_clusters(self, min_cluster_size=2):
        """
        Identify clusters in the network using community detection.
        
        Args:
            min_cluster_size: Minimum number of nodes in a cluster
            
        Returns:
            List of clusters (groups of related entities)
        """
        # Build the complete network
        graph = self.build_complete_network()
        
        # Convert to undirected graph for community detection
        undirected = graph.to_undirected()
        
        # Use connected components for simple clustering
        clusters = list(nx.connected_components(undirected))
        
        # Filter by minimum size
        clusters = [c for c in clusters if len(c) >= min_cluster_size]
        
        # Format the results
        result_clusters = []
        for i, cluster in enumerate(clusters):
            cluster_nodes = []
            for node in cluster:
                cluster_nodes.append({
                    "id": node,
                    "data": graph.nodes[node]
                })
            
            result_clusters.append({
                "cluster_id": i,
                "size": len(cluster),
                "nodes": cluster_nodes
            })
        
        return result_clusters
        
    def detect_sequential_change_orders(self, min_confidence=0.8):
        """
        Detect sequential change orders that may be used to bypass approval thresholds.
        
        Args:
            min_confidence: Minimum confidence threshold for detection
            
        Returns:
            List of detected patterns with explanations
        """
        # Build the document network
        graph = self.build_document_network()
        
        # Extract change order nodes
        change_order_nodes = [
            node for node, data in graph.nodes(data=True) 
            if data.get('doc_type') == 'change_order'
        ]
        
        # Find sequential relationships between change orders
        sequential_patterns = []
        
        # Look for chains of related change orders
        for node in change_order_nodes:
            # Look for outgoing sequential relationships
            successors = list(graph.successors(node))
            co_successors = [
                succ for succ in successors
                if graph.nodes[succ].get('doc_type') == 'change_order'
            ]
            
            # Check if there are sequential change orders (related to each other)
            if co_successors:
                # Get the edge data
                for succ in co_successors:
                    edge_data = graph.get_edge_data(node, succ)
                    if edge_data and edge_data.get('type') == 'sequential':
                        confidence = edge_data.get('confidence', 0.0)
                        
                        if confidence >= min_confidence:
                            # Pattern detected
                            node_data = graph.nodes[node]
                            succ_data = graph.nodes[succ]
                            
                            pattern = {
                                'type': 'sequential_change_orders',
                                'source_id': node,
                                'target_id': succ,
                                'confidence': confidence,
                                'explanation': f"Sequential change orders potentially bypassing approval thresholds",
                                'details': {
                                    'source_doc_type': node_data.get('doc_type'),
                                    'target_doc_type': succ_data.get('doc_type')
                                }
                            }
                            
                            sequential_patterns.append(pattern)
        
        # Look for longer chains of change orders
        if len(sequential_patterns) > 0:
            # Check if there are linked sequential patterns forming a longer chain
            chains = []
            for pattern in sequential_patterns:
                source_id = pattern['source_id']
                target_id = pattern['target_id']
                
                # Check if this pattern's source is the target of another pattern
                is_extension = False
                for chain in chains:
                    if chain[-1] == source_id:
                        chain.append(target_id)
                        is_extension = True
                        break
                
                # If not part of an existing chain, start a new one
                if not is_extension:
                    chains.append([source_id, target_id])
            
            # Report chains of length 3 or more as potential threshold bypass attempts
            for chain in chains:
                if len(chain) >= 3:
                    # This is a significant chain of sequential change orders
                    pattern = {
                        'type': 'change_order_threshold_bypass',
                        'confidence': 0.9,
                        'explanation': f"Chain of {len(chain)} sequential change orders potentially bypassing approval thresholds",
                        'details': {
                            'chain': chain,
                        }
                    }
                    sequential_patterns.append(pattern)
        
        return sequential_patterns
        
    def detect_rejected_reappearing_scope(self, min_confidence=0.8):
        """
        Detect patterns where rejected scope reappears in different documents.
        
        Args:
            min_confidence: Minimum confidence threshold for detection
            
        Returns:
            List of detected patterns with explanations
        """
        # Build the networks
        doc_graph = self.build_document_network()
        full_graph = self.build_complete_network()
        
        # Find rejected change orders
        rejected_cos = []
        for node, data in doc_graph.nodes(data=True):
            if data.get('doc_type') == 'change_order':
                # Check if this CO has a rejection relationship
                for succ in doc_graph.successors(node):
                    edge = doc_graph.get_edge_data(node, succ)
                    if edge and edge.get('type') == 'rejection':
                        rejected_cos.append(node)
                        break
        
        # For each rejected CO, look for related line items that appear in other documents
        reappearing_patterns = []
        
        # If there's at least one rejected change order
        if rejected_cos:
            for co_id in rejected_cos:
                # Find line items in this change order
                co_items = []
                co_node_id = f"doc_{co_id}"
                
                # Find items that belong to this change order
                for node in full_graph.nodes():
                    if node.startswith("item_") and full_graph.has_edge(co_node_id, node):
                        co_items.append(node)
                
                # For each item, check if similar items appear in other documents
                for item_id in co_items:
                    # Get amount of this item
                    item_data = full_graph.nodes[item_id]
                    item_amount = item_data.get('amount')
                    
                    if item_amount is not None:
                        # Look for items with similar amounts in other documents
                        similar_items = []
                        
                        for other_node in full_graph.nodes():
                            if other_node.startswith("item_") and other_node != item_id:
                                other_data = full_graph.nodes[other_node]
                                other_amount = other_data.get('amount')
                                
                                # Compare amounts (allow 1% tolerance)
                                if other_amount is not None:
                                    tolerance = item_amount * 0.01
                                    if abs(other_amount - item_amount) <= tolerance:
                                        # Get the document this item belongs to
                                        doc_id = None
                                        for pred in full_graph.predecessors(other_node):
                                            if pred.startswith("doc_"):
                                                doc_id = pred
                                                break
                                        
                                        if doc_id and doc_id != co_node_id:
                                            similar_items.append({
                                                'item_id': other_node,
                                                'doc_id': doc_id,
                                                'amount': other_amount
                                            })
                        
                        # If similar items found in other documents
                        if similar_items:
                            pattern = {
                                'type': 'rejected_scope_reappearing',
                                'source_id': co_id,
                                'confidence': 0.85,
                                'explanation': f"Amount from rejected change order appears in other documents",
                                'details': {
                                    'rejected_item': item_id,
                                    'amount': item_amount,
                                    'similar_items': similar_items
                                }
                            }
                            reappearing_patterns.append(pattern)
        
        return reappearing_patterns
        
    def detect_suspicious_patterns(self, min_confidence=0.7):
        """
        Detect suspicious patterns in the network.
        
        Args:
            min_confidence: Minimum confidence threshold for detection
            
        Returns:
            List of detected suspicious patterns
        """
        suspicious_patterns = []
        
        # Check for sequential change orders
        sequential_cos = self.detect_sequential_change_orders(min_confidence)
        suspicious_patterns.extend(sequential_cos)
        
        # Check for rejected scope reappearing
        rejected_scope = self.detect_rejected_reappearing_scope(min_confidence)
        suspicious_patterns.extend(rejected_scope)
        
        # Detect circular references in the document network
        graph = self.build_document_network()
        try:
            cycles = list(nx.simple_cycles(graph))
            for cycle in cycles:
                if len(cycle) >= 2:
                    cycle_types = [graph.nodes[node].get('doc_type', 'unknown') for node in cycle]
                    
                    pattern = {
                        'type': 'circular_reference',
                        'confidence': 0.9,
                        'explanation': f"Circular reference between {len(cycle)} documents",
                        'details': {
                            'cycle': cycle,
                            'doc_types': cycle_types
                        }
                    }
                    suspicious_patterns.append(pattern)
        except nx.NetworkXNoCycle:
            pass
        
        # Sort patterns by confidence (descending)
        suspicious_patterns.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return suspicious_patterns
    
    def calculate_centrality(self, centrality_type="degree"):
        """
        Calculate centrality measures for entities in the network.
        
        Args:
            centrality_type: Type of centrality to calculate
                             ('degree', 'betweenness', 'closeness', 'eigenvector')
            
        Returns:
            Dictionary mapping entity IDs to centrality scores
        """
        # Build the complete network
        graph = self.build_complete_network()
        
        # Calculate the requested centrality
        if centrality_type == "degree":
            centrality = nx.degree_centrality(graph)
        elif centrality_type == "betweenness":
            centrality = nx.betweenness_centrality(graph)
        elif centrality_type == "closeness":
            centrality = nx.closeness_centrality(graph)
        elif centrality_type == "eigenvector":
            centrality = nx.eigenvector_centrality(graph, max_iter=1000)
        else:
            raise ValueError(f"Unsupported centrality type: {centrality_type}")
        
        # Sort by centrality score (descending)
        sorted_centrality = sorted(
            centrality.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Format the results
        result = []
        for node_id, score in sorted_centrality:
            result.append({
                "id": node_id,
                "centrality_score": score,
                "data": graph.nodes[node_id]
            })
        
        return result