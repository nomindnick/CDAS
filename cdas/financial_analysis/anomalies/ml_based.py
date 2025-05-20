"""
Machine learning based anomaly detector for the financial analysis engine.

This module provides anomaly detection capabilities using machine learning techniques.
It is designed to be an optional, more advanced component that can be enabled
when sufficient data is available for training.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text
from decimal import Decimal

import logging
logger = logging.getLogger(__name__)


class MLBasedAnomalyDetector:
    """Detects anomalies in financial data using machine learning techniques."""
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """Initialize the ML-based anomaly detector.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Default configuration
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.min_data_points = self.config.get('min_data_points', 50)  # Need more data for ML
        self.is_trained = False
        self.ml_backend = self.config.get('ml_backend', 'isolation_forest')
        
        # Model parameters
        self.model = None
        self.feature_names = []
        self.feature_scalers = {}
        
        # Check if ML is enabled
        self.ml_enabled = self.config.get('enable_ml_anomaly_detection', False)
        if not self.ml_enabled:
            logger.info("ML-based anomaly detection is disabled")
        else:
            try:
                # Check if required dependencies are available
                self._check_dependencies()
                logger.info(f"ML-based anomaly detection is enabled (backend: {self.ml_backend})")
            except ImportError as e:
                self.ml_enabled = False
                logger.warning(f"ML-based anomaly detection is disabled due to missing dependencies: {e}")
    
    def detect_anomalies(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect ML-based anomalies in financial data.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of detected anomalies
        """
        if not self.ml_enabled:
            logger.info("ML-based anomaly detection is disabled, returning empty results")
            return []
            
        logger.info(f"Detecting ML-based anomalies{f' for document {doc_id}' if doc_id else ''}")
        
        # Train model if not already trained
        if not self.is_trained:
            success = self._train_model()
            if not success:
                logger.warning("Failed to train ML model, returning empty results")
                return []
        
        # Get data to analyze
        data = self._get_data_to_analyze(doc_id)
        if not data:
            logger.info("No data to analyze, returning empty results")
            return []
            
        # Predict anomalies
        anomalies = self._predict_anomalies(data)
        
        logger.info(f"Found {len(anomalies)} ML-based anomalies")
        return anomalies
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available.
        
        Raises:
            ImportError: If any required dependency is missing
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required for ML-based anomaly detection")
            
        # Check backend-specific dependencies
        if self.ml_backend == 'isolation_forest':
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.preprocessing import StandardScaler
            except ImportError:
                raise ImportError("scikit-learn is required for isolation forest backend")
        elif self.ml_backend == 'dbscan':
            try:
                from sklearn.cluster import DBSCAN
                from sklearn.preprocessing import StandardScaler
            except ImportError:
                raise ImportError("scikit-learn is required for DBSCAN backend")
        elif self.ml_backend == 'local_outlier_factor':
            try:
                from sklearn.neighbors import LocalOutlierFactor
                from sklearn.preprocessing import StandardScaler
            except ImportError:
                raise ImportError("scikit-learn is required for Local Outlier Factor backend")
    
    def _train_model(self) -> bool:
        """Train the anomaly detection model.
        
        Returns:
            True if model was successfully trained, False otherwise
        """
        logger.info(f"Training {self.ml_backend} model for anomaly detection")
        
        # Get training data
        data = self._get_training_data()
        if not data or len(data) < self.min_data_points:
            logger.warning(f"Insufficient data for ML training. Need at least {self.min_data_points} data points.")
            return False
            
        # Extract features
        features, self.feature_names = self._extract_features(data)
        if not features or not features[0]:
            logger.warning("Failed to extract features for ML training")
            return False
            
        # Normalize features
        normalized_features, self.feature_scalers = self._normalize_features(features)
        
        # Train model based on backend
        if self.ml_backend == 'isolation_forest':
            success = self._train_isolation_forest(normalized_features)
        elif self.ml_backend == 'dbscan':
            success = self._train_dbscan(normalized_features)
        elif self.ml_backend == 'local_outlier_factor':
            success = self._train_local_outlier_factor(normalized_features)
        else:
            logger.warning(f"Unsupported ML backend: {self.ml_backend}")
            return False
            
        if success:
            self.is_trained = True
            logger.info(f"Successfully trained {self.ml_backend} model with {len(data)} data points")
            return True
        else:
            logger.warning(f"Failed to train {self.ml_backend} model")
            return False
    
    def _train_isolation_forest(self, features: List[List[float]]) -> bool:
        """Train an Isolation Forest model.
        
        Args:
            features: Normalized feature vectors
            
        Returns:
            True if model was successfully trained, False otherwise
        """
        try:
            import numpy as np
            from sklearn.ensemble import IsolationForest
            
            # Convert to numpy array
            X = np.array(features)
            
            # Create and train the model
            self.model = IsolationForest(
                contamination=0.05,  # Assume 5% of data is anomalous
                random_state=42,
                n_estimators=100
            )
            self.model.fit(X)
            
            return True
        except Exception as e:
            logger.exception(f"Error training Isolation Forest model: {e}")
            return False
    
    def _train_dbscan(self, features: List[List[float]]) -> bool:
        """Train a DBSCAN model.
        
        Args:
            features: Normalized feature vectors
            
        Returns:
            True if model was successfully trained, False otherwise
        """
        try:
            import numpy as np
            from sklearn.cluster import DBSCAN
            
            # Convert to numpy array
            X = np.array(features)
            
            # Create and train the model
            self.model = DBSCAN(
                eps=0.5,  # Epsilon parameter
                min_samples=5  # Minimum samples in a cluster
            )
            self.model.fit(X)
            
            return True
        except Exception as e:
            logger.exception(f"Error training DBSCAN model: {e}")
            return False
    
    def _train_local_outlier_factor(self, features: List[List[float]]) -> bool:
        """Train a Local Outlier Factor model.
        
        Args:
            features: Normalized feature vectors
            
        Returns:
            True if model was successfully trained, False otherwise
        """
        try:
            import numpy as np
            from sklearn.neighbors import LocalOutlierFactor
            
            # Convert to numpy array
            X = np.array(features)
            
            # Create and train the model
            self.model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.05,  # Assume 5% of data is anomalous
                novelty=True  # Enable predict method
            )
            self.model.fit(X)
            
            return True
        except Exception as e:
            logger.exception(f"Error training Local Outlier Factor model: {e}")
            return False
    
    def _get_training_data(self) -> List[Dict[str, Any]]:
        """Get data for training the anomaly detection model.
        
        Returns:
            List of data points for training
        """
        # Get line items with their amounts, contexts, and other features
        query = text("""
            SELECT 
                li.item_id,
                li.doc_id,
                li.description,
                li.amount,
                li.quantity,
                li.unit_price,
                li.cost_code,
                d.doc_type,
                d.party,
                d.date_created
            FROM 
                line_items li
            JOIN
                documents d ON li.doc_id = d.doc_id
            WHERE
                li.amount IS NOT NULL
                AND li.amount > 0
            ORDER BY
                RANDOM()
            LIMIT 1000
        """)
        
        try:
            items = self.db_session.execute(query).fetchall()
        except Exception as e:
            logger.exception(f"Error getting training data: {e}")
            return []
        
        # Convert to dictionaries for easier handling
        item_dicts = []
        for item in items:
            item_id, doc_id, description, amount, quantity, unit_price, cost_code, doc_type, party, date_created = item
            
            item_dict = {
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': amount,
                'quantity': quantity,
                'unit_price': unit_price,
                'cost_code': cost_code,
                'doc_type': doc_type,
                'party': party,
                'date': date_created
            }
            
            item_dicts.append(item_dict)
        
        return item_dicts
    
    def _get_data_to_analyze(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get data to analyze for anomalies.
        
        Args:
            doc_id: Optional document ID to limit scope
            
        Returns:
            List of data points to analyze
        """
        # Get line items
        query = text("""
            SELECT 
                li.item_id,
                li.doc_id,
                li.description,
                li.amount,
                li.quantity,
                li.unit_price,
                li.cost_code,
                d.doc_type,
                d.party,
                d.date_created
            FROM 
                line_items li
            JOIN
                documents d ON li.doc_id = d.doc_id
            WHERE
                li.amount IS NOT NULL
                AND li.amount > 0
                :doc_filter
        """)
        
        doc_filter = "AND li.doc_id = :doc_id" if doc_id else ""
        params = {"doc_id": doc_id} if doc_id else {}
        
        try:
            items = self.db_session.execute(query, params).fetchall()
        except Exception as e:
            logger.exception(f"Error getting data to analyze: {e}")
            return []
        
        # Convert to dictionaries for easier handling
        item_dicts = []
        for item in items:
            item_id, doc_id, description, amount, quantity, unit_price, cost_code, doc_type, party, date_created = item
            
            item_dict = {
                'item_id': item_id,
                'doc_id': doc_id,
                'description': description,
                'amount': amount,
                'quantity': quantity,
                'unit_price': unit_price,
                'cost_code': cost_code,
                'doc_type': doc_type,
                'party': party,
                'date': date_created
            }
            
            item_dicts.append(item_dict)
        
        return item_dicts
    
    def _extract_features(self, data: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[str]]:
        """Extract feature vectors from data points.
        
        Args:
            data: List of data points
            
        Returns:
            Tuple of (feature vectors, feature names)
        """
        try:
            # Define features to extract
            features = []
            feature_names = [
                'amount',
                'has_quantity',
                'has_unit_price',
                'description_length',
                'is_round_amount',
                'day_of_month'
            ]
            
            for item in data:
                # Extract basic features
                amount = float(item['amount']) if item['amount'] is not None else 0.0
                has_quantity = 1.0 if item['quantity'] is not None else 0.0
                has_unit_price = 1.0 if item['unit_price'] is not None else 0.0
                description_length = float(len(item['description'] or ''))
                
                # Calculate derived features
                is_round_amount = 0.0
                if amount % 1000 == 0 or amount % 5000 == 0 or amount % 10000 == 0:
                    is_round_amount = 1.0
                
                day_of_month = float(item['date'].day) if item['date'] else 15.0
                
                # Combine features
                feature_vector = [
                    amount,
                    has_quantity,
                    has_unit_price,
                    description_length,
                    is_round_amount,
                    day_of_month
                ]
                
                features.append(feature_vector)
            
            return features, feature_names
        except Exception as e:
            logger.exception(f"Error extracting features: {e}")
            return [], []
    
    def _normalize_features(self, features: List[List[float]]) -> Tuple[List[List[float]], Dict[str, Any]]:
        """Normalize feature vectors for ML processing.
        
        Args:
            features: Raw feature vectors
            
        Returns:
            Tuple of (normalized features, feature scalers)
        """
        try:
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            
            # Convert to numpy array
            X = np.array(features)
            
            # Create scaler
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X)
            
            # Store scaler for later use
            feature_scalers = {'standard_scaler': scaler}
            
            # Convert back to list
            normalized_features = X_normalized.tolist()
            
            return normalized_features, feature_scalers
        except Exception as e:
            logger.exception(f"Error normalizing features: {e}")
            return features, {}  # Return raw features if normalization fails
    
    def _predict_anomalies(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict anomalies in the data using the trained model.
        
        Args:
            data: Data points to analyze
            
        Returns:
            List of anomalies
        """
        try:
            # Extract and normalize features
            features, _ = self._extract_features(data)
            if not features:
                return []
                
            import numpy as np
            X = np.array(features)
            
            # Apply stored scalers
            if 'standard_scaler' in self.feature_scalers:
                X = self.feature_scalers['standard_scaler'].transform(X)
            
            # Predict anomalies based on backend
            anomaly_scores = []
            
            if self.ml_backend == 'isolation_forest':
                # Predict anomaly scores (-1 for anomalies, 1 for normal)
                raw_scores = self.model.decision_function(X)
                # Convert to anomaly score (higher is more anomalous)
                anomaly_scores = [float(1.0 - (score + 1) / 2) for score in raw_scores]
                
            elif self.ml_backend == 'dbscan':
                # Predict cluster labels (-1 for outliers, >= 0 for clusters)
                labels = self.model.fit_predict(X)
                # Convert to anomaly score (1.0 for outliers, 0.0 for inliers)
                anomaly_scores = [float(1.0 if label == -1 else 0.0) for label in labels]
                
            elif self.ml_backend == 'local_outlier_factor':
                # Predict anomaly scores (negative is more anomalous)
                raw_scores = self.model.decision_function(X)
                # Convert to anomaly score (higher is more anomalous)
                anomaly_scores = [float(max(0, min(1, -score / 2))) for score in raw_scores]
            
            # Filter anomalies by confidence threshold
            anomalies = []
            for i, score in enumerate(anomaly_scores):
                if score >= self.min_confidence:
                    item = data[i]
                    feature_vec = features[i]
                    
                    # Format item for output
                    anomaly = {
                        'type': 'ml_anomaly',
                        'ml_backend': self.ml_backend,
                        'item_id': item['item_id'],
                        'doc_id': item['doc_id'],
                        'amount': float(item['amount']) if item['amount'] is not None else None,
                        'description': item['description'],
                        'cost_code': item['cost_code'],
                        'doc_type': item['doc_type'],
                        'party': item['party'],
                        'date': item['date'].isoformat() if item['date'] else None,
                        'anomaly_score': score,
                        'confidence': score,
                        'features': {name: feature_vec[i] for i, name in enumerate(self.feature_names)},
                        'explanation': f"Machine learning model ({self.ml_backend}) detected anomaly with score {score:.2f}"
                    }
                    
                    # Add reason based on feature analysis
                    anomaly['reasons'] = self._explain_anomaly(feature_vec)
                    
                    anomalies.append(anomaly)
            
            return anomalies
        except Exception as e:
            logger.exception(f"Error predicting anomalies: {e}")
            return []
    
    def _explain_anomaly(self, features: List[float]) -> List[str]:
        """Generate explanations for why a data point was flagged as anomalous.
        
        Args:
            features: Feature vector of the anomalous data point
            
        Returns:
            List of explanation strings
        """
        reasons = []
        
        # Check each feature for unusual values
        amount = features[0]
        description_length = features[3]
        is_round = features[4] > 0.5
        day_of_month = features[5]
        
        if amount > 100000:
            reasons.append("Unusually large amount")
        
        if description_length < 5:
            reasons.append("Very short description")
        
        if is_round and amount > 10000:
            reasons.append("Suspiciously round large amount")
        
        if day_of_month >= 28:
            reasons.append("End-of-month transaction")
        
        # If no specific reasons found, give a generic explanation
        if not reasons:
            reasons.append("Unusual combination of features detected by ML model")
        
        return reasons