# Database Schema Specification

## Overview

The Database Schema is the foundation of the Construction Document Analysis System, designed to store, organize, and enable efficient querying of all document content and relationships. The schema must support complex financial forensics while maintaining document traceability and evidence integrity.

## Key Requirements

1. Store document metadata and content
2. Track financial line items across documents
3. Maintain relationships between documents
4. Support evidence chains and citations
5. Enable pattern detection across document sets
6. Preserve context and document history
7. Support both structured and unstructured data
8. Maintain data integrity and audit trails

## Schema Design

### Core Tables

#### 1. Documents

Stores metadata about each document in the system.

```sql
CREATE TABLE documents (
    doc_id VARCHAR(40) PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    file_size INTEGER NOT NULL,
    file_type VARCHAR(20) NOT NULL,
    doc_type VARCHAR(50),
    party VARCHAR(50),
    date_created DATE,
    date_received DATE,
    date_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_by VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB,
    UNIQUE (file_hash)
);
```

#### 2. Pages

Stores information about individual pages within documents.

```sql
CREATE TABLE pages (
    page_id SERIAL PRIMARY KEY,
    doc_id VARCHAR(40) REFERENCES documents(doc_id),
    page_number INTEGER NOT NULL,
    content TEXT,
    has_tables BOOLEAN DEFAULT FALSE,
    has_handwriting BOOLEAN DEFAULT FALSE,
    has_financial_data BOOLEAN DEFAULT FALSE,
    embedding VECTOR(1536),
    UNIQUE (doc_id, page_number)
);
```

#### 3. Line Items

Stores financial line items extracted from documents.

```sql
CREATE TABLE line_items (
    item_id SERIAL PRIMARY KEY,
    doc_id VARCHAR(40) REFERENCES documents(doc_id),
    page_id INTEGER REFERENCES pages(page_id),
    description TEXT,
    amount DECIMAL(15,2),
    quantity DECIMAL(10,2),
    unit_price DECIMAL(15,2),
    total DECIMAL(15,2),
    cost_code VARCHAR(50),
    category VARCHAR(100),
    status VARCHAR(50),
    location_in_doc JSONB,
    extraction_confidence DECIMAL(5,2),
    context TEXT,
    metadata JSONB
);
```

#### 4. Document Relationships

Tracks relationships between documents.

```sql
CREATE TABLE document_relationships (
    relationship_id SERIAL PRIMARY KEY,
    source_doc_id VARCHAR(40) REFERENCES documents(doc_id),
    target_doc_id VARCHAR(40) REFERENCES documents(doc_id),
    relationship_type VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,2),
    metadata JSONB,
    UNIQUE (source_doc_id, target_doc_id, relationship_type)
);
```

#### 5. Financial Transactions

Tracks financial transactions across documents.

```sql
CREATE TABLE financial_transactions (
    transaction_id SERIAL PRIMARY KEY,
    item_id INTEGER REFERENCES line_items(item_id),
    transaction_type VARCHAR(50) NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    date DATE,
    status VARCHAR(50),
    metadata JSONB
);
```

#### 6. Annotations

Stores annotations extracted from documents.

```sql
CREATE TABLE annotations (
    annotation_id SERIAL PRIMARY KEY,
    doc_id VARCHAR(40) REFERENCES documents(doc_id),
    page_id INTEGER REFERENCES pages(page_id),
    content TEXT,
    location_x DECIMAL(10,2),
    location_y DECIMAL(10,2),
    width DECIMAL(10,2),
    height DECIMAL(10,2),
    annotation_type VARCHAR(50),
    is_handwritten BOOLEAN,
    confidence DECIMAL(5,2),
    metadata JSONB
);
```

#### 7. Analysis Flags

Stores flags for suspicious patterns or anomalies.

```sql
CREATE TABLE analysis_flags (
    flag_id SERIAL PRIMARY KEY,
    item_id INTEGER REFERENCES line_items(item_id),
    flag_type VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,2),
    explanation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB
);
```

#### 8. Reports

Stores generated reports.

```sql
CREATE TABLE reports (
    report_id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    content TEXT,
    format VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    parameters JSONB
);
```

#### 9. Report Evidence

Links reports to their supporting evidence.

```sql
CREATE TABLE report_evidence (
    evidence_id SERIAL PRIMARY KEY,
    report_id INTEGER REFERENCES reports(report_id),
    doc_id VARCHAR(40) REFERENCES documents(doc_id),
    item_id INTEGER REFERENCES line_items(item_id),
    page_id INTEGER REFERENCES pages(page_id),
    annotation_id INTEGER REFERENCES annotations(annotation_id),
    citation_text TEXT,
    relevance_score DECIMAL(5,2),
    metadata JSONB
);
```

### Specialized Tables

#### 10. Amount Matches

Tracks matching amounts across different documents.

```sql
CREATE TABLE amount_matches (
    match_id SERIAL PRIMARY KEY,
    source_item_id INTEGER REFERENCES line_items(item_id),
    target_item_id INTEGER REFERENCES line_items(item_id),
    match_type VARCHAR(50),
    confidence DECIMAL(5,2),
    difference DECIMAL(15,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    UNIQUE (source_item_id, target_item_id)
);
```

#### 11. Change Orders

Specialized tracking for change orders.

```sql
CREATE TABLE change_orders (
    change_order_id SERIAL PRIMARY KEY,
    doc_id VARCHAR(40) REFERENCES documents(doc_id),
    change_order_number VARCHAR(50),
    description TEXT,
    amount DECIMAL(15,2),
    status VARCHAR(50),
    date_submitted DATE,
    date_responded DATE,
    reason_code VARCHAR(50),
    metadata JSONB
);
```

#### 12. Payment Applications

Specialized tracking for payment applications.

```sql
CREATE TABLE payment_applications (
    payment_app_id SERIAL PRIMARY KEY,
    doc_id VARCHAR(40) REFERENCES documents(doc_id),
    payment_app_number VARCHAR(50),
    period_start DATE,
    period_end DATE,
    amount_requested DECIMAL(15,2),
    amount_approved DECIMAL(15,2),
    status VARCHAR(50),
    metadata JSONB
);
```

## SQLAlchemy Models

### Core Models

Below are the SQLAlchemy model definitions corresponding to the schema above:

```python
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    doc_id = Column(String(40), primary_key=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(20), nullable=False)
    doc_type = Column(String(50))
    party = Column(String(50))
    date_created = Column(DateTime)
    date_received = Column(DateTime)
    date_processed = Column(DateTime, default=datetime.utcnow)
    processed_by = Column(String(100))
    status = Column(String(20), default='active')
    metadata = Column(JSON)
    
    # Relationships
    pages = relationship("Page", back_populates="document")
    line_items = relationship("LineItem", back_populates="document")
    
    def __repr__(self):
        return f"<Document(doc_id='{self.doc_id}', doc_type='{self.doc_type}', party='{self.party}')>"

class Page(Base):
    __tablename__ = 'pages'
    
    page_id = Column(Integer, primary_key=True)
    doc_id = Column(String(40), ForeignKey('documents.doc_id'))
    page_number = Column(Integer, nullable=False)
    content = Column(Text)
    has_tables = Column(Boolean, default=False)
    has_handwriting = Column(Boolean, default=False)
    has_financial_data = Column(Boolean, default=False)
    
    # Relationships
    document = relationship("Document", back_populates="pages")
    line_items = relationship("LineItem", back_populates="page")
    annotations = relationship("Annotation", back_populates="page")
    
    def __repr__(self):
        return f"<Page(doc_id='{self.doc_id}', page_number={self.page_number})>"

class LineItem(Base):
    __tablename__ = 'line_items'
    
    item_id = Column(Integer, primary_key=True)
    doc_id = Column(String(40), ForeignKey('documents.doc_id'))
    page_id = Column(Integer, ForeignKey('pages.page_id'))
    description = Column(Text)
    amount = Column(Numeric(15, 2))
    quantity = Column(Numeric(10, 2))
    unit_price = Column(Numeric(15, 2))
    total = Column(Numeric(15, 2))
    cost_code = Column(String(50))
    category = Column(String(100))
    status = Column(String(50))
    location_in_doc = Column(JSON)
    extraction_confidence = Column(Numeric(5, 2))
    context = Column(Text)
    metadata = Column(JSON)
    
    # Relationships
    document = relationship("Document", back_populates="line_items")
    page = relationship("Page", back_populates="line_items")
    flags = relationship("AnalysisFlag", back_populates="line_item")
    
    def __repr__(self):
        return f"<LineItem(doc_id='{self.doc_id}', description='{self.description[:20]}...', amount={self.amount})>"

class DocumentRelationship(Base):
    __tablename__ = 'document_relationships'
    
    relationship_id = Column(Integer, primary_key=True)
    source_doc_id = Column(String(40), ForeignKey('documents.doc_id'))
    target_doc_id = Column(String(40), ForeignKey('documents.doc_id'))
    relationship_type = Column(String(50), nullable=False)
    confidence = Column(Numeric(5, 2))
    metadata = Column(JSON)
    
    # Relationships
    source_document = relationship("Document", foreign_keys=[source_doc_id])
    target_document = relationship("Document", foreign_keys=[target_doc_id])
    
    def __repr__(self):
        return f"<DocumentRelationship(source='{self.source_doc_id}', target='{self.target_doc_id}', type='{self.relationship_type}')>"

class FinancialTransaction(Base):
    __tablename__ = 'financial_transactions'
    
    transaction_id = Column(Integer, primary_key=True)
    item_id = Column(Integer, ForeignKey('line_items.item_id'))
    transaction_type = Column(String(50), nullable=False)
    amount = Column(Numeric(15, 2), nullable=False)
    date = Column(DateTime)
    status = Column(String(50))
    metadata = Column(JSON)
    
    # Relationships
    line_item = relationship("LineItem")
    
    def __repr__(self):
        return f"<FinancialTransaction(id={self.transaction_id}, type='{self.transaction_type}', amount={self.amount})>"

class Annotation(Base):
    __tablename__ = 'annotations'
    
    annotation_id = Column(Integer, primary_key=True)
    doc_id = Column(String(40), ForeignKey('documents.doc_id'))
    page_id = Column(Integer, ForeignKey('pages.page_id'))
    content = Column(Text)
    location_x = Column(Numeric(10, 2))
    location_y = Column(Numeric(10, 2))
    width = Column(Numeric(10, 2))
    height = Column(Numeric(10, 2))
    annotation_type = Column(String(50))
    is_handwritten = Column(Boolean)
    confidence = Column(Numeric(5, 2))
    metadata = Column(JSON)
    
    # Relationships
    page = relationship("Page", back_populates="annotations")
    
    def __repr__(self):
        return f"<Annotation(doc_id='{self.doc_id}', content='{self.content[:20]}...', is_handwritten={self.is_handwritten})>"

class AnalysisFlag(Base):
    __tablename__ = 'analysis_flags'
    
    flag_id = Column(Integer, primary_key=True)
    item_id = Column(Integer, ForeignKey('line_items.item_id'))
    flag_type = Column(String(50), nullable=False)
    confidence = Column(Numeric(5, 2))
    explanation = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100))
    status = Column(String(20), default='active')
    metadata = Column(JSON)
    
    # Relationships
    line_item = relationship("LineItem", back_populates="flags")
    
    def __repr__(self):
        return f"<AnalysisFlag(item_id={self.item_id}, type='{self.flag_type}', confidence={self.confidence})>"
```

## Database Operations

Common database operations will include:

### Document Registration

```python
def register_document(session, file_path, doc_type=None, party=None, metadata=None):
    """Register a document in the database.
    
    Args:
        session: SQLAlchemy session
        file_path: Path to the document file
        doc_type: Document type (payment_app, change_order, etc.)
        party: Document party (district, contractor)
        metadata: Additional metadata
        
    Returns:
        Document object
    """
    # Calculate file hash
    file_hash = calculate_file_hash(file_path)
    
    # Check if document already exists
    existing_doc = session.query(Document).filter(Document.file_hash == file_hash).first()
    if existing_doc:
        return existing_doc
    
    # Create new document
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    file_type = os.path.splitext(file_path)[1].lower()[1:]
    
    doc = Document(
        doc_id=generate_document_id(),
        file_name=file_name,
        file_path=file_path,
        file_hash=file_hash,
        file_size=file_size,
        file_type=file_type,
        doc_type=doc_type,
        party=party,
        date_processed=datetime.utcnow(),
        metadata=metadata or {}
    )
    
    session.add(doc)
    session.flush()
    
    return doc
```

### Storing Line Items

```python
def store_line_items(session, doc_id, line_items):
    """Store line items for a document.
    
    Args:
        session: SQLAlchemy session
        doc_id: Document ID
        line_items: List of line item dictionaries
        
    Returns:
        List of LineItem objects
    """
    result = []
    
    for item_data in line_items:
        line_item = LineItem(
            doc_id=doc_id,
            page_id=item_data.get('page_id'),
            description=item_data.get('description'),
            amount=item_data.get('amount'),
            quantity=item_data.get('quantity'),
            unit_price=item_data.get('unit_price'),
            total=item_data.get('total'),
            cost_code=item_data.get('cost_code'),
            category=item_data.get('category'),
            status=item_data.get('status'),
            location_in_doc=item_data.get('location_in_doc'),
            extraction_confidence=item_data.get('confidence', 1.0),
            context=item_data.get('context'),
            metadata=item_data.get('metadata', {})
        )
        
        session.add(line_item)
        result.append(line_item)
    
    session.flush()
    return result
```

### Finding Matching Amounts

```python
def find_matching_amounts(session, amount, tolerance=0.01):
    """Find line items with matching amounts.
    
    Args:
        session: SQLAlchemy session
        amount: Amount to match
        tolerance: Tolerance for matching
        
    Returns:
        List of matching LineItem objects
    """
    lower_bound = amount - tolerance
    upper_bound = amount + tolerance
    
    matches = session.query(LineItem).filter(
        LineItem.amount.between(lower_bound, upper_bound)
    ).all()
    
    return matches
```

### Finding Document Relationships

```python
def find_related_documents(session, doc_id, relationship_type=None):
    """Find documents related to the given document.
    
    Args:
        session: SQLAlchemy session
        doc_id: Document ID
        relationship_type: Optional relationship type filter
        
    Returns:
        List of related Document objects
    """
    query = session.query(Document).join(
        DocumentRelationship,
        DocumentRelationship.target_doc_id == Document.doc_id
    ).filter(
        DocumentRelationship.source_doc_id == doc_id
    )
    
    if relationship_type:
        query = query.filter(DocumentRelationship.relationship_type == relationship_type)
    
    return query.all()
```

## Migrations and Versioning

The database schema will evolve over time, requiring a migration strategy:

```python
# Using Alembic for migrations
def create_migration(message):
    """Create a new migration script."""
    os.system(f"alembic revision --autogenerate -m '{message}'")

def upgrade_database():
    """Upgrade database to latest version."""
    os.system("alembic upgrade head")

def downgrade_database(revision):
    """Downgrade database to specific revision."""
    os.system(f"alembic downgrade {revision}")
```

## Database Configuration

```python
def get_database_engine(config):
    """Get SQLAlchemy engine based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SQLAlchemy engine
    """
    db_type = config.get('db_type', 'sqlite')
    
    if db_type == 'sqlite':
        db_path = config.get('db_path', 'construction_analysis.db')
        connection_string = f"sqlite:///{db_path}"
    elif db_type == 'postgresql':
        host = config.get('db_host', 'localhost')
        port = config.get('db_port', 5432)
        database = config.get('db_name', 'construction_analysis')
        user = config.get('db_user', 'postgres')
        password = config.get('db_password', '')
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    
    return create_engine(connection_string)
```

## Implementation Considerations

1. **Initial Deployment**: Start with SQLite for simplicity, then migrate to PostgreSQL for production
2. **Performance**: Create appropriate indexes for common query patterns
3. **Scalability**: Design for potential growth in document volume
4. **Data Integrity**: Implement proper constraints and validation
5. **Security**: Ensure sensitive data is properly protected

## Security Considerations

1. **Authentication**: Implement proper authentication for database access
2. **Encryption**: Encrypt sensitive data at rest
3. **Backup**: Implement regular database backups
4. **Audit Trails**: Track all changes to the database

## Testing Strategy

1. **Schema Tests**: Verify schema integrity and constraints
2. **Migration Tests**: Ensure migrations work correctly
3. **Performance Tests**: Benchmark common queries
4. **Integration Tests**: Test database operations with actual documents
