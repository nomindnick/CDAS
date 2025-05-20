"""
Database models for the Construction Document Analysis System.

This module defines the SQLAlchemy models that represent the database schema
for storing documents, pages, line items, and other related data.
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, 
    Text, JSON, Numeric, Date, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, UTC
import uuid
import os

Base = declarative_base()


class Document(Base):
    """Document model for storing document metadata."""
    
    __tablename__ = 'documents'
    
    doc_id = Column(String(40), primary_key=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(20), nullable=False)
    doc_type = Column(String(50))
    party = Column(String(50))
    date_created = Column(Date)
    date_received = Column(Date)
    date_processed = Column(DateTime, default=lambda: datetime.now(UTC))
    processed_by = Column(String(100))
    status = Column(String(20), default='active')
    meta_data = Column(JSON)
    
    # Relationships
    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")
    line_items = relationship("LineItem", back_populates="document", cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_documents_doc_type', 'doc_type'),
        Index('idx_documents_party', 'party'),
        Index('idx_documents_status', 'status'),
    )
    
    def __repr__(self):
        return f"<Document(doc_id='{self.doc_id}', doc_type='{self.doc_type}', party='{self.party}')>"


class Page(Base):
    """Page model for storing page content and metadata."""
    
    __tablename__ = 'pages'
    
    page_id = Column(Integer, primary_key=True)
    doc_id = Column(String(40), ForeignKey('documents.doc_id', ondelete='CASCADE'), nullable=False)
    page_number = Column(Integer, nullable=False)
    content = Column(Text)
    has_tables = Column(Boolean, default=False)
    has_handwriting = Column(Boolean, default=False)
    has_financial_data = Column(Boolean, default=False)
    meta_data = Column(JSON)
    
    # Relationships
    document = relationship("Document", back_populates="pages")
    line_items = relationship("LineItem", back_populates="page", cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="page", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('doc_id', 'page_number', name='uq_page_doc_number'),
        Index('idx_pages_has_tables', 'has_tables'),
        Index('idx_pages_has_handwriting', 'has_handwriting'),
        Index('idx_pages_has_financial_data', 'has_financial_data'),
    )
    
    def __repr__(self):
        return f"<Page(doc_id='{self.doc_id}', page_number={self.page_number})>"


class LineItem(Base):
    """Line item model for storing financial line items from documents."""
    
    __tablename__ = 'line_items'
    
    item_id = Column(String(40), primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id = Column(String(40), ForeignKey('documents.doc_id', ondelete='CASCADE'), nullable=False)
    page_id = Column(Integer, ForeignKey('pages.page_id', ondelete='SET NULL'))
    item_number = Column(String(50))
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
    meta_data = Column(JSON)
    
    # Relationships
    document = relationship("Document", back_populates="line_items")
    page = relationship("Page", back_populates="line_items")
    flags = relationship("AnalysisFlag", back_populates="line_item", cascade="all, delete-orphan")
    transactions = relationship("FinancialTransaction", back_populates="line_item", cascade="all, delete-orphan")
    source_matches = relationship(
        "AmountMatch", 
        back_populates="source_item",
        foreign_keys="AmountMatch.source_item_id",
        cascade="all, delete-orphan"
    )
    target_matches = relationship(
        "AmountMatch", 
        back_populates="target_item",
        foreign_keys="AmountMatch.target_item_id",
        cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_line_items_amount', 'amount'),
        Index('idx_line_items_cost_code', 'cost_code'),
        Index('idx_line_items_category', 'category'),
        Index('idx_line_items_status', 'status'),
    )
    
    def __repr__(self):
        desc = self.description[:20] + "..." if self.description and len(self.description) > 20 else self.description
        return f"<LineItem(doc_id='{self.doc_id}', description='{desc}', amount={self.amount})>"


class DocumentRelationship(Base):
    """Document relationship model for tracking relationships between documents."""
    
    __tablename__ = 'document_relationships'
    
    relationship_id = Column(Integer, primary_key=True)
    source_doc_id = Column(String(40), ForeignKey('documents.doc_id', ondelete='CASCADE'), nullable=False)
    target_doc_id = Column(String(40), ForeignKey('documents.doc_id', ondelete='CASCADE'), nullable=False)
    relationship_type = Column(String(50), nullable=False)
    confidence = Column(Numeric(5, 2))
    meta_data = Column(JSON)
    
    # Relationships
    source_document = relationship("Document", foreign_keys=[source_doc_id])
    target_document = relationship("Document", foreign_keys=[target_doc_id])
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('source_doc_id', 'target_doc_id', 'relationship_type', name='uq_doc_relationship'),
        Index('idx_doc_relationships_type', 'relationship_type'),
    )
    
    def __repr__(self):
        return f"<DocumentRelationship(source='{self.source_doc_id}', target='{self.target_doc_id}', type='{self.relationship_type}')>"


class FinancialTransaction(Base):
    """Financial transaction model for tracking financial transactions across documents."""
    
    __tablename__ = 'financial_transactions'
    
    transaction_id = Column(Integer, primary_key=True)
    item_id = Column(String(40), ForeignKey('line_items.item_id', ondelete='CASCADE'), nullable=False)
    transaction_type = Column(String(50), nullable=False)
    amount = Column(Numeric(15, 2), nullable=False)
    date = Column(Date)
    status = Column(String(50))
    meta_data = Column(JSON)
    
    # Relationships
    line_item = relationship("LineItem", back_populates="transactions")
    
    # Indexes
    __table_args__ = (
        Index('idx_financial_transactions_type', 'transaction_type'),
        Index('idx_financial_transactions_amount', 'amount'),
        Index('idx_financial_transactions_date', 'date'),
        Index('idx_financial_transactions_status', 'status'),
    )
    
    def __repr__(self):
        return f"<FinancialTransaction(id={self.transaction_id}, type='{self.transaction_type}', amount={self.amount})>"


class Annotation(Base):
    """Annotation model for storing annotations extracted from documents."""
    
    __tablename__ = 'annotations'
    
    annotation_id = Column(Integer, primary_key=True)
    doc_id = Column(String(40), ForeignKey('documents.doc_id', ondelete='CASCADE'), nullable=False)
    page_id = Column(Integer, ForeignKey('pages.page_id', ondelete='CASCADE'))
    content = Column(Text)
    location_x = Column(Numeric(10, 2))
    location_y = Column(Numeric(10, 2))
    width = Column(Numeric(10, 2))
    height = Column(Numeric(10, 2))
    annotation_type = Column(String(50))
    is_handwritten = Column(Boolean)
    confidence = Column(Numeric(5, 2))
    meta_data = Column(JSON)
    
    # Relationships
    document = relationship("Document", back_populates="annotations")
    page = relationship("Page", back_populates="annotations")
    report_evidence = relationship("ReportEvidence", back_populates="annotation", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_annotations_type', 'annotation_type'),
        Index('idx_annotations_is_handwritten', 'is_handwritten'),
    )
    
    def __repr__(self):
        content_preview = self.content[:20] + "..." if self.content and len(self.content) > 20 else self.content
        return f"<Annotation(doc_id='{self.doc_id}', content='{content_preview}', is_handwritten={self.is_handwritten})>"


class AnalysisFlag(Base):
    """Analysis flag model for storing flags for suspicious patterns or anomalies."""
    
    __tablename__ = 'analysis_flags'
    
    flag_id = Column(Integer, primary_key=True)
    item_id = Column(String(40), ForeignKey('line_items.item_id', ondelete='CASCADE'), nullable=False)
    flag_type = Column(String(50), nullable=False)
    confidence = Column(Numeric(5, 2))
    explanation = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    created_by = Column(String(100))
    status = Column(String(20), default='active')
    meta_data = Column(JSON)
    
    # Relationships
    line_item = relationship("LineItem", back_populates="flags")
    
    # Indexes
    __table_args__ = (
        Index('idx_analysis_flags_type', 'flag_type'),
        Index('idx_analysis_flags_status', 'status'),
        Index('idx_analysis_flags_confidence', 'confidence'),
    )
    
    def __repr__(self):
        return f"<AnalysisFlag(item_id={self.item_id}, type='{self.flag_type}', confidence={self.confidence})>"


class Report(Base):
    """Report model for storing generated reports."""
    
    __tablename__ = 'reports'
    
    report_id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    content = Column(Text)
    format = Column(String(20))
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    created_by = Column(String(100))
    parameters = Column(JSON)
    
    # Relationships
    evidence = relationship("ReportEvidence", back_populates="report", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_reports_created_at', 'created_at'),
        Index('idx_reports_format', 'format'),
    )
    
    def __repr__(self):
        return f"<Report(id={self.report_id}, title='{self.title}', format='{self.format}')>"


class ReportEvidence(Base):
    """Report evidence model for linking reports to their supporting evidence."""
    
    __tablename__ = 'report_evidence'
    
    evidence_id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey('reports.report_id', ondelete='CASCADE'), nullable=False)
    doc_id = Column(String(40), ForeignKey('documents.doc_id', ondelete='CASCADE'))
    item_id = Column(String(40), ForeignKey('line_items.item_id', ondelete='CASCADE'))
    page_id = Column(Integer, ForeignKey('pages.page_id', ondelete='CASCADE'))
    annotation_id = Column(Integer, ForeignKey('annotations.annotation_id', ondelete='CASCADE'))
    citation_text = Column(Text)
    relevance_score = Column(Numeric(5, 2))
    meta_data = Column(JSON)
    
    # Relationships
    report = relationship("Report", back_populates="evidence")
    document = relationship("Document", foreign_keys=[doc_id])
    line_item = relationship("LineItem", foreign_keys=[item_id])
    page = relationship("Page", foreign_keys=[page_id])
    annotation = relationship("Annotation", back_populates="report_evidence")
    
    def __repr__(self):
        return f"<ReportEvidence(id={self.evidence_id}, report_id={self.report_id}, doc_id='{self.doc_id}')>"


class AmountMatch(Base):
    """Amount match model for tracking matching amounts across different documents."""
    
    __tablename__ = 'amount_matches'
    
    match_id = Column(Integer, primary_key=True)
    source_item_id = Column(String(40), ForeignKey('line_items.item_id', ondelete='CASCADE'), nullable=False)
    target_item_id = Column(String(40), ForeignKey('line_items.item_id', ondelete='CASCADE'), nullable=False)
    match_type = Column(String(50))
    confidence = Column(Numeric(5, 2))
    difference = Column(Numeric(15, 2))
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    meta_data = Column(JSON)
    
    # Relationships
    source_item = relationship("LineItem", foreign_keys=[source_item_id], back_populates="source_matches")
    target_item = relationship("LineItem", foreign_keys=[target_item_id], back_populates="target_matches")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('source_item_id', 'target_item_id', name='uq_amount_match'),
        Index('idx_amount_matches_type', 'match_type'),
        Index('idx_amount_matches_confidence', 'confidence'),
        Index('idx_amount_matches_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<AmountMatch(id={self.match_id}, source={self.source_item_id}, target={self.target_item_id}, confidence={self.confidence})>"


class ChangeOrder(Base):
    """Change order model for specialized tracking of change orders."""
    
    __tablename__ = 'change_orders'
    
    change_order_id = Column(Integer, primary_key=True)
    doc_id = Column(String(40), ForeignKey('documents.doc_id', ondelete='CASCADE'), nullable=False)
    change_order_number = Column(String(50))
    description = Column(Text)
    amount = Column(Numeric(15, 2))
    status = Column(String(50))
    date_submitted = Column(Date)
    date_responded = Column(Date)
    reason_code = Column(String(50))
    meta_data = Column(JSON)
    
    # Relationships
    document = relationship("Document")
    
    # Indexes
    __table_args__ = (
        Index('idx_change_orders_number', 'change_order_number'),
        Index('idx_change_orders_status', 'status'),
        Index('idx_change_orders_amount', 'amount'),
        Index('idx_change_orders_date_submitted', 'date_submitted'),
        Index('idx_change_orders_reason_code', 'reason_code'),
    )
    
    def __repr__(self):
        return f"<ChangeOrder(id={self.change_order_id}, number='{self.change_order_number}', amount={self.amount})>"


class PaymentApplication(Base):
    """Payment application model for specialized tracking of payment applications."""
    
    __tablename__ = 'payment_applications'
    
    payment_app_id = Column(Integer, primary_key=True)
    doc_id = Column(String(40), ForeignKey('documents.doc_id', ondelete='CASCADE'), nullable=False)
    payment_app_number = Column(String(50))
    period_start = Column(Date)
    period_end = Column(Date)
    amount_requested = Column(Numeric(15, 2))
    amount_approved = Column(Numeric(15, 2))
    status = Column(String(50))
    meta_data = Column(JSON)
    
    # Relationships
    document = relationship("Document")
    
    # Indexes
    __table_args__ = (
        Index('idx_payment_apps_number', 'payment_app_number'),
        Index('idx_payment_apps_period_start', 'period_start'),
        Index('idx_payment_apps_period_end', 'period_end'),
        Index('idx_payment_apps_amount_requested', 'amount_requested'),
        Index('idx_payment_apps_amount_approved', 'amount_approved'),
        Index('idx_payment_apps_status', 'status'),
    )
    
    def __repr__(self):
        return f"<PaymentApplication(id={self.payment_app_id}, number='{self.payment_app_number}', requested={self.amount_requested})>"