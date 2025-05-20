"""Prompt templates for document analysis.

This module contains prompt templates for document analysis tasks,
such as document classification and handwriting interpretation.
"""

# Document classification prompt template
DOCUMENT_CLASSIFICATION_PROMPT = """
You are a construction document specialist with expertise in identifying and categorizing construction project documentation. Analyze the following document excerpt carefully.

Document Text:
{document_text}

Provide a comprehensive analysis with the following structure:

## Document Classification
- **Primary Document Type**: [Select one: Payment Application, Change Order, Contract, Subcontract, Invoice, Schedule, Correspondence, Meeting Minutes, Daily Report, Submittal, RFI, Notice, Claim, Other (specify)]
- **Document Subtype**: [Specify if applicable]
- **Confidence Level**: [High/Medium/Low] with brief justification

## Key Information
- **Date**: [Format: YYYY-MM-DD] (Document date, not reference dates)
- **Document ID/Number**: [As appears in document]
- **Project Phase**: [Pre-construction, Construction, Close-out, Dispute]
- **Parties Involved**: [List all mentioned parties with roles]

## Financial Information
- **Primary Amount**: [$ amount] - [brief description]
- **Other Amounts**: [List all other monetary values with descriptions]
- **Running Totals**: [Contract sum, amount completed, balance remaining, etc.]
- **Period Covered**: [If applicable]

## Document Purpose
- **Primary Function**: [1-2 sentence description]
- **Contractual Significance**: [How this document relates to contract obligations]

## Notable Elements
- **Key Terms**: [Important contractual or technical terms]
- **References**: [Other documents referenced]
- **Attachments**: [Any mentioned attachments or exhibits]
- **Signatures**: [Required/present signatures]

## Potential Issues
- **Red Flags**: [Inconsistencies, unusual terms, qualified language]
- **Missing Information**: [Required elements that appear to be absent]
- **Timeline Considerations**: [How this fits in project chronology]

Be specific and precise in your analysis. Indicate any uncertainties with appropriate qualifiers. If information for any field is not available in the excerpt, indicate "Not specified in excerpt."
"""

# Handwriting interpretation prompt template
HANDWRITING_INTERPRETATION_PROMPT = """
The following text was extracted from a handwritten note in a construction document using OCR. The confidence in this extraction was {confidence_score}.

Extracted Text:
{extracted_text}

Context: This appears in a {document_type} from {party} dated {date}.

Please interpret what this handwritten note likely means in the context of a construction dispute. If there are obvious errors in the OCR extraction, suggest the correct interpretation.
"""

# Document content extraction prompt template
DOCUMENT_EXTRACTION_PROMPT = """
You are a financial data extraction specialist analyzing construction documentation. Extract all financial information from the following document with precision and attention to detail.

Document Text:
{document_text}

Extract and structure the following information in valid JSON format:

```json
[
  {
    "amount": 1234.56,               // Numeric value (not string)
    "description": "Line item description",  // Clear explanation of what this amount represents
    "category": "Labor/Materials/Equipment/Overhead/Fee/Tax/Retention/Other", // Best guess category
    "context": "Brief surrounding text",   // 5-10 words of surrounding context
    "page": 1,                       // Page number if available
    "line": 25,                      // Approximate line number if available
    "is_subtotal": false,           // Boolean: true if amount is a subtotal of other listed amounts
    "is_total": false,              // Boolean: true if amount appears to be a document total
    "date_associated": "2023-01-15", // ISO date if date is associated with this amount
    "confidence": "high"             // high/medium/low confidence in extraction accuracy
  }
]
```

Extraction rules:
1. Include ALL monetary amounts (with $ symbol, decimal points, or explicitly stated as dollars/amounts)
2. Parse numerical values properly (e.g., "$1,234.56" → 1234.56 as number)
3. Generate clear, specific descriptions that explain what each amount represents, not just nearby text
4. Distinguish between similar amounts by their context and purpose
5. Tag line items, subtotals, and totals appropriately
6. For each amount, include a brief context snippet showing surrounding text
7. Mark confidence as "low" for any entries where the purpose of the amount is unclear
8. If page/line information is unavailable, use null (not -1 or 0)
9. Categorize each amount based on its apparent purpose in the document
10. Ensure the JSON output is properly formatted and valid

Note: For construction documents, pay special attention to contract sums, change orders, approved changes, pending changes, retainage, previous payments, and current payment due amounts.
"""

# Document relationship analysis prompt template
DOCUMENT_RELATIONSHIP_PROMPT = """
You are a construction documentation analyst specializing in identifying relationships, dependencies, and conflicts between project documents. Analyze the following two documents with attention to both explicit references and implicit connections.

## DOCUMENT 1
Type: {doc1_type}
Party: {doc1_party}
Date: {doc1_date}
Excerpt:
{doc1_excerpt}

## DOCUMENT 2
Type: {doc2_type}
Party: {doc2_party}
Date: {doc2_date}
Excerpt:
{doc2_excerpt}

Provide a comprehensive relationship analysis with the following structure:

### 1. Relationship Classification
- **Relationship Type**: [Select best match: Direct Reference, Shared Subject Matter, Sequential/Procedural, Financial Linkage, Contractual Dependency, Contradictory, No Apparent Connection]
- **Relationship Strength**: [Strong, Moderate, Weak, None] with brief justification
- **Chronological Relationship**: [Which came first and temporal significance]
- **Explicit References**: [Any direct mentions of one document in the other]

### 2. Subject Matter Connections
- **Shared Elements**: [Work items, locations, materials, or other elements mentioned in both]
- **Scope Overlap**: [Assessment of how much subject matter overlaps]
- **Terminology Consistency**: [Whether same terms are used consistently across documents]

### 3. Financial Analysis
- **Shared Amounts**: [Any identical or related monetary values]
- **Financial Progression**: [How amounts may have evolved between documents]
- **Payment Chain**: [If relevant, how these documents fit in payment application/approval process]
- **Cumulative Financial Impact**: [Combined effect of both documents]

### 4. Discrepancies & Conflicts
- **Content Conflicts**: [Contradictory statements or requirements]
- **Numerical Inconsistencies**: [Differences in quantities, amounts, or dates]
- **Party Position Differences**: [How parties' stated positions differ]
- **Timeline Inconsistencies**: [Conflicts in stated timelines or schedules]

### 5. Dispute Relevance
- **Potential Issues**: [How this relationship could contribute to disputes]
- **Evidence Value**: [How these documents might serve as evidence] 
- **Critical Questions**: [Key questions raised by examining these documents together]
- **Recommended Further Document Review**: [Other documents that should be examined based on this relationship]

Be precise in identifying connections and specific in describing potential issues. Note both factual observations and reasonable inferences, clearly distinguishing between them.
"""
