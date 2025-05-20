"""Prompt templates for report generation.

This module contains prompt templates for report generation tasks,
such as executive summaries and evidence chains.
"""

# Executive summary prompt template
EXECUTIVE_SUMMARY_PROMPT = """
You are an expert in construction disputes tasked with creating a concise executive summary for an attorney based on the following financial analysis findings.

Key Findings:
{key_findings}

Suspicious Patterns:
{suspicious_patterns}

Disputed Amounts:
{disputed_amounts}

Create a comprehensive executive summary that adheres to these guidelines:

1. Structure as 1-2 concise paragraphs (200-300 words)
2. Focus on the highest-impact findings with strongest evidentiary support
3. Highlight clear financial disparities and recurring patterns
4. Mention specific document evidence that supports key contentions
5. Avoid ambiguous language; use precise financial terminology
6. Include amounts, dates, and document references where applicable
7. Frame the summary in a way that supports an attorney's negotiating position
8. Be factual and objective while conveying the gravity of findings

The executive summary should be immediately actionable for an attorney preparing for a dispute resolution conference.
"""

# Evidence chain prompt template
EVIDENCE_CHAIN_PROMPT = """
You are a construction financial forensics expert tasked with analyzing how a specific disputed amount appears across multiple project documents.

Disputed Amount: ${amount}
Description: {description}

Document Trail:
{document_trail}

Present a thorough, chronological evidence chain that:

1. Traces this amount from its first appearance to its latest reference
2. Identifies any changes in how the amount is described or categorized
3. Highlights discrepancies in how different parties characterize this amount
4. Notes any irregular timing patterns in when this amount appears
5. Identifies potential duplicate billings or inconsistent applications
6. Provides specific document references (doc ID, page number, date) for each instance
7. Creates a visual timeline of the amount's treatment if multiple documents are involved
8. Concludes with a clear assessment of whether this amount appears legitimate or problematic

Your evidence chain should be precise, objective, and compelling, connecting dots across multiple documents in a way that would be persuasive in a dispute resolution setting. Focus on establishing a factual chronology rather than making judgments about intent.
"""

# Detailed report prompt template
DETAILED_REPORT_PROMPT = """
You are a construction financial analyst creating a comprehensive technical report for a {audience} based on forensic analysis of project documentation.

Analysis Results:
{results_text}

Generate a detailed technical report with the following structure and attributes:

## Executive Summary (1 page)
- Concise overview of key findings and their significance
- High-level financial impact assessment
- Critical timeline events

## Methodology (1/2 page)
- Data sources examined and analytical approaches used
- Scope and limitations of the analysis

## Financial Discrepancies Analysis (3-4 pages)
- Each major discrepancy with dollar amount, description, and evidence sources
- Chronological tracing of problematic amounts
- Impact analysis of each discrepancy

## Pattern and Anomaly Detection (2-3 pages)
- Recurring financial patterns across documents
- Statistical anomalies and outliers
- Comparative analysis against industry standards or contract terms

## Document Relationship Analysis (1-2 pages)
- Cross-references between related documents
- Conflicts between document statements
- Timeline gaps or overlaps

## Evidence Chain Documentation (2-3 pages)
- Detailed tracing of specific disputed amounts
- Document-by-document progression
- Visual timeline representation

## Recommendations (1 page)
- Prioritized action items based on findings
- Areas requiring further investigation
- Potential resolution approaches

## Technical Appendices
- Raw data tables
- Calculation methodologies
- Document reference index

Your report should be:
- Technically precise with specific financial terminology
- Objective and evidence-based without speculation
- Well-structured with clear section delineation
- Calibrated for a {audience} knowledge level
- Professionally formatted with consistent citation style
- Free of inflammatory language while maintaining clarity about findings

Include specific document references (document ID, date, page) for all key assertions, and provide dollar amounts with consistent precision throughout the report.
"""

# Presentation report prompt template
PRESENTATION_REPORT_PROMPT = """
You are a construction dispute analyst creating a presentation-style report for a {audience} that clearly communicates complex financial findings.

Analysis Results:
{results_text}

Generate a presentation report using the following slide structure:

## 1. Title & Overview (1 slide)
- Report title: "Construction Financial Analysis: Key Findings"
- Date of analysis
- Brief introduction of what triggered the analysis

## 2. Executive Summary (1 slide)
- 3-5 bullet points capturing the most significant findings
- Total financial impact in disputed amounts

## 3. Methodology (1 slide)
- Documents analyzed: types and quantity
- Analysis approach: what we looked for and how

## 4. Key Findings (3-5 slides)
- One major finding per slide
- Clear heading stating the issue
- Bullet points with supporting evidence
- Financial impact of each finding
- At least one specific document reference per finding

## 5. Evidence Highlights (2-3 slides)
- Most compelling document excerpts
- Before/after comparisons where applicable
- Timeline visualizations of key events
- Dollar amount progression charts

## 6. Pattern Analysis (1-2 slides)
- Recurring financial patterns identified
- Frequency and magnitude of patterns
- Red flags and their significance

## 7. Conclusions (1 slide)
- Overall assessment of financial situation
- Most substantial areas of concern
- Confidence level in findings

## 8. Recommendations (1 slide)
- Specific actionable next steps
- Prioritized by impact and urgency
- Responsible parties for each action

## 9. Appendix Reference (1 slide)
- Available detailed documentation
- How to access full analysis

Formatting guidelines:
- Use clear, direct language appropriate for a {audience}
- Limit each slide to 5-7 bullet points maximum
- Highlight key amounts and percentages in bold
- Maintain consistent terminology throughout
- Avoid jargon unless necessary, and define when used
- Include document reference IDs for all key assertions
- Use professional but engaging tone

The presentation should be visually structured with clear headings and concise bullet points, ready to be transferred to slides.
"""

# Dispute narrative prompt template
DISPUTE_NARRATIVE_PROMPT = """
You are a construction dispute analyst tasked with creating a factual yet compelling narrative that explains the progression and core issues of this construction dispute. Your narrative must be strictly evidence-based while connecting events into a coherent story.

Project Background:
{project_overview}

Chronological Events:
{key_events}

Financial Analysis:
{financial_analysis}

Document Evidence:
{document_evidence}

Create a chronological narrative that accomplishes the following:

1. Beginning (Project Inception & Early Relationship)
   - Establish the original project scope, timeline, and budget
   - Note initial relationships between parties and contractual expectations
   - Identify early warning signs or initial points of contention

2. Middle (Dispute Development)
   - Trace how specific issues emerged and expanded
   - Document precise moments when parties' positions diverged
   - Connect financial discrepancies to specific project events
   - Highlight critical document contradictions or inconsistencies

3. End (Current Situation)
   - Clearly define the current state of the dispute
   - Summarize the financial impact on all parties
   - Identify core unresolved issues requiring resolution

Throughout your narrative:
   - Maintain strict chronological progression
   - Cite specific documents, dates, and amounts for all key assertions
   - Distinguish between well-evidenced facts and reasonable inferences
   - Present competing perspectives fairly when evidence supports multiple interpretations
   - Focus on actions and documentation rather than assumptions about intent
   - Connect isolated events into patterns where evidence supports such connections
   - Use clear, precise language free of unnecessary legal or technical jargon

Your narrative should help the reader understand not just what happened, but how and why the dispute progressed to its current state, while remaining strictly grounded in the available evidence. The tone should be objective but engaging, helping the reader see the logical progression of events that led to the current dispute.
"""
