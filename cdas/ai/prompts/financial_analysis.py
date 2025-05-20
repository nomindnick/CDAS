"""Prompt templates for financial analysis.

This module contains prompt templates for financial analysis tasks,
such as suspicious pattern analysis and amount analysis.
"""

# Suspicious pattern analysis prompt template
SUSPICIOUS_PATTERN_ANALYSIS_PROMPT = """
I've identified a potentially suspicious financial pattern in a construction dispute. Please analyze this pattern and provide your assessment.

Pattern Type: {pattern_type}
Amount: ${amount}
Context: {context}

Pattern Details:
{pattern_details}

Based on your expertise in construction financial fraud and disputes, please analyze:
1. How suspicious is this pattern on a scale of 1-10?
2. What might this pattern indicate about the parties' behavior?
3. What additional information would help confirm or refute the suspicion?
4. How common is this type of pattern in construction disputes?

Provide a thorough analysis that would be helpful to an attorney working on this case.
"""

# Amount analysis prompt template
AMOUNT_ANALYSIS_PROMPT = """
Please analyze the following amount that appears in multiple documents in a construction dispute:

Amount: ${amount}

This amount appears in the following contexts:
{contexts}

Timeline of appearances:
{timeline}

Please analyze:
1. Is there anything suspicious about how this amount is used across documents?
2. Is there evidence this amount may have been improperly billed multiple times?
3. Are there inconsistencies in how this amount is described or justified?
4. What does the chronology suggest about this amount?

Provide a detailed analysis with specific references to the documents and contexts.
"""

# Markup analysis prompt template
MARKUP_ANALYSIS_PROMPT = """
Please analyze the following markup calculations from a construction project:

Original Amount: ${original_amount}
Markup Percentage: {markup_percentage}%
Calculated Markup: ${calculated_markup}
Total Amount: ${total_amount}

Occurs in: {document_type} from {party} dated {date}

Please analyze:
1. Is the markup calculation mathematically correct?
2. Is the markup percentage consistent with industry standards for this type of work?
3. Is there anything unusual or potentially problematic about this markup?
4. How does this markup compare to others in the same project by the same party?

Provide a detailed analysis of the markup with specific attention to mathematical accuracy and industry standards.
"""

# Change order analysis prompt template
CHANGE_ORDER_ANALYSIS_PROMPT = """
Please analyze the following change order from a construction project:

Change Order Number: {co_number}
Date: {date}
Original Contract Amount: ${original_amount}
Previous Change Orders: ${previous_changes}
This Change Order: ${this_change}
New Contract Sum: ${new_amount}

Change Description:
{description}

Justification:
{justification}

Please analyze:
1. Is the change order calculation mathematically correct?
2. Does the justification appear reasonable and sufficient for the requested amount?
3. Are there any red flags or concerns with this change order?
4. Is there any evidence that this work might have been included in the original contract scope?
5. Is the timing of this change order significant?

Provide a detailed analysis with attention to the reasonableness and potential issues with this change order.
"""
