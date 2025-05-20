# AI Integration Specification

## Overview

The AI Integration component enhances the Construction Document Analysis System with advanced natural language processing, semantic understanding, and agentic capabilities. By integrating large language models (LLMs) and other AI technologies, the system can intelligently analyze document content, understand context, and generate insights that would be difficult to achieve with rule-based approaches alone.

## Key Capabilities

1. **Document Understanding**: Extract meaning and context from unstructured text
2. **Semantic Search**: Find relevant information based on meaning rather than keywords
3. **Pattern Recognition**: Identify complex patterns in financial data and document relationships
4. **Anomaly Detection**: Detect unusual or suspicious patterns that may indicate fraud or errors
5. **Report Generation**: Create natural language summaries and explanations of findings
6. **Agentic Analysis**: Perform multi-step reasoning to investigate complex scenarios
7. **Interactive Querying**: Allow natural language querying of the document database

## Component Architecture

```
ai/
├─ __init__.py
├─ llm.py                   # LLM integration
├─ embeddings.py            # Document embeddings
├─ prompts/                 # Prompt templates
│   ├─ __init__.py
│   ├─ document_analysis.py # Document analysis prompts
│   ├─ financial_analysis.py # Financial analysis prompts
│   └─ report_generation.py # Report generation prompts
├─ agents/                  # Agent implementations
│   ├─ __init__.py
│   ├─ investigator.py      # Financial investigation agent
│   ├─ reporter.py          # Report generation agent
│   └─ tool_registry.py     # Agent tool registry
├─ tools/                   # Tool implementations
│   ├─ __init__.py
│   ├─ document_tools.py    # Document retrieval tools
│   ├─ financial_tools.py   # Financial analysis tools
│   └─ database_tools.py    # Database query tools
└─ semantic_search/         # Semantic search capabilities
    ├─ __init__.py
    ├─ vectorizer.py        # Text vectorization
    ├─ index.py             # Vector index management
    └─ search.py            # Semantic search implementation
```

## Core Classes

### LLMManager

Manages interactions with language models.

```python
class LLMManager:
    """Manages interactions with language models."""
    
    def __init__(self, config=None):
        """Initialize the LLM manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.model = self.config.get('model', 'gpt-4')
        self.api_key = self.config.get('api_key')
        self.max_tokens = self.config.get('max_tokens', 4000)
        self.temperature = self.config.get('temperature', 0.0)
        
        # Initialize client based on provider
        self.provider = self.config.get('provider', 'openai')
        if self.provider == 'openai':
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        elif self.provider == 'anthropic':
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None):
        """Generate text from the language model.
        
        Args:
            prompt: Prompt text
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            Generated text
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        if self.provider == 'openai':
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens
            )
            
            return response.choices[0].message.content
        
        elif self.provider == 'anthropic':
            if system_prompt:
                prompt = f"{system_prompt}\n\n{prompt}"
            
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=temp,
                max_tokens_to_sample=tokens
            )
            
            return response.completion
    
    def generate_with_tools(self, prompt, tools, system_prompt=None, temperature=None, max_tokens=None):
        """Generate text with function calling capabilities.
        
        Args:
            prompt: Prompt text
            tools: List of tool definitions
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            Generated text and any tool calls
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        if self.provider == 'openai':
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                tools=tools
            )
            
            return {
                'content': response.choices[0].message.content,
                'tool_calls': response.choices[0].message.tool_calls if hasattr(response.choices[0].message, 'tool_calls') else None
            }
        else:
            # For providers without native function calling
            # Use a workaround or raise an error
            raise ValueError(f"Tool calling not implemented for provider: {self.provider}")
```

### EmbeddingManager

Manages document embeddings for semantic search.

```python
class EmbeddingManager:
    """Manages document embeddings for semantic search."""
    
    def __init__(self, db_session, config=None):
        """Initialize the embedding manager.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Initialize embedding model
        self.embedding_model = self.config.get('embedding_model', 'text-embedding-ada-002')
        self.api_key = self.config.get('api_key')
        
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
    
    def generate_embeddings(self, text):
        """Generate embeddings for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        
        return response.data[0].embedding
    
    def embed_document(self, doc_id):
        """Generate and store embeddings for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Success flag
        """
        # Get document content
        query = """
            SELECT 
                p.page_id, 
                p.content
            FROM 
                pages p
            WHERE 
                p.doc_id = %s
            ORDER BY 
                p.page_number
        """
        
        pages = self.db_session.execute(query, (doc_id,)).fetchall()
        
        for page_id, content in pages:
            # Skip if no content
            if not content:
                continue
            
            # Generate embedding
            embedding = self.generate_embeddings(content)
            
            # Store embedding in database
            update_query = """
                UPDATE 
                    pages
                SET 
                    embedding = %s
                WHERE 
                    page_id = %s
            """
            
            self.db_session.execute(update_query, (embedding, page_id))
        
        # Commit changes
        self.db_session.commit()
        
        return True
    
    def search(self, query, limit=5):
        """Search for documents similar to query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of similar documents
        """
        # Generate embedding for query
        query_embedding = self.generate_embeddings(query)
        
        # Find similar documents
        search_query = """
            SELECT 
                p.page_id,
                p.doc_id,
                p.page_number,
                p.content,
                d.doc_type,
                d.party,
                1 - (p.embedding <=> %s) AS similarity
            FROM 
                pages p
            JOIN
                documents d ON p.doc_id = d.doc_id
            WHERE 
                p.embedding IS NOT NULL
            ORDER BY 
                similarity DESC
            LIMIT %s
        """
        
        results = self.db_session.execute(search_query, (query_embedding, limit)).fetchall()
        
        return [dict(zip(
            ['page_id', 'doc_id', 'page_number', 'content', 'doc_type', 'party', 'similarity'],
            result
        )) for result in results]
```

### InvestigatorAgent

Agent that investigates financial discrepancies.

```python
class InvestigatorAgent:
    """Agent that investigates financial discrepancies."""
    
    def __init__(self, db_session, llm_manager, config=None):
        """Initialize the investigator agent.
        
        Args:
            db_session: Database session
            llm_manager: LLM manager
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.llm = llm_manager
        self.config = config or {}
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
    
    def investigate(self, question, context=None):
        """Investigate a question about the construction dispute.
        
        Args:
            question: Question to investigate
            context: Optional context information
            
        Returns:
            Investigation results
        """
        # Prepare initial prompt
        prompt = self._prepare_prompt(question, context)
        
        # Run agent loop
        max_iterations = self.config.get('max_iterations', 10)
        full_response = []
        
        for i in range(max_iterations):
            # Generate response with tools
            response = self.llm.generate_with_tools(prompt, self.tools, self.system_prompt)
            
            # Check for tool calls
            if response.get('tool_calls'):
                # Execute tool calls
                tool_results = self._execute_tool_calls(response['tool_calls'])
                
                # Add tool results to prompt
                tool_results_text = "\n".join([f"Tool: {tool}\nResult: {result}" for tool, result in tool_results.items()])
                prompt += f"\n\nTool Results:\n{tool_results_text}\n\nContinue your investigation:"
            else:
                # No more tool calls, we're done
                full_response.append(response['content'])
                break
            
            # Add response content to full response
            if response['content']:
                full_response.append(response['content'])
        
        # Generate final report
        final_report = self._generate_final_report(question, context, full_response)
        
        return {
            'investigation_steps': full_response,
            'final_report': final_report
        }
    
    def _initialize_tools(self):
        """Initialize agent tools."""
        # Define tools for the agent
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search for documents matching a query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "doc_type": {
                                "type": "string",
                                "description": "Optional document type filter"
                            },
                            "party": {
                                "type": "string",
                                "description": "Optional party filter (district or contractor)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_line_items",
                    "description": "Search for line items matching criteria",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "Optional description search"
                            },
                            "min_amount": {
                                "type": "number",
                                "description": "Optional minimum amount"
                            },
                            "max_amount": {
                                "type": "number",
                                "description": "Optional maximum amount"
                            },
                            "amount": {
                                "type": "number",
                                "description": "Optional exact amount to match (with small tolerance)"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_suspicious_patterns",
                    "description": "Find suspicious financial patterns",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern_type": {
                                "type": "string",
                                "description": "Optional pattern type (recurring_amount, reappearing_amount, inconsistent_markup)"
                            },
                            "min_confidence": {
                                "type": "number",
                                "description": "Optional minimum confidence threshold (0.0-1.0)"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_sql_query",
                    "description": "Run a custom SQL query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SQL query to execute"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        return tools
    
    def _load_system_prompt(self):
        """Load system prompt for the agent."""
        return """You are a Financial Investigator Agent for construction disputes. Your job is to investigate financial discrepancies between a school district and a contractor. You have access to various tools to help you analyze documents, find patterns, and identify suspicious activities.

Follow these steps in your investigation:
1. Understand the question or issue to investigate
2. Plan your investigation approach
3. Use tools to gather relevant information
4. Analyze the information and identify patterns or discrepancies
5. Form hypotheses and test them with additional tools
6. Summarize your findings with evidence

Always cite specific evidence from documents when making claims. Be thorough and methodical in your investigation. Your goal is to uncover the truth about financial matters in the construction dispute."""
    
    def _prepare_prompt(self, question, context=None):
        """Prepare initial prompt for the agent."""
        prompt = f"I need you to investigate the following question about a construction dispute:\n\n{question}\n"
        
        if context:
            prompt += f"\nContext Information:\n{context}\n"
        
        prompt += "\nPlease start your investigation. Use tools as needed to gather information and analyze the situation. Be thorough and methodical."
        
        return prompt
    
    def _execute_tool_calls(self, tool_calls):
        """Execute tool calls and return results."""
        results = {}
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            if function_name == "search_documents":
                results[function_name] = self._search_documents(arguments)
            elif function_name == "search_line_items":
                results[function_name] = self._search_line_items(arguments)
            elif function_name == "find_suspicious_patterns":
                results[function_name] = self._find_suspicious_patterns(arguments)
            elif function_name == "run_sql_query":
                results[function_name] = self._run_sql_query(arguments)
            else:
                results[function_name] = f"Unknown tool: {function_name}"
        
        return results
    
    def _search_documents(self, arguments):
        """Search for documents matching criteria."""
        # Implementation for document search
        pass
    
    def _search_line_items(self, arguments):
        """Search for line items matching criteria."""
        # Implementation for line item search
        pass
    
    def _find_suspicious_patterns(self, arguments):
        """Find suspicious financial patterns."""
        # Implementation for pattern detection
        pass
    
    def _run_sql_query(self, arguments):
        """Run a custom SQL query."""
        # Implementation for SQL query execution
        pass
    
    def _generate_final_report(self, question, context, investigation_steps):
        """Generate final investigation report."""
        # Prepare prompt for final report
        prompt = f"""Please generate a comprehensive final report for the following investigation question:

Question: {question}

Investigation Steps:
{''.join(investigation_steps)}

Your final report should:
1. Summarize the key findings of the investigation
2. Present evidence for each finding, citing specific documents and amounts
3. Explain any suspicious patterns or discrepancies found
4. Answer the original question based on the evidence
5. Suggest next steps or additional areas to investigate

Make the report professional, clear, and well-structured."""

        # Generate report
        report = self.llm.generate(prompt)
        
        return report
```

### ReportGenerator

Generates reports from analysis results.

```python
class ReportGenerator:
    """Generates reports from analysis results."""
    
    def __init__(self, llm_manager, config=None):
        """Initialize the report generator.
        
        Args:
            llm_manager: LLM manager
            config: Optional configuration dictionary
        """
        self.llm = llm_manager
        self.config = config or {}
    
    def generate_report(self, analysis_results, report_type="summary", audience="attorney"):
        """Generate a report from analysis results.
        
        Args:
            analysis_results: Analysis results
            report_type: Type of report to generate
            audience: Target audience for the report
            
        Returns:
            Generated report
        """
        # Prepare prompt based on report type and audience
        prompt = self._prepare_report_prompt(analysis_results, report_type, audience)
        
        # Generate report
        report = self.llm.generate(prompt)
        
        return report
    
    def _prepare_report_prompt(self, analysis_results, report_type, audience):
        """Prepare prompt for report generation."""
        # Convert analysis results to text representation
        results_text = json.dumps(analysis_results, indent=2)
        
        if report_type == "summary":
            prompt = f"""Please generate a summary report based on the following analysis results:

Analysis Results:
{results_text}

This report should provide a clear overview of the key findings and their implications for the construction dispute. The target audience is a {audience}.

The report should include:
1. An executive summary of key findings
2. A clear presentation of the most significant financial discrepancies
3. Evidence supporting each finding, with citations to specific documents
4. A concise explanation of any suspicious patterns detected
5. Recommendations for further investigation or action

The tone should be professional, objective, and evidence-based. Avoid technical jargon that might confuse the audience."""
        
        elif report_type == "detailed":
            prompt = f"""Please generate a detailed technical report based on the following analysis results:

Analysis Results:
{results_text}

This report should provide a comprehensive analysis of all findings related to the construction dispute. The target audience is a {audience}.

The report should include:
1. A detailed executive summary
2. Comprehensive analysis of all financial discrepancies found
3. Full evidence chain for each finding, with document references
4. Thorough explanation of all patterns and anomalies detected
5. Timeline of key financial events
6. Detailed recommendations for further action
7. Technical appendices with supporting data

The tone should be professional, thorough, and technically precise."""
        
        elif report_type == "presentation":
            prompt = f"""Please generate a presentation-style report based on the following analysis results:

Analysis Results:
{results_text}

This report should be structured as a presentation for a {audience}, with clear sections and bullet points highlighting key information.

The presentation should include:
1. Title slide and agenda
2. Key findings (1-2 bullets each)
3. Evidence highlights (visual representations where possible)
4. Suspicious patterns identified (with clear explanations)
5. Conclusions and recommendations
6. Next steps

The tone should be professional but accessible, with clear explanations of technical concepts."""
        
        return prompt
```

## Prompt Templates

The system will use a collection of prompt templates for different AI tasks:

### Document Analysis Prompts

```python
# Example prompt template for document analysis
DOCUMENT_CLASSIFICATION_PROMPT = """
Analyze the following document excerpt and classify it based on its content.

Document Text:
{document_text}

Determine the following:
1. Document Type (e.g., payment application, change order, invoice, etc.)
2. Key Financial Information (amounts, dates, parties involved)
3. Purpose of the Document
4. Stage in Construction Process
5. Any Potential Red Flags or Inconsistencies

Provide your analysis in a structured format.
"""

# Example prompt template for handwriting interpretation
HANDWRITING_INTERPRETATION_PROMPT = """
The following text was extracted from a handwritten note in a construction document using OCR. The confidence in this extraction was {confidence_score}.

Extracted Text:
{extracted_text}

Context: This appears in a {document_type} from {party} dated {date}.

Please interpret what this handwritten note likely means in the context of a construction dispute. If there are obvious errors in the OCR extraction, suggest the correct interpretation.
"""
```

### Financial Analysis Prompts

```python
# Example prompt template for suspicious pattern analysis
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

# Example prompt template for amount analysis
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
```

### Report Generation Prompts

```python
# Example prompt template for executive summary
EXECUTIVE_SUMMARY_PROMPT = """
Please create an executive summary for an attorney based on the following financial analysis findings:

Key Findings:
{key_findings}

Suspicious Patterns:
{suspicious_patterns}

Disputed Amounts:
{disputed_amounts}

The executive summary should be 1-2 paragraphs long, highlight the most significant findings, and focus on information that would be most useful for an attorney preparing for a dispute resolution conference.
"""

# Example prompt template for evidence chain
EVIDENCE_CHAIN_PROMPT = """
Please create a clear presentation of the evidence chain for the following disputed amount:

Amount: ${amount}
Description: {description}

Document Trail:
{document_trail}

This evidence chain should clearly show how this amount has been treated across different documents, highlighting any discrepancies or suspicious patterns. The presentation should be clear and compelling, suitable for use in a dispute resolution conference.
"""
```

## AI Workflow Integration

### Document Processing Enhancement

The AI integration enhances document processing by:

1. **Classification**: Automatically classifying document types
2. **Entity Extraction**: Identifying key entities (amounts, dates, parties)
3. **Context Understanding**: Understanding document purpose and context
4. **Relationship Inference**: Inferring relationships between documents

```python
def enhance_document_processing(document, llm_manager):
    """Enhance document processing with AI capabilities.
    
    Args:
        document: Document object
        llm_manager: LLM manager
        
    Returns:
        Enhanced document data
    """
    # Extract text from document
    text = document.get('content', '')
    if not text:
        return document
    
    # Classify document
    classification_prompt = DOCUMENT_CLASSIFICATION_PROMPT.format(document_text=text[:2000])
    classification_result = llm_manager.generate(classification_prompt)
    
    # Extract key financial information
    extraction_prompt = f"""
    Extract all financial amounts and their descriptions from the following document:
    
    {text[:3000]}
    
    Format the output as a JSON array of objects with the following structure:
    [
        {{"amount": 123.45, "description": "Description of the amount"}}
    ]
    """
    
    extraction_result = llm_manager.generate(extraction_prompt)
    
    # Parse extraction result as JSON
    try:
        extracted_amounts = json.loads(extraction_result)
    except json.JSONDecodeError:
        extracted_amounts = []
    
    # Enhance document data
    enhanced_data = {
        **document,
        'ai_classification': classification_result,
        'extracted_amounts': extracted_amounts
    }
    
    return enhanced_data
```

### Financial Analysis Enhancement

The AI integration enhances financial analysis by:

1. **Pattern Recognition**: Identifying complex financial patterns
2. **Context Analysis**: Understanding the context of financial transactions
3. **Anomaly Explanation**: Providing explanations for anomalies
4. **Evidence Assessment**: Assessing the strength of evidence

```python
def enhance_financial_analysis(analysis_results, llm_manager):
    """Enhance financial analysis with AI capabilities.
    
    Args:
        analysis_results: Analysis results
        llm_manager: LLM manager
        
    Returns:
        Enhanced analysis results
    """
    # Enhance suspicious patterns
    enhanced_patterns = []
    for pattern in analysis_results.get('suspicious_patterns', []):
        pattern_prompt = SUSPICIOUS_PATTERN_ANALYSIS_PROMPT.format(
            pattern_type=pattern.get('type', 'unknown'),
            amount=pattern.get('amount', 0),
            context=pattern.get('explanation', ''),
            pattern_details=json.dumps(pattern, indent=2)
        )
        
        pattern_analysis = llm_manager.generate(pattern_prompt)
        
        enhanced_pattern = {
            **pattern,
            'ai_analysis': pattern_analysis
        }
        
        enhanced_patterns.append(enhanced_pattern)
    
    # Enhance disputed amounts
    enhanced_disputed_amounts = []
    for amount in analysis_results.get('disputed_amounts', []):
        amount_prompt = AMOUNT_ANALYSIS_PROMPT.format(
            amount=amount.get('amount', 0),
            contexts="\n".join([f"- {ctx}" for ctx in amount.get('contexts', [])]),
            timeline="\n".join([f"- {entry}" for entry in amount.get('timeline', [])])
        )
        
        amount_analysis = llm_manager.generate(amount_prompt)
        
        enhanced_amount = {
            **amount,
            'ai_analysis': amount_analysis
        }
        
        enhanced_disputed_amounts.append(enhanced_amount)
    
    # Create enhanced analysis results
    enhanced_results = {
        **analysis_results,
        'suspicious_patterns': enhanced_patterns,
        'disputed_amounts': enhanced_disputed_amounts,
        'executive_summary': _generate_executive_summary(analysis_results, llm_manager)
    }
    
    return enhanced_results

def _generate_executive_summary(analysis_results, llm_manager):
    """Generate executive summary from analysis results."""
    key_findings = "\n".join([f"- {finding}" for finding in analysis_results.get('key_findings', [])])
    suspicious_patterns = "\n".join([
        f"- ${pattern.get('amount', 0)}: {pattern.get('explanation', '')}"
        for pattern in analysis_results.get('suspicious_patterns', [])
    ])
    disputed_amounts = "\n".join([
        f"- ${amount.get('amount', 0)}: {amount.get('description', '')}"
        for amount in analysis_results.get('disputed_amounts', [])
    ])
    
    summary_prompt = EXECUTIVE_SUMMARY_PROMPT.format(
        key_findings=key_findings,
        suspicious_patterns=suspicious_patterns,
        disputed_amounts=disputed_amounts
    )
    
    summary = llm_manager.generate(summary_prompt)
    
    return summary
```

## Agentic Workflows

The system will implement agentic workflows for complex analysis tasks:

### Financial Investigation Workflow

```python
def investigate_financial_issue(issue_description, db_session, llm_manager, config=None):
    """Run a financial investigation workflow.
    
    Args:
        issue_description: Description of the issue to investigate
        db_session: Database session
        llm_manager: LLM manager
        config: Optional configuration
        
    Returns:
        Investigation results
    """
    # Create investigator agent
    agent = InvestigatorAgent(db_session, llm_manager, config)
    
    # Run investigation
    results = agent.investigate(issue_description)
    
    return results
```

### Document Analysis Workflow

```python
def analyze_document_set(doc_ids, db_session, llm_manager, config=None):
    """Run a document analysis workflow.
    
    Args:
        doc_ids: List of document IDs to analyze
        db_session: Database session
        llm_manager: LLM manager
        config: Optional configuration
        
    Returns:
        Analysis results
    """
    # Get documents
    documents = []
    for doc_id in doc_ids:
        # Get document data
        doc_data = get_document_data(doc_id, db_session)
        documents.append(doc_data)
    
    # Prepare prompt
    prompt = f"""I have a set of {len(documents)} construction documents that I need analyzed as a group.

Please analyze these documents for:
1. Common financial elements across documents
2. Potential inconsistencies or discrepancies
3. Chronological relationships and implications
4. Any red flags or suspicious patterns

Document Information:
{json.dumps(documents, indent=2)}

Provide a comprehensive analysis that would help understand the relationships and implications of these documents in a construction dispute context.
"""
    
    # Generate analysis
    analysis = llm_manager.generate(prompt)
    
    return {
        'documents': documents,
        'analysis': analysis
    }
```

## Implementation Guidelines

1. **API Security**: Secure API keys and credentials
2. **Resource Management**: Manage token usage and API costs
3. **Error Handling**: Implement robust error handling for API calls
4. **Caching**: Cache results to reduce API calls
5. **Fallbacks**: Implement fallbacks for API failures
6. **Privacy**: Ensure sensitive document information is handled properly

## Model Selection

1. **Primary LLM**: GPT-4 (OpenAI) or Claude (Anthropic) for high-quality reasoning
2. **Embedding Model**: text-embedding-ada-002 (OpenAI) for document embeddings
3. **Local Options**: Consider local models for privacy-sensitive deployments

## Testing Strategy

1. **Unit Tests**: Test individual AI components
2. **Integration Tests**: Test integration with other system components
3. **Prompt Tests**: Test effectiveness of prompts
4. **Scenario Tests**: Test complex scenarios
5. **Fallback Tests**: Test fallback mechanisms

## Ethical Considerations

1. **Bias Mitigation**: Ensure AI outputs are objective and unbiased
2. **Transparency**: Make AI contribution to analysis clear
3. **Human Oversight**: Maintain human review of AI-generated content
4. **Accountability**: Maintain traceability of AI-generated analysis
