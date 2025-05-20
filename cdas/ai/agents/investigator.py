"""Financial investigation agent implementation.

This module contains the InvestigatorAgent class, which is responsible for
investigating financial discrepancies and suspicious patterns in construction
disputes.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)


class InvestigatorAgent:
    """Agent that investigates financial discrepancies in construction disputes.
    
    This agent uses LLMs with tool-calling capabilities to investigate complex financial
    questions by searching documents, analyzing line items, identifying suspicious patterns,
    and executing custom SQL queries.
    
    The agent operates in an iterative loop, using tools to gather information until it can
    generate a comprehensive final report based on the evidence collected.
    
    Attributes:
        db_session: Database session for accessing construction document data
        llm: LLM manager for interacting with language models
        config (Dict[str, Any]): Configuration settings for the agent
        tools (List[Dict[str, Any]]): List of available tools for the agent to use
        system_prompt (str): System prompt that defines the agent's behavior
    
    Examples:
        >>> from cdas.db.session import get_session
        >>> from cdas.ai.llm import LLMManager
        >>> from cdas.ai.agents.investigator import InvestigatorAgent
        >>> 
        >>> session = get_session()
        >>> llm_manager = LLMManager()
        >>> 
        >>> # Initialize investigator agent
        >>> agent = InvestigatorAgent(session, llm_manager)
        >>> 
        >>> # Investigate a question
        >>> results = agent.investigate(
        ...     "What evidence suggests the contractor double-billed for HVAC equipment?"
        ... )
        >>> 
        >>> # Print the final report
        >>> print(results['final_report'])
    """
    
    def __init__(self, db_session, llm_manager, config: Optional[Dict[str, Any]] = None):
        """Initialize the investigator agent.
        
        Args:
            db_session: Database session for accessing construction document data
            llm_manager: LLM manager for interacting with language models
            config (Optional[Dict[str, Any]]): Configuration dictionary with the following options:
                - max_iterations (int): Maximum number of tool-calling iterations (default: 10)
                - additional configuration options can be added as needed
        
        Note:
            The agent uses the mock mode setting from the LLM manager if it is enabled.
        """
        self.db_session = db_session
        self.llm = llm_manager
        self.config = config or {}
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
    
    def investigate(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Investigate a question about the construction dispute.
        
        This method runs the investigator agent to answer a complex financial question
        about a construction dispute. The agent will use its available tools to search 
        for relevant information, analyze financial patterns, and compile evidence from
        construction documents.
        
        In mock mode (detected from the LLM manager), the agent will return simulated
        results without making actual database or API calls.
        
        Args:
            question (str): The question to investigate about the construction dispute
            context (Optional[str]): Additional context information to help guide the
                investigation, such as relevant document IDs, parties involved, or
                specific areas to focus on
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'investigation_steps' (List[str]): Detailed steps taken during the investigation
                - 'final_report' (str): Comprehensive report summarizing findings with evidence
                - 'error' (str, optional): Error message if an exception occurred
        
        Raises:
            Exception: If an error occurs during the investigation process, it's caught,
                logged, and returned in the result dictionary under the 'error' key
        
        Examples:
            >>> agent = InvestigatorAgent(session, llm_manager)
            >>> 
            >>> # Simple investigation
            >>> results = agent.investigate("Were there any change orders rejected but later included in payment applications?")
            >>> 
            >>> # Investigation with context
            >>> context = "Focus on HVAC-related items and examine payment applications #3 and #4."
            >>> results = agent.investigate("Is there evidence of duplicate billing?", context)
            >>> 
            >>> # Access investigation steps and final report
            >>> for step in results['investigation_steps']:
            ...     print(f"Investigation step: {step}")
            >>> print(f"Final report:\\n{results['final_report']}")
        """
        # Check if we're in mock mode (via LLM manager)
        is_mock_mode = getattr(self.llm, 'mock_mode', False)
        
        # If in mock mode, return a simulated investigation right away
        if is_mock_mode:
            logger.info("In mock mode, generating simplified mock investigation")
            
            # Mock steps for the investigation
            mock_steps = [
                "Analyzing documents related to potential double-billing instances...",
                "Identifying suspicious patterns in billing amounts across documents...",
                "Cross-referencing payment applications with change orders...",
                "Verifying item descriptions and amounts in sequential documents..."
            ]
            
            # Generate mock final report
            final_report = self._generate_final_report(question, context, mock_steps)
            
            return {
                'investigation_steps': mock_steps,
                'final_report': final_report
            }
        
        # Regular investigation flow
        # Prepare initial prompt
        prompt = self._prepare_prompt(question, context)
        
        # Run agent loop
        max_iterations = self.config.get('max_iterations', 10)
        full_response = []
        
        try:
            for i in range(max_iterations):
                logger.info(f"Investigation iteration {i+1}/{max_iterations}")
                
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
                    if response['content']:
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
            
        except Exception as e:
            logger.error(f"Error during investigation: {str(e)}")
            return {
                'investigation_steps': full_response,
                'error': str(e)
            }
    
    def _initialize_tools(self) -> List[Dict[str, Any]]:
        """Initialize agent tools.
        
        This method creates a list of tool definitions that the agent can use during
        investigations. The tools are defined in the OpenAI function-calling format,
        which is converted as needed for different LLM providers.
        
        Returns:
            List[Dict[str, Any]]: List of tool definitions with the following tools:
                - search_documents: Search for documents matching specific criteria
                - search_line_items: Search for financial line items with filters
                - find_suspicious_patterns: Identify potentially suspicious financial patterns
                - run_sql_query: Execute custom SQL queries for advanced analysis
        """
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
    
    def _load_system_prompt(self) -> str:
        """Load system prompt for the agent.
        
        This method returns the system prompt that defines the agent's behavior and
        instructions. The system prompt establishes the agent as a financial investigator
        specialized in construction disputes, and outlines the investigation process.
        
        Returns:
            str: System prompt that guides the agent's behavior and investigation approach
        """
        return """You are a Financial Investigator Agent for construction disputes. Your job is to investigate financial discrepancies between a school district and a contractor. You have access to various tools to help you analyze documents, find patterns, and identify suspicious activities.

Follow these steps in your investigation:
1. Understand the question or issue to investigate
2. Plan your investigation approach
3. Use tools to gather relevant information
4. Analyze the information and identify patterns or discrepancies
5. Form hypotheses and test them with additional tools
6. Summarize your findings with evidence

Always cite specific evidence from documents when making claims. Be thorough and methodical in your investigation. Your goal is to uncover the truth about financial matters in the construction dispute."""
    
    def _prepare_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Prepare initial prompt for the agent.
        
        This method formats the investigation question and optional context information
        into a prompt that initiates the agent's investigation process.
        
        Args:
            question (str): The question to investigate
            context (Optional[str]): Additional context information (if provided)
            
        Returns:
            str: Formatted prompt to start the investigation
        """
        prompt = f"I need you to investigate the following question about a construction dispute:\n\n{question}\n"
        
        if context:
            prompt += f"\nContext Information:\n{context}\n"
        
        prompt += "\nPlease start your investigation. Use tools as needed to gather information and analyze the situation. Be thorough and methodical."
        
        return prompt
    
    def _execute_tool_calls(self, tool_calls) -> Dict[str, str]:
        """Execute tool calls and return results.
        
        This method processes and executes the tool calls made by the language model during
        the investigation. It handles different formats of tool calls from different LLM
        providers and maps them to the appropriate implementation methods.
        
        In mock mode, it returns predefined mock results instead of making actual database calls.
        
        Args:
            tool_calls: List of tool call objects from the LLM (format depends on provider)
            
        Returns:
            Dict[str, str]: Dictionary mapping function names to their execution results as strings
        
        Note:
            The method dynamically handles different tool call formats from Anthropic and OpenAI,
            extracting the function name and arguments appropriately in each case.
        """
        results = {}
        
        # Check if we're in mock mode (via LLM manager)
        is_mock_mode = getattr(self.llm, 'mock_mode', False)
        
        # Handle mock mode differently
        if is_mock_mode:
            logger.info("In mock mode, generating mock tool results")
            
            for tool_call in tool_calls:
                # In mock mode, tool_call might not have function attribute directly
                # It might be a dict instead of an object
                if isinstance(tool_call, dict):
                    function_name = tool_call.get('function', {}).get('name', 'unknown_function')
                    arguments_str = tool_call.get('function', {}).get('arguments', '{}')
                    try:
                        arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                    except:
                        arguments = {}
                else:
                    # Regular OpenAI response format
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                
                logger.info(f"Mock executing tool call: {function_name} with arguments: {arguments}")
                
                # Generate mock results based on function name
                if function_name == "search_documents":
                    results[function_name] = "[MOCK] Found 3 documents matching the criteria related to billing disputes."
                elif function_name == "search_line_items":
                    results[function_name] = "[MOCK] Found 5 line items with similar amounts across different documents."
                elif function_name == "find_suspicious_patterns":
                    results[function_name] = "[MOCK] Identified 2 suspicious patterns with recurring amounts in different documents."
                elif function_name == "run_sql_query":
                    results[function_name] = "[MOCK] Query returned 10 rows showing potential duplicate billing entries."
                else:
                    results[function_name] = f"[MOCK] Unknown tool: {function_name}"
            
            return results
        
        # Regular execution mode - handle different LLM providers
        provider = getattr(self.llm, 'provider', 'anthropic')
        
        for tool_call in tool_calls:
            # Extract function name and arguments based on provider format
            if provider == 'anthropic':
                # Handle Anthropic format
                if isinstance(tool_call, dict):
                    function_name = tool_call.get('function', {}).get('name')
                    arguments_str = tool_call.get('function', {}).get('arguments')
                    try:
                        arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                    except Exception as e:
                        logger.error(f"Error parsing tool call arguments: {str(e)}")
                        arguments = {}
                else:
                    # TypedObject format
                    function_name = getattr(getattr(tool_call, 'function', None), 'name', None)
                    arguments_str = getattr(getattr(tool_call, 'function', None), 'arguments', '{}')
                    try:
                        arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                    except Exception as e:
                        logger.error(f"Error parsing tool call arguments: {str(e)}")
                        arguments = {}
            else:
                # Handle OpenAI format
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
            
            logger.info(f"Executing tool call: {function_name} with arguments: {arguments}")
            
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
    
    def _search_documents(self, arguments: Dict[str, Any]) -> str:
        """Search for documents matching criteria.
        
        This method queries the database for documents that match the specified search criteria.
        It performs a full-text search on document content and can filter by document type and party.
        The results include contextual content snippets showing the matching text.
        
        Args:
            arguments (Dict[str, Any]): Dictionary containing search parameters:
                - query (str): Text to search for in document content
                - doc_type (str, optional): Filter by document type (e.g., 'change_order', 'payment_app')
                - party (str, optional): Filter by party (e.g., 'contractor', 'district')
            
        Returns:
            str: Formatted string of search results with document metadata and relevant content snippets
            
        Raises:
            Exception: If an error occurs during the database query, it's caught, logged,
                and an error message is returned
        """
        try:
            query = arguments.get('query', '')
            doc_type = arguments.get('doc_type')
            party = arguments.get('party')
            
            # Build SQL query
            sql_query = """
                SELECT 
                    d.doc_id,
                    d.title,
                    d.doc_type,
                    d.party,
                    d.date,
                    d.status,
                    p.page_id,
                    p.page_number,
                    p.content
                FROM 
                    documents d
                JOIN
                    pages p ON d.doc_id = p.doc_id
                WHERE 
                    1=1
            """
            
            params = []
            
            if query:
                sql_query += " AND p.content ILIKE %s"
                params.append(f"%{query}%")
            
            if doc_type:
                sql_query += " AND d.doc_type = %s"
                params.append(doc_type)
            
            if party:
                sql_query += " AND d.party = %s"
                params.append(party)
            
            sql_query += " ORDER BY d.date DESC, p.page_number ASC"
            
            # Execute query
            results = self.db_session.execute(sql_query, params).fetchall()
            
            # Format results
            docs = {}
            for doc_id, title, doc_type, party, date, status, page_id, page_number, content in results:
                if doc_id not in docs:
                    docs[doc_id] = {
                        'doc_id': doc_id,
                        'title': title,
                        'doc_type': doc_type,
                        'party': party,
                        'date': date.isoformat() if date else None,
                        'status': status,
                        'pages': []
                    }
                
                if content and query and query.lower() in content.lower():
                    # Extract context around the match
                    start_idx = max(0, content.lower().find(query.lower()) - 100)
                    end_idx = min(len(content), content.lower().find(query.lower()) + len(query) + 100)
                    context = content[start_idx:end_idx]
                    
                    # Add page with context
                    docs[doc_id]['pages'].append({
                        'page_id': page_id,
                        'page_number': page_number,
                        'context': f"...{context}..."
                    })
            
            if not docs:
                return "No documents found matching the criteria."
            
            # Format as string
            result_str = f"Found {len(docs)} documents matching the criteria:\n\n"
            
            for doc in docs.values():
                result_str += f"Document ID: {doc['doc_id']}\n"
                result_str += f"Title: {doc['title']}\n"
                result_str += f"Type: {doc['doc_type']}\n"
                result_str += f"Party: {doc['party']}\n"
                result_str += f"Date: {doc['date']}\n"
                result_str += f"Status: {doc['status']}\n"
                
                for page in doc['pages']:
                    result_str += f"\nPage {page['page_number']}:\n{page['context']}\n"
                
                result_str += "\n---\n\n"
            
            return result_str
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return f"Error searching documents: {str(e)}"
    
    def _search_line_items(self, arguments: Dict[str, Any]) -> str:
        """Search for line items matching criteria.
        
        This method searches the database for financial line items that match the specified criteria.
        It can filter by description keywords and amount ranges, allowing precise identification of
        specific financial entries across different documents.
        
        Args:
            arguments (Dict[str, Any]): Dictionary containing search parameters:
                - description (str, optional): Text to search for in line item descriptions
                - min_amount (float, optional): Minimum amount filter
                - max_amount (float, optional): Maximum amount filter
                - amount (float, optional): Exact amount to match (with small tolerance of 0.01%)
            
        Returns:
            str: Formatted string of matching line items with their document context and financial details
            
        Raises:
            Exception: If an error occurs during the database query, it's caught, logged,
                and an error message is returned
                
        Note:
            When searching for an exact amount, a small tolerance is applied to account for
            rounding differences across documents (0.01% of the amount).
        """
        try:
            description = arguments.get('description')
            min_amount = arguments.get('min_amount')
            max_amount = arguments.get('max_amount')
            amount = arguments.get('amount')
            
            # Build SQL query
            sql_query = """
                SELECT 
                    li.item_id,
                    li.doc_id,
                    d.title,
                    d.doc_type,
                    d.party,
                    d.date,
                    li.description,
                    li.amount,
                    li.quantity,
                    li.unit_price
                FROM 
                    line_items li
                JOIN
                    documents d ON li.doc_id = d.doc_id
                WHERE 
                    1=1
            """
            
            params = []
            
            if description:
                sql_query += " AND li.description ILIKE %s"
                params.append(f"%{description}%")
            
            if min_amount is not None:
                sql_query += " AND li.amount >= %s"
                params.append(min_amount)
            
            if max_amount is not None:
                sql_query += " AND li.amount <= %s"
                params.append(max_amount)
            
            if amount is not None:
                # Search with tolerance (0.01%)
                tolerance = amount * 0.0001
                sql_query += " AND li.amount BETWEEN %s AND %s"
                params.append(amount - tolerance)
                params.append(amount + tolerance)
            
            sql_query += " ORDER BY d.date DESC, li.amount DESC"
            
            # Execute query
            results = self.db_session.execute(sql_query, params).fetchall()
            
            if not results:
                return "No line items found matching the criteria."
            
            # Format results
            result_str = f"Found {len(results)} line items matching the criteria:\n\n"
            
            for item_id, doc_id, title, doc_type, party, date, item_desc, amount, quantity, unit_price in results:
                result_str += f"Item ID: {item_id}\n"
                result_str += f"Document: {title} ({doc_type} from {party}, {date.isoformat() if date else 'Unknown date'})\n"
                result_str += f"Description: {item_desc}\n"
                result_str += f"Amount: ${amount:.2f}\n"
                
                if quantity is not None and unit_price is not None:
                    result_str += f"Quantity: {quantity}\n"
                    result_str += f"Unit Price: ${unit_price:.2f}\n"
                
                result_str += "\n---\n\n"
            
            return result_str
            
        except Exception as e:
            logger.error(f"Error searching line items: {str(e)}")
            return f"Error searching line items: {str(e)}"
    
    def _find_suspicious_patterns(self, arguments: Dict[str, Any]) -> str:
        """Find suspicious financial patterns.
        
        This method uses the financial analysis engine to identify potentially suspicious
        patterns in the construction financial data. It can detect patterns like recurring amounts,
        reappearing amounts after rejection, and inconsistent markups.
        
        Args:
            arguments (Dict[str, Any]): Dictionary containing search parameters:
                - pattern_type (str, optional): Type of pattern to search for
                  ('recurring_amount', 'reappearing_amount', 'inconsistent_markup')
                - min_confidence (float, optional): Minimum confidence threshold (0.0-1.0)
                  for including patterns in results (default: 0.5)
            
        Returns:
            str: Formatted string describing the suspicious patterns found, their confidence levels,
                 explanations, and occurrences across documents
            
        Raises:
            Exception: If an error occurs during pattern analysis, it's caught, logged,
                and an error message is returned
                
        Note:
            This method uses the FinancialAnalysisEngine from the financial_analysis module
            to perform the actual pattern detection.
        """
        try:
            pattern_type = arguments.get('pattern_type')
            min_confidence = arguments.get('min_confidence', 0.5)
            
            # Import the financial analysis module
            from cdas.financial_analysis.engine import FinancialAnalysisEngine
            
            # Create analysis engine
            engine = FinancialAnalysisEngine(self.db_session)
            
            # Find suspicious patterns
            if pattern_type:
                patterns = engine.find_suspicious_patterns(pattern_type=pattern_type, min_confidence=min_confidence)
            else:
                patterns = engine.find_suspicious_patterns(min_confidence=min_confidence)
            
            if not patterns:
                return "No suspicious patterns found matching the criteria."
            
            # Format results
            result_str = f"Found {len(patterns)} suspicious patterns:\n\n"
            
            for pattern in patterns:
                result_str += f"Pattern Type: {pattern['type']}\n"
                result_str += f"Amount: ${pattern['amount']:.2f}\n"
                result_str += f"Confidence: {pattern['confidence']:.2f}\n"
                result_str += f"Explanation: {pattern['explanation']}\n"
                
                if 'occurrences' in pattern:
                    result_str += "\nOccurrences:\n"
                    for occurrence in pattern['occurrences']:
                        result_str += f"- {occurrence['doc_type']} from {occurrence['party']} dated "
                        result_str += f"{occurrence['date']}: ${occurrence['amount']:.2f}\n"
                
                result_str += "\n---\n\n"
            
            return result_str
            
        except Exception as e:
            logger.error(f"Error finding suspicious patterns: {str(e)}")
            return f"Error finding suspicious patterns: {str(e)}"
    
    def _run_sql_query(self, arguments: Dict[str, Any]) -> str:
        """Run a custom SQL query.
        
        This method allows the agent to execute custom SQL queries for more advanced
        data analysis. For security reasons, only SELECT statements are permitted.
        Results are formatted as a text table with headers.
        
        Args:
            arguments (Dict[str, Any]): Dictionary containing the query:
                - query (str): SQL query to execute (must be a SELECT statement)
            
        Returns:
            str: Formatted string containing the query results as a text table,
                 or an error message if the query is invalid or fails
            
        Raises:
            Exception: If an error occurs during query execution, it's caught, logged,
                and an error message is returned
                
        Note:
            For security reasons, only SELECT queries are allowed. The result is limited
            to 50 rows to prevent excessive output.
        """
        try:
            query = arguments.get('query')
            
            if not query:
                return "No query provided."
            
            # Validate query (only allow SELECT statements)
            if not query.strip().lower().startswith('select'):
                return "Error: Only SELECT queries are allowed for security reasons."
            
            # Execute query
            results = self.db_session.execute(query).fetchall()
            
            if not results:
                return "Query executed successfully but returned no results."
            
            # Get column names
            column_names = results[0].keys() if hasattr(results[0], 'keys') else [f"col_{i}" for i in range(len(results[0]))]
            
            # Format results as table
            result_str = f"Query returned {len(results)} rows:\n\n"
            
            # Add header
            header = " | ".join(column_names)
            result_str += header + "\n"
            result_str += "-" * len(header) + "\n"
            
            # Add rows
            for row in results[:50]:  # Limit to 50 rows
                values = [str(val) if val is not None else "NULL" for val in row]
                result_str += " | ".join(values) + "\n"
            
            if len(results) > 50:
                result_str += "\n... (output truncated, showing first 50 rows)"
            
            return result_str
            
        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            return f"Error executing SQL query: {str(e)}"
    
    def _generate_final_report(self, question: str, context: Optional[str], investigation_steps: List[str]) -> str:
        """Generate final investigation report.
        
        This method creates a comprehensive final report that summarizes the findings
        of the investigation. It uses the LLM to synthesize the information gathered
        during the investigation steps into a well-structured report.
        
        In mock mode, it returns a predefined mock report for testing purposes.
        
        Args:
            question (str): The original investigation question
            context (Optional[str]): Additional context information provided initially
            investigation_steps (List[str]): Collection of investigation steps and
                intermediate findings from the agent's investigation process
            
        Returns:
            str: Formatted investigation report with findings, evidence, and conclusions
            
        Raises:
            Exception: If an error occurs during report generation, it's caught, logged,
                and an error message is returned
                
        Note:
            The report is structured to include a summary of findings, key evidence,
            explanation of suspicious patterns or discrepancies, answers to the original
            question, and suggested next steps.
        """
        try:
            # Check if we're in mock mode (via LLM manager)
            is_mock_mode = getattr(self.llm, 'mock_mode', False)
            
            if is_mock_mode:
                logger.info("In mock mode, generating mock final report")
                
                # Generate simple mock report
                mock_report = f"""# Investigation Report: {question}

## Summary of Findings

Based on my investigation, I've identified several instances of potential financial irregularities in this project.

## Key Evidence

1. Two separate payment applications (#3 and #4) both include identical line items for "Foundation waterproofing" with the same amount of $24,500.

2. The contractor submitted change order #CS-103 for "Additional electrical work" which appears to overlap with scope already included in the base contract.

3. Invoice #INV-2023-087 contains line items that match previously approved and paid work from payment application #2.

## Suspicious Patterns

Analysis reveals a pattern of resubmitting previously rejected change orders with slightly modified descriptions but identical amounts.

## Conclusion

The evidence suggests there are instances of potential double-billing in this project that warrant further investigation.

## Recommended Next Steps

1. Conduct a detailed audit of all payment applications and change orders
2. Request clarification from the contractor regarding the specific items identified
3. Cross-reference all amounts against the original contract scope
"""
                return mock_report
            
            # Regular execution - prepare prompt for final report
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
            
        except Exception as e:
            logger.error(f"Error generating final report: {str(e)}")
            return f"Error generating final report: {str(e)}"