"""Reporter Agent for generating comprehensive analysis reports.

This module implements the ReporterAgent class, which uses LLMs to generate
various types of reports based on financial analysis and document evidence.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union

from cdas.ai.prompts.report_generation import (
    EXECUTIVE_SUMMARY_PROMPT, 
    EVIDENCE_CHAIN_PROMPT,
    DETAILED_REPORT_PROMPT,
    PRESENTATION_REPORT_PROMPT,
    DISPUTE_NARRATIVE_PROMPT
)

logger = logging.getLogger(__name__)


class ReporterAgent:
    """Agent that generates comprehensive reports based on analysis results."""
    
    def __init__(self, db_session, llm_manager, config: Optional[Dict[str, Any]] = None):
        """Initialize the reporter agent.
        
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
    
    def generate_executive_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate an executive summary of analysis results.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Executive summary text
        """
        logger.info("Generating executive summary")
        
        # Extract key components from analysis results
        key_findings = self._format_key_findings(analysis_results.get('key_findings', []))
        suspicious_patterns = self._format_suspicious_patterns(analysis_results.get('suspicious_patterns', []))
        disputed_amounts = self._format_disputed_amounts(analysis_results.get('disputed_amounts', []))
        
        # Fill prompt template
        prompt = EXECUTIVE_SUMMARY_PROMPT.format(
            key_findings=key_findings,
            suspicious_patterns=suspicious_patterns,
            disputed_amounts=disputed_amounts
        )
        
        # Generate summary
        summary = self.llm.generate(
            prompt=prompt,
            system_prompt="You are an expert in construction dispute analysis, specializing in creating clear, concise executive summaries for attorneys.",
            temperature=0.3  # Slightly higher temperature for more natural language
        )
        
        return summary
    
    def generate_evidence_chain(self, amount: float, description: str, document_trail: List[Dict[str, Any]]) -> str:
        """Generate an evidence chain for a disputed amount.
        
        Args:
            amount: The disputed amount
            description: Description of the amount
            document_trail: List of documents mentioning this amount
            
        Returns:
            Evidence chain text
        """
        logger.info(f"Generating evidence chain for amount ${amount}")
        
        # Format document trail
        doc_trail_text = self._format_document_trail(document_trail)
        
        # Fill prompt template
        prompt = EVIDENCE_CHAIN_PROMPT.format(
            amount=amount,
            description=description,
            document_trail=doc_trail_text
        )
        
        # Generate evidence chain
        evidence_chain = self.llm.generate(
            prompt=prompt,
            system_prompt="You are an expert in construction dispute analysis, specializing in tracing financial amounts through documents to create clear evidence chains.",
            temperature=0.2
        )
        
        return evidence_chain
    
    def generate_detailed_report(self, analysis_results: Dict[str, Any], audience: str = "technical expert") -> str:
        """Generate a detailed technical report.
        
        Args:
            analysis_results: Dictionary containing analysis results
            audience: Target audience for the report
            
        Returns:
            Detailed report text
        """
        logger.info(f"Generating detailed report for {audience}")
        
        # Convert analysis results to text
        results_text = json.dumps(analysis_results, indent=2)
        
        # Fill prompt template
        prompt = DETAILED_REPORT_PROMPT.format(
            results_text=results_text,
            audience=audience
        )
        
        # Generate detailed report
        report = self.llm.generate(
            prompt=prompt,
            system_prompt="You are an expert in construction dispute analysis, specializing in creating comprehensive technical reports that present complex financial findings clearly.",
            temperature=0.2
        )
        
        return report
    
    def generate_presentation_report(self, analysis_results: Dict[str, Any], audience: str = "attorney") -> str:
        """Generate a presentation-style report.
        
        Args:
            analysis_results: Dictionary containing analysis results
            audience: Target audience for the presentation
            
        Returns:
            Presentation report text
        """
        logger.info(f"Generating presentation report for {audience}")
        
        # Convert analysis results to text
        results_text = json.dumps(analysis_results, indent=2)
        
        # Fill prompt template
        prompt = PRESENTATION_REPORT_PROMPT.format(
            results_text=results_text,
            audience=audience
        )
        
        # Generate presentation report
        report = self.llm.generate(
            prompt=prompt,
            system_prompt="You are an expert in construction dispute analysis, specializing in creating clear, persuasive presentation materials that highlight key findings and evidence.",
            temperature=0.3
        )
        
        return report
    
    def generate_dispute_narrative(self, project_info: Dict[str, Any], analysis_results: Dict[str, Any]) -> str:
        """Generate a narrative explanation of the dispute.
        
        Args:
            project_info: Dictionary containing project information
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Dispute narrative text
        """
        logger.info("Generating dispute narrative")
        
        # Extract and format components
        project_overview = self._format_project_overview(project_info)
        key_events = self._format_key_events(project_info.get('events', []))
        financial_analysis = self._format_financial_analysis(analysis_results)
        document_evidence = self._format_document_evidence(analysis_results)
        
        # Fill prompt template
        prompt = DISPUTE_NARRATIVE_PROMPT.format(
            project_overview=project_overview,
            key_events=key_events,
            financial_analysis=financial_analysis,
            document_evidence=document_evidence
        )
        
        # Generate dispute narrative
        narrative = self.llm.generate(
            prompt=prompt,
            system_prompt="You are an expert in construction dispute analysis, specializing in creating clear, evidence-based narratives that explain complex construction disputes.",
            temperature=0.4  # Higher temperature for more engaging narrative
        )
        
        return narrative
    
    def generate_report_with_tools(self, report_type: str, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a report using tool calls for additional data.
        
        Args:
            report_type: Type of report to generate
            report_data: Initial data for the report
            
        Returns:
            Generated report with metadata
        """
        logger.info(f"Generating {report_type} report with tools")
        
        # Check if we're in mock mode (via LLM manager)
        is_mock_mode = getattr(self.llm, 'mock_mode', False)
        
        # If in mock mode, return a simulated report
        if is_mock_mode:
            logger.info("In mock mode, generating simplified mock report")
            
            return {
                'report_type': report_type,
                'report_content': f"[MOCK {report_type.upper()} REPORT] This is a simulated report based on the provided data.",
                'metadata': {
                    'generated_at': self._get_current_timestamp(),
                    'model': getattr(self.llm, 'model', 'unknown'),
                    'tool_calls': 0
                }
            }
        
        # Prepare initial prompt based on report type
        prompt = self._prepare_report_prompt(report_type, report_data)
        
        # Run agent loop with tools
        max_iterations = self.config.get('max_iterations', 5)
        full_response = []
        tool_call_count = 0
        
        try:
            for i in range(max_iterations):
                logger.info(f"Report generation iteration {i+1}/{max_iterations}")
                
                # Generate response with tools
                response = self.llm.generate_with_tools(prompt, self.tools, self.system_prompt)
                
                # Check for tool calls
                if response.get('tool_calls'):
                    # Execute tool calls
                    tool_results = self._execute_tool_calls(response['tool_calls'])
                    tool_call_count += len(response['tool_calls'])
                    
                    # Add tool results to prompt
                    tool_results_text = "\n".join([f"Tool: {tool}\nResult: {result}" for tool, result in tool_results.items()])
                    prompt += f"\n\nTool Results:\n{tool_results_text}\n\nContinue generating the report:"
                else:
                    # No more tool calls, we're done
                    if response.get('content'):
                        full_response.append(response['content'])
                    break
                
                # Add response content to full response
                if response.get('content'):
                    full_response.append(response['content'])
            
            # Combine all response parts
            combined_report = self._combine_report_parts(full_response)
            
            return {
                'report_type': report_type,
                'report_content': combined_report,
                'metadata': {
                    'generated_at': self._get_current_timestamp(),
                    'model': getattr(self.llm, 'model', 'unknown'),
                    'tool_calls': tool_call_count
                }
            }
        
        except Exception as e:
            logger.error(f"Error during report generation: {str(e)}")
            return {
                'report_type': report_type,
                'error': str(e),
                'partial_report': self._combine_report_parts(full_response) if full_response else None
            }
    
    def _initialize_tools(self) -> List[Dict[str, Any]]:
        """Initialize tools for the reporter agent.
        
        Returns:
            List of tool definitions
        """
        # Import tool registry
        from cdas.ai.agents.tool_registry import get_tool_definitions
        
        # Define needed tools
        tool_names = [
            "search_documents",
            "search_line_items",
            "find_suspicious_patterns",
            "get_document_relationships",
            "get_document_metadata"
        ]
        
        # Get tool definitions
        tools = get_tool_definitions(tool_names)
        
        return tools
    
    def _load_system_prompt(self) -> str:
        """Load system prompt for the reporter agent.
        
        Returns:
            System prompt
        """
        return """You are the Reporter Agent for the Construction Document Analysis System.

Your role is to generate comprehensive, evidence-backed reports based on financial analysis results and document evidence.

When generating reports:
1. Focus on clarity and precision
2. Always cite specific documents and evidence
3. Use professional, objective language
4. Organize information logically
5. Highlight the most important findings clearly
6. Tailor the content to the specified audience
7. Include relevant context for all findings
8. Avoid speculation unless clearly labeled as such

You have access to tools to gather additional information when necessary. Use them to:
- Find supporting evidence for claims
- Locate conflicting information across documents
- Gather more context about disputed amounts
- Access document metadata and relationships

Your reports should help attorneys and other stakeholders understand complex construction disputes and prepare for dispute resolution proceedings."""
    
    def _prepare_report_prompt(self, report_type: str, report_data: Dict[str, Any]) -> str:
        """Prepare initial prompt based on report type.
        
        Args:
            report_type: Type of report to generate
            report_data: Initial data for the report
            
        Returns:
            Formatted prompt
        """
        if report_type == "executive_summary":
            return f"""Please generate an executive summary based on the following analysis results:

Analysis Results:
{json.dumps(report_data, indent=2)}

The executive summary should be concise, highlight the most significant findings, and focus on what would be most useful for an attorney preparing for a dispute resolution conference."""
        
        elif report_type == "evidence_chain":
            return f"""Please generate an evidence chain for the following disputed amount:

Amount: ${report_data.get('amount')}
Description: {report_data.get('description')}
Initial Document: {report_data.get('initial_document')}

Please trace this amount through all relevant documents, showing how it was treated across different document types and identifying any discrepancies or suspicious patterns."""
        
        elif report_type == "detailed_report":
            return f"""Please generate a detailed technical report based on the following analysis results:

Analysis Results:
{json.dumps(report_data, indent=2)}

Target Audience: {report_data.get('audience', 'technical expert')}

This report should be comprehensive, covering all findings related to the construction dispute with supporting evidence and detailed explanation of patterns and anomalies detected."""
        
        elif report_type == "presentation_report":
            return f"""Please generate a presentation-style report based on the following analysis results:

Analysis Results:
{json.dumps(report_data, indent=2)}

Target Audience: {report_data.get('audience', 'attorney')}

This report should be structured as a presentation with clear sections and bullet points highlighting key information."""
        
        elif report_type == "dispute_narrative":
            return f"""Please generate a narrative explanation of this construction dispute based on the following information:

Project Info:
{json.dumps(report_data.get('project_info', {}), indent=2)}

Analysis Results:
{json.dumps(report_data.get('analysis_results', {}), indent=2)}

Create a clear, chronological narrative that explains how the dispute likely developed, the key points of disagreement, financial implications, and the most compelling evidence."""
        
        else:
            return f"""Please generate a {report_type} report based on the following information:

Report Data:
{json.dumps(report_data, indent=2)}

The report should be well-structured, evidence-based, and tailored to assist in construction dispute resolution."""
    
    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, str]:
        """Execute tool calls and return results.
        
        Args:
            tool_calls: List of tool calls to execute
            
        Returns:
            Dictionary mapping tool names to results
        """
        # Import tool registry
        from cdas.ai.agents.tool_registry import execute_tool
        
        results = {}
        
        # Check if we're in mock mode (via LLM manager)
        is_mock_mode = getattr(self.llm, 'mock_mode', False)
        
        # Handle mock mode differently
        if is_mock_mode:
            logger.info("In mock mode, generating mock tool results")
            
            for tool_call in tool_calls:
                # Extract function details based on format
                if isinstance(tool_call, dict):
                    function_name = tool_call.get('function', {}).get('name', 'unknown_function')
                    arguments_str = tool_call.get('function', {}).get('arguments', '{}')
                    try:
                        arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                    except:
                        arguments = {}
                else:
                    # Regular object format
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                
                logger.info(f"Mock executing tool call: {function_name} with arguments: {arguments}")
                
                # Generate mock results based on function name
                if function_name == "search_documents":
                    results[function_name] = "[MOCK] Found 3 documents related to the dispute."
                elif function_name == "search_line_items":
                    results[function_name] = "[MOCK] Found 4 line items matching the description across payment applications."
                elif function_name == "find_suspicious_patterns":
                    results[function_name] = "[MOCK] Identified 2 suspicious patterns in the financial data."
                elif function_name == "get_document_relationships":
                    results[function_name] = "[MOCK] Retrieved relationships between 5 related documents."
                elif function_name == "get_document_metadata":
                    results[function_name] = "[MOCK] Retrieved metadata for the requested document."
                else:
                    results[function_name] = f"[MOCK] Unknown tool: {function_name}"
            
            return results
        
        # Regular execution mode
        for tool_call in tool_calls:
            # Extract function details based on format
            if isinstance(tool_call, dict):
                function_name = tool_call.get('function', {}).get('name', 'unknown_function')
                arguments_str = tool_call.get('function', {}).get('arguments', '{}')
                try:
                    arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                except:
                    arguments = {}
            else:
                # Regular object format
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
            
            logger.info(f"Executing tool call: {function_name} with arguments: {arguments}")
            
            # Execute the tool
            try:
                result = execute_tool(function_name, self.db_session, **arguments)
                results[function_name] = result
            except Exception as e:
                logger.error(f"Error executing tool {function_name}: {str(e)}")
                results[function_name] = f"Error: {str(e)}"
        
        return results
    
    def _combine_report_parts(self, parts: List[str]) -> str:
        """Combine report parts into a single report.
        
        Args:
            parts: List of report parts
            
        Returns:
            Combined report
        """
        if not parts:
            return ""
        
        # If there's only one part, return it directly
        if len(parts) == 1:
            return parts[0]
        
        # Otherwise, stitch them together
        combined = parts[0]
        
        for part in parts[1:]:
            # Avoid duplication by checking for overlap
            overlap_size = self._find_overlap(combined, part)
            if overlap_size > 0:
                combined += part[overlap_size:]
            else:
                combined += "\n\n" + part
        
        return combined
    
    def _find_overlap(self, text1: str, text2: str, min_overlap: int = 20) -> int:
        """Find the size of the overlap between the end of text1 and start of text2.
        
        Args:
            text1: First text
            text2: Second text
            min_overlap: Minimum size of overlap to consider
            
        Returns:
            Size of overlap (0 if no significant overlap)
        """
        # If either text is too short, return 0
        if len(text1) < min_overlap or len(text2) < min_overlap:
            return 0
        
        # Check for the largest overlap
        max_overlap = min(100, len(text1), len(text2))  # Limit to reasonable size
        
        for overlap_size in range(max_overlap, min_overlap - 1, -1):
            if text1[-overlap_size:] == text2[:overlap_size]:
                return overlap_size
        
        return 0
    
    def _format_key_findings(self, findings: List[Dict[str, Any]]) -> str:
        """Format key findings for prompt.
        
        Args:
            findings: List of key findings
            
        Returns:
            Formatted text
        """
        if not findings:
            return "No key findings available."
        
        formatted = []
        for i, finding in enumerate(findings):
            formatted.append(f"{i+1}. {finding.get('description', '')}")
            if 'amount' in finding:
                formatted[-1] += f" (Amount: ${finding['amount']:,.2f})"
            if 'confidence' in finding:
                formatted[-1] += f" - Confidence: {finding['confidence']*100:.0f}%"
        
        return "\n".join(formatted)
    
    def _format_suspicious_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Format suspicious patterns for prompt.
        
        Args:
            patterns: List of suspicious patterns
            
        Returns:
            Formatted text
        """
        if not patterns:
            return "No suspicious patterns detected."
        
        formatted = []
        for i, pattern in enumerate(patterns):
            formatted.append(f"{i+1}. {pattern.get('pattern_type', 'Unknown pattern')}: {pattern.get('description', '')}")
            if 'documents' in pattern:
                formatted.append(f"   Documents: {', '.join(pattern['documents'])}")
            if 'amounts' in pattern and pattern['amounts']:
                amounts_str = ', '.join([f"${amount:,.2f}" for amount in pattern['amounts']])
                formatted.append(f"   Amounts: {amounts_str}")
        
        return "\n".join(formatted)
    
    def _format_disputed_amounts(self, amounts: List[Dict[str, Any]]) -> str:
        """Format disputed amounts for prompt.
        
        Args:
            amounts: List of disputed amounts
            
        Returns:
            Formatted text
        """
        if not amounts:
            return "No disputed amounts identified."
        
        formatted = []
        for i, amount in enumerate(amounts):
            formatted.append(f"{i+1}. ${amount.get('amount', 0):,.2f} - {amount.get('description', 'No description')}")
            if 'documents' in amount:
                formatted.append(f"   Appears in: {', '.join(amount['documents'])}")
        
        return "\n".join(formatted)
    
    def _format_document_trail(self, documents: List[Dict[str, Any]]) -> str:
        """Format document trail for prompt.
        
        Args:
            documents: List of documents
            
        Returns:
            Formatted text
        """
        if not documents:
            return "No document trail available."
        
        formatted = []
        for i, doc in enumerate(documents):
            doc_info = f"{i+1}. {doc.get('doc_type', 'Unknown type')} ({doc.get('date', 'Unknown date')})"
            if 'title' in doc:
                doc_info += f": {doc['title']}"
            
            formatted.append(doc_info)
            
            if 'amount' in doc:
                formatted.append(f"   Amount: ${doc['amount']:,.2f}")
            
            if 'description' in doc:
                formatted.append(f"   Description: {doc['description']}")
            
            if 'context' in doc:
                formatted.append(f"   Context: \"{doc['context']}\"")
            
            formatted.append("")  # Add blank line between documents
        
        return "\n".join(formatted)
    
    def _format_project_overview(self, project_info: Dict[str, Any]) -> str:
        """Format project overview for prompt.
        
        Args:
            project_info: Project information
            
        Returns:
            Formatted text
        """
        if not project_info:
            return "No project information available."
        
        formatted = []
        formatted.append(f"Project Name: {project_info.get('name', 'Unknown')}")
        formatted.append(f"Project Type: {project_info.get('type', 'Unknown')}")
        
        if 'parties' in project_info:
            formatted.append("Parties Involved:")
            for party, role in project_info['parties'].items():
                formatted.append(f"- {party}: {role}")
        
        if 'contract_value' in project_info:
            formatted.append(f"Original Contract Value: ${project_info['contract_value']:,.2f}")
        
        if 'start_date' in project_info:
            formatted.append(f"Start Date: {project_info['start_date']}")
        
        if 'end_date' in project_info:
            formatted.append(f"End Date: {project_info['end_date']}")
        
        if 'description' in project_info:
            formatted.append(f"Description: {project_info['description']}")
        
        return "\n".join(formatted)
    
    def _format_key_events(self, events: List[Dict[str, Any]]) -> str:
        """Format key events for prompt.
        
        Args:
            events: List of key events
            
        Returns:
            Formatted text
        """
        if not events:
            return "No key events documented."
        
        # Sort events by date
        sorted_events = sorted(events, key=lambda e: e.get('date', ''))
        
        formatted = []
        for i, event in enumerate(sorted_events):
            event_info = f"{i+1}. {event.get('date', 'Unknown date')}: {event.get('description', 'No description')}"
            formatted.append(event_info)
            
            if 'documents' in event:
                formatted.append(f"   Documents: {', '.join(event['documents'])}")
            
            if 'amount' in event:
                formatted.append(f"   Amount: ${event['amount']:,.2f}")
        
        return "\n".join(formatted)
    
    def _format_financial_analysis(self, analysis_results: Dict[str, Any]) -> str:
        """Format financial analysis for prompt.
        
        Args:
            analysis_results: Analysis results
            
        Returns:
            Formatted text
        """
        if not analysis_results:
            return "No financial analysis available."
        
        formatted = []
        
        # Key findings
        if 'key_findings' in analysis_results:
            formatted.append("Key Findings:")
            for finding in analysis_results['key_findings']:
                formatted.append(f"- {finding.get('description', '')}")
            formatted.append("")
        
        # Disputed amounts
        if 'disputed_amounts' in analysis_results:
            formatted.append("Disputed Amounts:")
            for amount in analysis_results['disputed_amounts']:
                formatted.append(f"- ${amount.get('amount', 0):,.2f} - {amount.get('description', '')}")
            formatted.append("")
        
        # Suspicious patterns
        if 'suspicious_patterns' in analysis_results:
            formatted.append("Suspicious Patterns:")
            for pattern in analysis_results['suspicious_patterns']:
                formatted.append(f"- {pattern.get('pattern_type', '')}: {pattern.get('description', '')}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _format_document_evidence(self, analysis_results: Dict[str, Any]) -> str:
        """Format document evidence for prompt.
        
        Args:
            analysis_results: Analysis results
            
        Returns:
            Formatted text
        """
        if not analysis_results or 'documents' not in analysis_results:
            return "No document evidence available."
        
        documents = analysis_results.get('documents', [])
        
        formatted = []
        formatted.append(f"Document Evidence ({len(documents)} documents):")
        
        for doc in documents:
            formatted.append(f"- {doc.get('doc_type', 'Unknown type')}: {doc.get('title', 'Untitled')}")
            
            if 'key_excerpts' in doc:
                for excerpt in doc['key_excerpts']:
                    formatted.append(f"  \"{excerpt}\"")
        
        return "\n".join(formatted)
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format.
        
        Returns:
            ISO formatted timestamp
        """
        from datetime import datetime
        return datetime.now().isoformat()