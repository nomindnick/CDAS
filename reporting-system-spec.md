# Reporting System Specification

## Overview

The Reporting System is a critical component of the Construction Document Analysis System, transforming raw analysis results into clear, actionable reports for attorneys in construction disputes. The system produces well-structured, evidence-backed documents that present financial findings in a format suitable for dispute resolution conferences, settlement negotiations, and potentially litigation support.

## Key Capabilities

1. **Multiple Report Types**: Generate summaries, detailed analyses, evidence chains, and custom reports
2. **Multiple Output Formats**: Support for PDF, HTML, Markdown, and Excel outputs
3. **Evidence Citation**: Link findings directly to source documents with page references
4. **Visual Presentation**: Charts, tables, and timelines to visualize financial data
5. **Template-Based Generation**: Customizable templates for different report types and audiences
6. **Interactive Elements**: Dynamic elements in digital formats for exploring the evidence
7. **Audience Adaptation**: Tailored content for different audiences (attorneys, clients, experts, mediators)
8. **Executive Summaries**: Clear, concise summaries of key findings

## Component Architecture

```
reporting/
├─ __init__.py
├─ generator.py              # Main report generation
├─ formatters/               # Output format handlers
│   ├─ __init__.py
│   ├─ markdown.py           # Markdown formatter
│   ├─ html.py               # HTML formatter
│   ├─ pdf.py                # PDF formatter
│   └─ excel.py              # Excel formatter
├─ templates/                # Report templates
│   ├─ __init__.py
│   ├─ summary/              # Summary report templates
│   ├─ detailed/             # Detailed report templates
│   ├─ evidence/             # Evidence report templates
│   └─ custom/               # Custom report templates
├─ components/               # Report components
│   ├─ __init__.py
│   ├─ tables.py             # Table generation
│   ├─ charts.py             # Chart generation
│   ├─ timelines.py          # Timeline generation
│   └─ citations.py          # Citation formatting
└─ utils/                    # Utility functions
    ├─ __init__.py
    └─ document_linking.py   # Document reference utilities
```

## Core Classes

### ReportGenerator

Main entry point for report generation.

```python
class ReportGenerator:
    """Generates reports from analysis results."""
    
    def __init__(self, db_session=None, config=None):
        """Initialize the report generator.
        
        Args:
            db_session: Optional database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Initialize formatters
        self.formatters = {
            'markdown': MarkdownFormatter(self.config),
            'html': HTMLFormatter(self.config),
            'pdf': PDFFormatter(self.config),
            'excel': ExcelFormatter(self.config)
        }
    
    def generate_summary_report(self, report_data, format='markdown', output_path=None):
        """Generate a summary report.
        
        Args:
            report_data: Report data dictionary
            format: Output format ('markdown', 'html', 'pdf', 'excel')
            output_path: Optional output file path
            
        Returns:
            Report content or file path
        """
        # Load summary report template
        template = self._load_template('summary')
        
        # Generate report content
        content = template.render(data=report_data)
        
        # Apply formatter
        return self._format_output(content, format, output_path)
    
    def generate_detailed_report(self, report_data, format='markdown', output_path=None):
        """Generate a detailed report.
        
        Args:
            report_data: Report data dictionary
            format: Output format ('markdown', 'html', 'pdf', 'excel')
            output_path: Optional output file path
            
        Returns:
            Report content or file path
        """
        # Load detailed report template
        template = self._load_template('detailed')
        
        # Generate report content
        content = template.render(data=report_data)
        
        # Apply formatter
        return self._format_output(content, format, output_path)
    
    def generate_evidence_report(self, amount, matches, history, include_images=True, format='markdown', output_path=None):
        """Generate an evidence chain report for a specific amount.
        
        Args:
            amount: Amount to create evidence chain for
            matches: List of matches for the amount
            history: Chronological history of the amount
            include_images: Whether to include document images
            format: Output format ('markdown', 'html', 'pdf', 'excel')
            output_path: Optional output file path
            
        Returns:
            Report content or file path
        """
        # Load evidence report template
        template = self._load_template('evidence')
        
        # Prepare report data
        report_data = {
            'amount': amount,
            'matches': matches,
            'history': history,
            'include_images': include_images
        }
        
        # Generate report content
        content = template.render(data=report_data)
        
        # Apply formatter
        return self._format_output(content, format, output_path)
    
    def generate_custom_report(self, question, analysis_results, title=None, audience='attorney', format='markdown', output_path=None):
        """Generate a custom report focused on a specific question.
        
        Args:
            question: Question or focus for the report
            analysis_results: Analysis results
            title: Optional report title
            audience: Target audience ('attorney', 'client', 'expert', 'mediator')
            format: Output format ('markdown', 'html', 'pdf', 'excel')
            output_path: Optional output file path
            
        Returns:
            Report content or file path
        """
        # Load custom report template for the audience
        template = self._load_template(f'custom/{audience}')
        
        # Prepare report data
        report_data = {
            'question': question,
            'title': title or f"Analysis: {question}",
            'results': analysis_results,
            'audience': audience
        }
        
        # Generate report content
        content = template.render(data=report_data)
        
        # Apply formatter
        return self._format_output(content, format, output_path)
    
    def _load_template(self, template_name):
        """Load a report template.
        
        Args:
            template_name: Template name
            
        Returns:
            Template object
        """
        # Implementation depends on template engine (e.g., Jinja2)
        from jinja2 import Environment, FileSystemLoader
        
        # Get template directory from config or use default
        template_dir = self.config.get('template_dir', os.path.join(os.path.dirname(__file__), 'templates'))
        
        # Create Jinja2 environment
        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=True
        )
        
        # Load template
        return env.get_template(f"{template_name}.jinja")
    
    def _format_output(self, content, format, output_path=None):
        """Format output based on requested format.
        
        Args:
            content: Report content
            format: Output format
            output_path: Optional output file path
            
        Returns:
            Formatted content or file path
        """
        formatter = self.formatters.get(format)
        if not formatter:
            raise ValueError(f"Unsupported format: {format}")
        
        return formatter.format(content, output_path)
```

### MarkdownFormatter

Formats reports as Markdown.

```python
class MarkdownFormatter:
    """Formats reports as Markdown."""
    
    def __init__(self, config=None):
        """Initialize the Markdown formatter.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
    
    def format(self, content, output_path=None):
        """Format content as Markdown.
        
        Args:
            content: Report content
            output_path: Optional output file path
            
        Returns:
            Markdown content or file path
        """
        # Content is already in Markdown format
        markdown_content = content
        
        # Save to file if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(markdown_content)
            return output_path
        
        return markdown_content
```

### HTMLFormatter

Formats reports as HTML.

```python
class HTMLFormatter:
    """Formats reports as HTML."""
    
    def __init__(self, config=None):
        """Initialize the HTML formatter.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
    
    def format(self, content, output_path=None):
        """Format content as HTML.
        
        Args:
            content: Report content (Markdown)
            output_path: Optional output file path
            
        Returns:
            HTML content or file path
        """
        # Convert Markdown to HTML
        html_content = self._convert_markdown_to_html(content)
        
        # Save to file if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html_content)
            return output_path
        
        return html_content
    
    def _convert_markdown_to_html(self, markdown_content):
        """Convert Markdown to HTML.
        
        Args:
            markdown_content: Markdown content
            
        Returns:
            HTML content
        """
        # Use markdown library for conversion
        import markdown
        
        # Get HTML template
        html_template = self._get_html_template()
        
        # Convert Markdown to HTML
        html_body = markdown.markdown(
            markdown_content,
            extensions=[
                'extra',
                'tables',
                'toc',
                'fenced_code',
                'codehilite'
            ]
        )
        
        # Insert HTML body into template
        html_content = html_template.replace('{{content}}', html_body)
        
        return html_content
    
    def _get_html_template(self):
        """Get HTML template.
        
        Returns:
            HTML template
        """
        # Default template with CSS styling
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Construction Document Analysis Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3, h4 {
            color: #0066cc;
            margin-top: 1.5em;
        }
        h1 {
            border-bottom: 2px solid #0066cc;
            padding-bottom: 10px;
        }
        h2 {
            border-bottom: 1px solid #ccc;
            padding-bottom: 5px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        code {
            background-color: #f5f5f5;
            border: 1px solid #e1e1e8;
            border-radius: 3px;
            padding: 2px 4px;
            font-family: Consolas, monospace;
        }
        pre {
            background-color: #f5f5f5;
            border: 1px solid #e1e1e8;
            border-radius: 3px;
            padding: 10px;
            overflow: auto;
        }
        blockquote {
            border-left: 4px solid #0066cc;
            padding-left: 15px;
            color: #666;
            margin: 20px 0;
        }
        img {
            max-width: 100%;
            height: auto;
        }
        .evidence {
            background-color: #f6f9ff;
            border-left: 4px solid #0066cc;
            padding: 10px;
            margin: 20px 0;
        }
        .warning {
            background-color: #fff9f6;
            border-left: 4px solid #ff6600;
            padding: 10px;
            margin: 20px 0;
        }
        .citation {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    {{content}}
</body>
</html>"""
```

### PDFFormatter

Formats reports as PDF.

```python
class PDFFormatter:
    """Formats reports as PDF."""
    
    def __init__(self, config=None):
        """Initialize the PDF formatter.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.html_formatter = HTMLFormatter(config)
    
    def format(self, content, output_path=None):
        """Format content as PDF.
        
        Args:
            content: Report content (Markdown)
            output_path: Optional output file path
            
        Returns:
            PDF file path
        """
        if not output_path:
            raise ValueError("Output path is required for PDF formatting")
        
        # Convert Markdown to HTML
        html_content = self.html_formatter._convert_markdown_to_html(content)
        
        # Convert HTML to PDF
        self._convert_html_to_pdf(html_content, output_path)
        
        return output_path
    
    def _convert_html_to_pdf(self, html_content, output_path):
        """Convert HTML to PDF.
        
        Args:
            html_content: HTML content
            output_path: Output file path
        """
        # Use weasyprint for PDF conversion
        from weasyprint import HTML, CSS
        
        # Create temporary HTML file
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(html_content.encode('utf-8'))
        
        try:
            # Convert HTML to PDF
            HTML(filename=temp_file_path).write_pdf(
                output_path,
                stylesheets=[CSS(string='@page { margin: 1cm }')]
            )
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
```

### ExcelFormatter

Formats reports as Excel spreadsheets.

```python
class ExcelFormatter:
    """Formats reports as Excel spreadsheets."""
    
    def __init__(self, config=None):
        """Initialize the Excel formatter.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
    
    def format(self, content, output_path=None):
        """Format content as Excel.
        
        Args:
            content: Report content (Markdown)
            output_path: Optional output file path
            
        Returns:
            Excel file path
        """
        if not output_path:
            raise ValueError("Output path is required for Excel formatting")
        
        # Parse Markdown content to extract tables and data
        report_data = self._parse_markdown_tables(content)
        
        # Create Excel workbook
        workbook = self._create_excel_workbook(report_data)
        
        # Save workbook
        workbook.save(output_path)
        
        return output_path
    
    def _parse_markdown_tables(self, content):
        """Parse Markdown tables from content.
        
        Args:
            content: Markdown content
            
        Returns:
            Dictionary of tables and metadata
        """
        # Implementation for parsing Markdown tables
        # This is a simplified version; a real implementation would be more robust
        
        import re
        
        # Extract sections and tables
        sections = {}
        current_section = "Summary"
        sections[current_section] = []
        
        # Split content into lines
        lines = content.split('\n')
        
        # Regex for table row
        table_row_pattern = r'^\s*\|(.+)\|\s*$'
        
        in_table = False
        current_table = []
        
        for line in lines:
            # Check for heading
            if line.startswith('#'):
                level = len(re.match(r'^#+', line).group(0))
                if level <= 2:  # Main section
                    current_section = line.lstrip('#').strip()
                    sections[current_section] = []
                    in_table = False
            
            # Check for table row
            if re.match(table_row_pattern, line):
                if not in_table:
                    in_table = True
                    current_table = []
                
                # Parse row cells
                cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
                current_table.append(cells)
            elif in_table:
                # End of table
                if current_table:
                    sections[current_section].append(current_table)
                in_table = False
                current_table = []
        
        # Add last table if exists
        if in_table and current_table:
            sections[current_section].append(current_table)
        
        return sections
    
    def _create_excel_workbook(self, report_data):
        """Create Excel workbook from report data.
        
        Args:
            report_data: Report data dictionary
            
        Returns:
            Excel workbook
        """
        import openpyxl
        from openpyxl.styles import Font, Alignment, PatternFill
        
        # Create new workbook
        workbook = openpyxl.Workbook()
        
        # Remove default sheet
        default_sheet = workbook.active
        workbook.remove(default_sheet)
        
        # Create sheet for each section
        for section_name, tables in report_data.items():
            # Create sheet
            sheet = workbook.create_sheet(title=section_name[:31])  # Excel sheet names limited to 31 chars
            
            # Set column widths
            for col in range(1, 10):
                sheet.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 20
            
            # Add section title
            sheet.cell(row=1, column=1, value=section_name)
            sheet.cell(row=1, column=1).font = Font(bold=True, size=14)
            
            # Current row
            current_row = 3
            
            # Add tables
            for table in tables:
                if not table:
                    continue
                    
                # Add header row
                header_row = table[0]
                for col_idx, cell_value in enumerate(header_row, start=1):
                    cell = sheet.cell(row=current_row, column=col_idx, value=cell_value)
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
                
                # Add data rows
                for row_idx, row in enumerate(table[1:], start=1):
                    for col_idx, cell_value in enumerate(row, start=1):
                        sheet.cell(row=current_row + row_idx, column=col_idx, value=cell_value)
                
                # Move to next table position
                current_row += len(table) + 2
        
        return workbook
```

## Report Templates

The system uses templates to generate consistent, well-structured reports.

### Summary Report Template

```markdown
# {{ data.title }}

**Generated:** {{ data.generated_at }}

## Executive Summary

{{ data.executive_summary }}

## Key Findings

{% for finding in data.key_findings %}
- {{ finding }}
{% endfor %}

## Financial Overview

| Category | Amount | Notes |
|----------|--------|-------|
{% for category, info in data.summary.items() %}
| {{ category }} | ${{ info.amount }} | {{ info.notes }} |
{% endfor %}

## Suspicious Patterns

{% for pattern in data.suspicious_patterns %}
### {{ pattern.type }} - ${{ pattern.amount }}

**Confidence:** {{ pattern.confidence }}

{{ pattern.explanation }}

**Evidence:**
{% for evidence in pattern.evidence %}
- {{ evidence.document }} ({{ evidence.date }}): {{ evidence.description }}
{% endfor %}

{% endfor %}

## Disputed Amounts

{% for amount in data.disputed_amounts %}
### ${{ amount.amount }} - {{ amount.description }}

**Status:** {{ amount.status }}

{{ amount.notes }}

**Document Trail:**
{% for doc in amount.documents %}
- {{ doc.type }} ({{ doc.date }}): {{ doc.description }}
{% endfor %}

{% endfor %}

## Recommendations

{% for recommendation in data.recommendations %}
- {{ recommendation }}
{% endfor %}

## Appendix: Document Inventory

| Document ID | Type | Party | Date | Description |
|-------------|------|-------|------|-------------|
{% for doc in data.documents %}
| {{ doc.id }} | {{ doc.type }} | {{ doc.party }} | {{ doc.date }} | {{ doc.description }} |
{% endfor %}
```

### Evidence Report Template

```markdown
# Evidence Chain: ${{ data.amount }}

## Overview

This report traces the amount ${{ data.amount }} through all relevant documents in the construction dispute.

## Chronological Appearance

{% for event in data.history.timeline %}
### {{ event.date }} - {{ event.doc_type }}

**Document:** {{ event.doc_id }}
**Party:** {{ event.party }}
**Description:** {{ event.description }}

{% if data.include_images and event.image_path %}
![Document Image]({{ event.image_path }})
{% endif %}

{% endfor %}

## Key Events

{% for event in data.history.key_events %}
### {{ event.type }}

{% if event.type == 'first_occurrence' %}
**First appeared in {{ event.event.doc_type }} on {{ event.event.date }}**
{% elif event.type == 'status_change' %}
**Changed from {{ event.from_event.doc_type }} to {{ event.to_event.doc_type }} on {{ event.to_event.date }}**
{% endif %}

{% endfor %}

## Analysis

{% if data.analysis %}
{{ data.analysis }}
{% else %}
No additional analysis available.
{% endif %}

## Conclusion

{% if data.conclusion %}
{{ data.conclusion }}
{% else %}
This evidence chain documents the history of the amount ${{ data.amount }} throughout the project documentation.
{% endif %}
```

## Chart Generation

For visual representation of financial data.

```python
def generate_chart(chart_type, data, title=None, output_path=None):
    """Generate a chart for the report.
    
    Args:
        chart_type: Type of chart ('bar', 'line', 'pie', etc.)
        data: Chart data
        title: Optional chart title
        output_path: Optional output file path
        
    Returns:
        Chart image path or base64 encoded image
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    import base64
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Generate chart based on type
    if chart_type == 'bar':
        # Bar chart
        labels = [item['label'] for item in data]
        values = [item['value'] for item in data]
        colors = [item.get('color', '#1f77b4') for item in data]
        
        bars = plt.bar(labels, values, color=colors)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.2f}',
                    ha='center', va='bottom')
        
    elif chart_type == 'pie':
        # Pie chart
        labels = [item['label'] for item in data]
        values = [item['value'] for item in data]
        colors = [item.get('color') for item in data if 'color' in item]
        
        plt.pie(values, labels=labels, colors=colors if colors else None,
                autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
    
    elif chart_type == 'line':
        # Line chart
        for series in data:
            plt.plot(series['x'], series['y'], label=series['label'],
                    color=series.get('color'), marker=series.get('marker', 'o'))
        
        plt.legend()
    
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    # Add title if provided
    if title:
        plt.title(title)
    
    # Format axes
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save chart to file or return as base64
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        # Save to in-memory file
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Encode as base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return f"data:image/png;base64,{img_str}"
```

## Timeline Generation

For visualizing the chronology of financial events.

```python
def generate_timeline(events, title=None, output_path=None):
    """Generate a timeline visualization.
    
    Args:
        events: List of timeline events
        title: Optional timeline title
        output_path: Optional output file path
        
    Returns:
        Timeline image path or base64 encoded image
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    import numpy as np
    import io
    import base64
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Parse dates and sort events
    for event in events:
        if isinstance(event['date'], str):
            event['date'] = datetime.strptime(event['date'], '%Y-%m-%d')
    
    events = sorted(events, key=lambda x: x['date'])
    
    # Extract dates and descriptions
    dates = [event['date'] for event in events]
    descriptions = [event['description'] for event in events]
    categories = [event.get('category', 'default') for event in events]
    
    # Define colors for categories
    category_colors = {
        'default': '#1f77b4',
        'payment_app': '#2ca02c',
        'change_order': '#d62728',
        'rejection': '#ff7f0e',
        'approval': '#17becf'
    }
    
    # Create timeline
    y_positions = np.arange(len(dates))
    colors = [category_colors.get(cat, category_colors['default']) for cat in categories]
    
    ax.scatter(dates, y_positions, s=100, c=colors, zorder=3)
    
    # Add lines connecting events
    ax.plot(dates, y_positions, '-', color='gray', alpha=0.3, zorder=1)
    
    # Add event descriptions
    for i, (date, desc, y_pos) in enumerate(zip(dates, descriptions, y_positions)):
        ax.annotate(desc, (date, y_pos),
                   xytext=(10, 0), textcoords='offset points',
                   va='center', fontsize=9)
    
    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    
    # Remove y-axis ticks and labels
    ax.set_yticks([])
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    # Add title if provided
    if title:
        plt.title(title)
    
    # Add legend for categories
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=cat)
        for cat, color in category_colors.items()
        if cat in categories
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save timeline to file or return as base64
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        # Save to in-memory file
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Encode as base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return f"data:image/png;base64,{img_str}"
```

## Citation Formatting

For linking findings to source documents.

```python
def format_citation(document, page=None, item_id=None, db_session=None):
    """Format a citation to a document.
    
    Args:
        document: Document object or ID
        page: Optional page number
        item_id: Optional line item ID
        db_session: Optional database session
        
    Returns:
        Formatted citation
    """
    # Get document information
    doc_info = _get_document_info(document, db_session)
    
    if not doc_info:
        return f"Unknown Document"
    
    # Build citation
    citation = f"{doc_info['file_name']}"
    
    if doc_info['doc_type']:
        citation += f" ({doc_info['doc_type']})"
    
    if page:
        citation += f", Page {page}"
    
    if item_id and db_session:
        # Get line item information
        item_info = _get_line_item_info(item_id, db_session)
        if item_info:
            citation += f", Line Item: {item_info['description'][:30]}..."
    
    if doc_info['date_created']:
        citation += f", {doc_info['date_created'].strftime('%Y-%m-%d')}"
    
    return citation

def _get_document_info(document, db_session):
    """Get document information.
    
    Args:
        document: Document object or ID
        db_session: Optional database session
        
    Returns:
        Document information dictionary
    """
    if isinstance(document, dict):
        # Document is already a dictionary
        return document
    
    if not db_session:
        # Can't look up document without session
        return {'file_name': str(document)}
    
    # Look up document in database
    doc_id = document
    result = db_session.execute(
        "SELECT doc_id, file_name, doc_type, party, date_created FROM documents WHERE doc_id = %s",
        (doc_id,)
    ).fetchone()
    
    if not result:
        return None
    
    return dict(zip(['doc_id', 'file_name', 'doc_type', 'party', 'date_created'], result))

def _get_line_item_info(item_id, db_session):
    """Get line item information.
    
    Args:
        item_id: Line item ID
        db_session: Database session
        
    Returns:
        Line item information dictionary
    """
    result = db_session.execute(
        "SELECT item_id, description, amount FROM line_items WHERE item_id = %s",
        (item_id,)
    ).fetchone()
    
    if not result:
        return None
    
    return dict(zip(['item_id', 'description', 'amount'], result))
```

## Document Linking

For creating links to source documents.

```python
def create_document_link(document, page=None, item_id=None, link_type='html', base_url=None):
    """Create a link to a document.
    
    Args:
        document: Document object or ID
        page: Optional page number
        item_id: Optional line item ID
        link_type: Link type ('html', 'pdf', 'markdown')
        base_url: Optional base URL for document viewer
        
    Returns:
        Document link
    """
    # Get document ID
    doc_id = document['doc_id'] if isinstance(document, dict) else document
    
    # Build query parameters
    params = []
    if page:
        params.append(f"page={page}")
    if item_id:
        params.append(f"item={item_id}")
    
    query_string = f"?{'&'.join(params)}" if params else ""
    
    # Build link based on type
    if link_type == 'html':
        base = base_url or "/documents"
        return f"{base}/{doc_id}{query_string}"
    
    elif link_type == 'pdf':
        base = base_url or "file://"
        return f"{base}/{doc_id}.pdf{query_string}"
    
    elif link_type == 'markdown':
        base = base_url or "/documents"
        link = f"{base}/{doc_id}{query_string}"
        
        # Get document info for link text
        if isinstance(document, dict):
            doc_info = document
            link_text = doc_info.get('file_name', doc_id)
        else:
            link_text = doc_id
        
        return f"[{link_text}]({link})"
    
    else:
        raise ValueError(f"Unsupported link type: {link_type}")
```

## Sample Report Generation

Example of generating a summary report:

```python
def generate_sample_summary_report():
    """Generate a sample summary report."""
    # Create report generator
    generator = ReportGenerator()
    
    # Create sample data
    report_data = {
        'title': 'Financial Summary Report',
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'executive_summary': 'This report summarizes financial findings in the construction dispute between Fowler Unified School District and General Contractor. Analysis identified several suspicious patterns, including potential double-billing for HVAC materials and inconsistent labor rates.',
        'key_findings': [
            'Potential double-billing for HVAC materials totaling $16,750.00',
            'Inconsistent labor rates across multiple change orders',
            'Rejected change order items reappearing in later payment applications',
            'Total disputed amount: $97,250.00'
        ],
        'summary': {
            'Original Contract': {'amount': '1,423,150.00', 'notes': 'Original contract amount'},
            'Approved Change Orders': {'amount': '212,345.00', 'notes': '15 approved change orders'},
            'Rejected Change Orders': {'amount': '97,250.00', 'notes': '3 rejected change orders'},
            'Paid to Date': {'amount': '1,512,758.00', 'notes': 'As of 2023-04-15'},
            'Disputed Amount': {'amount': '122,737.00', 'notes': 'In contention'}
        },
        'suspicious_patterns': [
            {
                'type': 'Recurring Amount',
                'amount': '16750.00',
                'confidence': 0.92,
                'explanation': 'The amount $16,750.00 for "HVAC Materials" appears in both a rejected change order and a subsequent payment application with a different description.',
                'evidence': [
                    {'document': 'CO #1 (Rejected)', 'date': '2023-01-15', 'description': 'HVAC Materials'},
                    {'document': 'Payment App #6', 'date': '2023-02-28', 'description': 'Mechanical Equipment'}
                ]
            },
            {
                'type': 'Inconsistent Markup',
                'amount': '12000.00',
                'confidence': 0.85,
                'explanation': 'Inconsistent markup percentages applied to labor rates across multiple change orders, resulting in approximately $12,000.00 in excessive charges.',
                'evidence': [
                    {'document': 'CO #2', 'date': '2023-01-30', 'description': 'Electrical Labor (15% markup)'},
                    {'document': 'CO #3', 'date': '2023-02-15', 'description': 'Electrical Labor (25% markup)'}
                ]
            }
        ],
        'disputed_amounts': [
            {
                'amount': '16750.00',
                'description': 'HVAC Materials / Mechanical Equipment',
                'status': 'Disputed',
                'notes': 'This amount was rejected in CO #1 but appears to have been included in Payment App #6 under a different description.',
                'documents': [
                    {'type': 'Change Order (Rejected)', 'date': '2023-01-15', 'description': 'HVAC Materials - $16,750.00'},
                    {'type': 'Payment Application', 'date': '2023-02-28', 'description': 'Mechanical Equipment - $16,750.00'}
                ]
            },
            {
                'amount': '42000.00',
                'description': 'Electrical Materials and Labor',
                'status': 'Disputed',
                'notes': 'Inconsistent labor rates and markup percentages.',
                'documents': [
                    {'type': 'Change Order', 'date': '2023-01-30', 'description': 'Electrical Labor - $28,000.00'},
                    {'type': 'Payment Application', 'date': '2023-02-28', 'description': 'Electrical Materials and Labor - $42,000.00'}
                ]
            }
        ],
        'recommendations': [
            'Request detailed breakdown of all HVAC and mechanical equipment charges',
            'Dispute Payment Application #6 based on inclusion of previously rejected items',
            'Request detailed labor records to verify rates and hours',
            'Consider requesting audit of all markup calculations'
        ],
        'documents': [
            {'id': 'doc_123', 'type': 'Contract', 'party': 'Both', 'date': '2022-09-01', 'description': 'Original Contract'},
            {'id': 'doc_124', 'type': 'Change Order', 'party': 'Contractor', 'date': '2023-01-15', 'description': 'CO #1 (Rejected)'},
            {'id': 'doc_125', 'type': 'Change Order', 'party': 'Contractor', 'date': '2023-01-30', 'description': 'CO #2'},
            {'id': 'doc_126', 'type': 'Change Order', 'party': 'Contractor', 'date': '2023-02-15', 'description': 'CO #3'},
            {'id': 'doc_127', 'type': 'Payment Application', 'party': 'Contractor', 'date': '2023-02-28', 'description': 'Payment App #6'}
        ]
    }
    
    # Generate report
    report_content = generator.generate_summary_report(report_data)
    
    # Save report to file
    with open('summary_report.md', 'w') as f:
        f.write(report_content)
    
    # Also generate HTML version
    html_content = generator.formatters['html'].format(report_content, 'summary_report.html')
    
    # Generate PDF version
    generator.formatters['pdf'].format(report_content, 'summary_report.pdf')
    
    return 'summary_report.md'
```

## Implementation Guidelines

1. **Template Design**: Design templates for clarity and professionalism
2. **Citation System**: Ensure all findings link back to source documents
3. **Visual Elements**: Use charts and timelines to clarify complex financial data
4. **Accessibility**: Ensure reports are readable by non-technical stakeholders
5. **Evidence First**: Prioritize evidence-backed statements over speculation
6. **Professional Tone**: Maintain a neutral, professional tone

## Dependencies

- jinja2: Template rendering
- markdown: Markdown to HTML conversion
- matplotlib: Chart and timeline generation
- weasyprint: HTML to PDF conversion
- openpyxl: Excel file creation
- pandas: Data manipulation

## Testing Strategy

1. **Unit Tests**: Test individual formatters and components
2. **Template Tests**: Test template rendering with different data
3. **Visual Tests**: Test chart and timeline generation
4. **Integration Tests**: Test end-to-end report generation
5. **Formatting Tests**: Test different output formats

## Security Considerations

1. **Input Validation**: Validate all input data before rendering templates
2. **Output Sanitization**: Sanitize output to prevent XSS in HTML reports
3. **File Path Security**: Prevent path traversal in output file handling
4. **Data Privacy**: Ensure sensitive data is handled appropriately
