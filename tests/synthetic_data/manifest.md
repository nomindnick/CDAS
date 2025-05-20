# Synthetic Test Data Manifest

This directory contains synthetic test data for evaluating the Construction Document Analysis System (CDAS). The data represents various documents from a fictional construction project with intentionally embedded issues of varying difficulty.

## Project Information

- **Project Name**: Oakridge Elementary School Renovation
- **Project ID**: OESR-2023
- **Contract Value**: $4,250,000.00

See `project_info.md` for detailed project information.

## Document Types

### Contracts
- `OESR-2023_PrimeContract.txt` - Prime contract between Owner and Contractor

### Payment Applications
- `OESR-2023_PayApp_01.txt` - Initial payment application (March-April 2023)
- `OESR-2023_PayApp_02.txt` - Second payment application with duplicate billing issue (May 2023)
- `OESR-2023_PayApp_03.txt` - Third payment application with suspicious line items (June 2023)
- `OESR-2023_PayApp_04.txt` - Fourth payment application showing previously rejected items (July 2023)

### Change Orders
- `OESR-2023_CO_01.txt` - First change order for electrical panel upgrade (approved)
- `OESR-2023_CO_02.txt` - Second change order for structural reinforcement (rejected)
- `OESR-2023_CO_03.txt` - Third change order for equipment pad (approved)
- `OESR-2023_CO_04.txt` - Fourth change order for waterproofing (approved)
- `OESR-2023_CO_05.txt` - Fifth change order for structural support (approved)
- `OESR-2023_CO_06.txt` - Sixth change order for seismic bracing (approved)

### Correspondence
- `OESR-2023_CORR_01.txt` - Rejection letter for Change Order #2
- `OESR-2023_CORR_02.txt` - Letter from architect about structural requirements
- `OESR-2023_CORR_03.txt` - Contractor response with revised approach

### Invoices
- `OESR-2023_INV_Central_Mechanical.txt` - Subcontractor invoice showing rejected items

## Embedded Issues

### Easy Issues
1. **Duplicate billing for material delivery** - HVAC equipment delivery billed twice in Payment Application #2
2. **Exact amount match between rejected change order and invoice line item** - Rejected Change Order #2 amount appears in the subcontractor invoice
3. **Missing change order documentation** - Items 19-29 in Payment App #3 lack formal change order approval
4. **Simple math errors** - Calculation errors in payment applications

### Medium Issues
5. **Contradictory approval information** - Architect's letter contradicts owner's rejection
6. **Change order amount before approval** - Items appear on invoice before formal approval
7. **Multiple small changes bypassing thresholds** - Change Orders #3-6 all under $5,000 requiring only PM approval
8. **Sequential change orders restoring rejected costs** - CO #3-6 collectively implement the scope rejected in CO #2
9. **Fuzzy matches between amounts** - Slight variations in amounts across documents
10. **Pattern of change orders after payments** - Change orders strategically timed after payment applications

### Hard Issues
11. **Splitting large items** - Large rejected change order split into multiple smaller items
12. **Chronological inconsistencies** - Submission dates vs processing dates show suspicious patterns
13. **Complex substitution** - Rejected scope reappears with different descriptions
14. **Hidden fees** - Markup percentage inconsistencies across change orders
15. **Circular references** - Documents referring to each other in circular patterns
16. **Sequential interconnected changes** - Related line items modified across multiple documents
17. **Inconsistent scope descriptions** - Same work described differently in different documents
18. **Markup inconsistencies** - Varying markup percentages across related change orders
19. **Strategic timing patterns** - Strategic timing of submissions and approvals
20. **Coordination network** - Pattern of relationships indicating coordination

## Test Script

Run the synthetic tests using:

```bash
python run_synthetic_tests.py --reset
```

This will:
1. Reset the database (if --reset flag is used)
2. Process all documents
3. Run financial analysis
4. Perform network analysis
5. Evaluate the system's detection rate against known issues