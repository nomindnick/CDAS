# Fictional Construction Project

## Oakridge Elementary School Renovation Project

### Project Details
- **Project Name**: Oakridge Elementary School Renovation
- **Project ID**: OESR-2023
- **Location**: 123 Oakridge Lane, Springfield, State 12345
- **Owner**: Springfield School District
- **Contractor**: Summit Construction Group
- **Architect**: Horizon Design Associates
- **Engineer**: TechStructure Engineering
- **Project Manager**: Michael Chen (Owner), Sarah Johnson (Contractor)
- **Contract Value**: $4,250,000.00
- **Start Date**: March 15, 2023
- **Planned Completion**: August 30, 2024
- **Current Status**: In Progress

### Project Scope
1. Classroom wing renovation (10 classrooms)
2. New HVAC system installation
3. Cafeteria expansion
4. Library modernization
5. Administrative offices reconfiguration
6. ADA compliance upgrades
7. Electrical system upgrades
8. Roof replacement
9. Playground equipment installation
10. Site improvements (parking, landscaping)

### Key Subcontractors
1. Eastern Electrical Services - Electrical work
2. Central Mechanical Inc. - HVAC and plumbing
3. Reliable Roofing Solutions - Roofing
4. Modern Interiors LLC - Drywall and interior finishes
5. Precision Concrete & Excavation - Site work
6. TopGrade Landscaping - Landscaping
7. SecureTech Systems - Security and low voltage
8. PremierGlass - Windows and glazing

### Project Issues (Designed Test Cases)

#### Easy Issues
1. Duplicate billing for material delivery in two different payment applications
2. Exact amount match between rejected change order and later invoice line item
3. Missing change order documentation for approved work
4. Simple math errors in payment application calculations

#### Medium Issues
5. Contradictory approval information between correspondence and payment documentation
6. Change order amount that appears on invoice before formal approval
7. Multiple small changes that collectively bypass approval thresholds
8. Sequential change orders that restore previously rejected scope/costs
9. Fuzzy matches between amounts (slight variations like $10,250.75 vs $10,250.00)
10. Recurring pattern of change orders submitted immediately after payment applications

#### Hard Issues
11. Pattern of splitting large items into multiple smaller items across different applications
12. Chronological inconsistencies in document dating vs submission timing
13. Complex substitution where rejected scope reappears with different descriptions
14. Amounts that include hidden fees not authorized in contract
15. Circular references between documents that are difficult to track manually
16. Sequential changes to multiple interconnected line items across several documents
17. Scope changes described differently in different documents but referring to same work
18. Cumulative markup inconsistencies across multiple change orders
19. Time-based patterns showing strategic timing of financial requests
20. Network of relationships between documents that indicates coordination but requires graph analysis to detect