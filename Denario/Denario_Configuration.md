# Denario – Analysis Configuration

This document describes the configuration used for the **Analysis** stage in the Denario workflow.  
The configuration defines how results are generated, reviewed, orchestrated, and formatted.

---

## Model Configuration

### Core Roles and Assigned Models

| Role | Description | Model |
|-----|------------|-------|
| Engineer | Generates and executes code to compute results | `gpt-5` |
| Researcher | Processes results and writes the research report | `gpt-5` |
| Planner | Creates a detailed plan for generating research results | `gpt-4o` |
| Plan Reviewer | Reviews and improves the proposed plan | `gpt-4o` |
| Orchestration | Coordinates the execution flow between components | `gpt-5` |
| Formatter | Formats the final outputs | `o3-mini` |