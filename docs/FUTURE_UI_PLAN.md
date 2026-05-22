# Future Interactive UI Plan

This document outlines how the CSV analytics reporter could evolve into an interactive graph-building experience. **Do not build this until the PDF/report engine is solid and stable.**

## Current State

The CLI pipeline already produces:

- Structured analysis and quality scoring
- Auto-selected charts with rich metadata
- A polished PDF analytics brief

Each generated chart includes metadata designed for future UI dropdowns:

- `chart_type`, `x_column`, `y_column`, `group_column`
- `available_x_columns`, `available_y_columns`, `available_group_columns`
- `supported_chart_types`
- `title`, `caption`, `insight_label`, `reason_selected`, `caution`, `layout`

That metadata is the bridge between today's automated reports and tomorrow's interactive builder.

## Phase 1 — Streamlit Prototype (Future)

A lightweight Streamlit app could:

1. Upload a CSV (or select a local file path)
2. Show the same KPI cards and executive summary as the PDF
3. Render charts from existing `chart_metadata`
4. Export the current view back to PDF via the existing `create_pdf_report()` function

**Why Streamlit first:** fast to prototype, no React build chain, reuses the current Python modules directly.

## Phase 2 — Dropdown Chart Builder (Future)

Inspired by simple BI tools, not full JMP yet:

| Control | Metadata source |
|---------|-----------------|
| X variable | `available_x_columns` |
| Y variable | `available_y_columns` |
| Group by | `available_group_columns` |
| Chart type | `supported_chart_types` |

When the user changes a dropdown, the UI would:

1. Validate the combination (reuse `assess_grouped_comparison()` and chart planning rules)
2. Regenerate a single chart via `chart_generator.py`
3. Refresh metadata for that chart slot
4. Optionally add the chart to an export list

## Phase 3 — JMP-Inspired Drag-and-Drop (Long Term)

A richer web UI could eventually support:

- Dragging variables onto X, Y, and Group zones
- Live chart preview as variables are assigned
- Saving chart "recipes" as JSON (same metadata structure)
- Exporting selected charts to PDF

This is intentionally deferred. Drag-and-drop graph builders require stable chart metadata, consistent rendering, and predictable validation rules — all of which the CLI pipeline is establishing now.

## Architecture Constraint

Keep the function-based module layout:

```
data_loader → analyzer → insight_generator → chart_generator → pdf_report
```

Any future UI should **call these modules**, not duplicate their logic. The CLI must remain the reference implementation and regression test.

## Warning

Do **not** start the interactive UI until:

- [ ] PDF layout is product-quality across varied datasets
- [ ] Chart metadata covers all chart types consistently
- [ ] Validation rules are shared between auto-selection and manual builder paths
- [ ] CLI commands continue to work without a web server

Build the report engine first. Build the UI second.
