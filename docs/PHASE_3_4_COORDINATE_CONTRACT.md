# Phase 3.4 Coordinate Contract Specification

## Canonical Bounding Box Contract
All bounding boxes in the pipeline follow the canonical coordinate contract:
`[x1, y1, x2, y2]`

## Validation Rule
`validate_bounding_box_contract(bbox, width, height)` ensures:
- `0 <= x1 < x2 <= width`
- `0 <= y1 < y2 <= height`

Strictly enforced across subject extraction, rigid warping, and diagnostic metrics.
