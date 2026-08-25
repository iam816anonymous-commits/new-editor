# Phase 3.4 Disocclusion Specification

## Overview
Primary subject interior pixels are strictly protected from background inpainting or disocclusion hole filling.

## Implementation Details
1. `PRIMARY_SUBJECT_INTERIOR` designated as a protected non-inpaint layer.
2. `protect_primary_subject_interior` erodes mask to isolate rigid interior core.
3. `PersistentBackgroundCanvas` maintains temporal background plate across frame sequence to avoid Telea inpainting flicker.
