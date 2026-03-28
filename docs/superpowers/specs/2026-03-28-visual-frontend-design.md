# Visual Frontend Design

## Summary

This document defines a Gradio-based visual frontend upgrade for the project. The new frontend remains within the current Python application stack and replaces the existing command-form web UI with a structured single-page application built on `gr.Blocks`.

The goal is to support two user roles in one interface:

- End users asking historical questions and reviewing cited evidence
- Developers or operators managing initialization, indexing, deletion, and debug queries

The scope is intentionally limited to a practical first version. It does not introduce a separate frontend framework, persistent conversation storage, or graph-style visualizations.

## Goals

- Keep the existing `Gradio` deployment model
- Replace the current form-based web UI with a single-page, two-tab interface
- Provide a user-facing Q&A tab with readable answers and source evidence cards
- Preserve full management capabilities in a separate knowledge-base management tab
- Return structured results to the UI instead of relying on terminal `print` output
- Improve error visibility and state awareness in the web interface

## Non-Goals

- No migration to React, Vue, or a separate frontend/backend architecture
- No new database for sessions, analytics, or UI state persistence
- No advanced relationship graph, timeline, or knowledge-map visualization
- No broad refactor of `cli.py` outside small reuse-oriented adjustments if required

## Current State

The current web UI in [gradioui.py](/home/sopuser/code/history_rag/gradioui.py) is a thin wrapper around CLI-style operations:

- One initialization form
- One index build/delete form
- One query form

This structure has several limitations:

- The page is operation-oriented instead of workflow-oriented
- Shared application state is implicit and fragile
- Query output is not structured for visual presentation
- Debug information is printed to terminal instead of returned to the page
- User-facing Q&A and management workflows are mixed without clear separation

## Proposed Approach

Use a single `gr.Blocks` application as the new web frontend. The page contains two top-level tabs:

- `问答助手`
- `知识库管理`

The application keeps a shared state object for:

- Current config path
- Current mode (`milvus` or `pipeline`)
- Initialized executor instance
- Initialization status
- Debug toggle and last operation status when useful

The UI layer becomes responsible for coordinating user actions and rendering structured results. Core retrieval and index logic remains in `executor.py`.

## Architecture

### 1. UI Layer

`gradioui.py` is refactored into a real page application instead of three `gr.Interface` blocks.

Responsibilities:

- Build the two-tab layout
- Hold shared Gradio state
- Route button actions to UI adapter functions
- Render status, answer text, evidence cards, and debug output
- Disable or guard actions when initialization has not completed

### 2. UI Adapter Functions

The page should use a thin adapter layer inside `gradioui.py` or a nearby helper module to translate UI events into executor calls.

Responsibilities:

- Initialize the correct executor from config and mode
- Normalize success and failure payloads for the UI
- Convert query results into answer, source list, and debug summary fields
- Convert build/delete operations into explicit status messages

### 3. Executor Layer

[executor.py](/home/sopuser/code/history_rag/executor.py) continues to own:

- Index creation
- Index loading
- Retrieval
- Reranking
- Final answer generation

The executor must expose structured methods for UI consumption instead of only returning raw response objects or printing to stdout.

## Page Structure

### Tab 1: 问答助手

This tab is optimized for end-user question answering.

Sections:

1. Session controls
   - Config path input
   - Mode selector (`milvus` / `pipeline`)
   - Initialize button
   - Current status text

2. Question input
   - Main question textbox
   - Submit button
   - Optional toggles such as:
     - show evidence
     - debug mode

3. Result area
   - Answer panel for the final response
   - Evidence card area for supporting source passages
   - Optional debug summary panel when enabled

Expected behavior:

- The query action is disabled or guarded until initialization succeeds
- The answer is displayed as the primary result
- Evidence appears as readable cards, not raw serialized objects
- Errors are shown inline in the page

### Tab 2: 知识库管理

This tab is optimized for developers, demos, and operational workflows.

Sections:

1. Initialization area
   - Config path
   - Mode selector
   - Initialize button
   - Active mode/status summary

2. Build index area
   - File or directory path input
   - Overwrite checkbox
   - Build action
   - Status/result message

3. Delete index area
   - File path input
   - Delete action
   - Deleted count/result message

4. Debug query area
   - Question textbox
   - Query action
   - Answer panel
   - Debug retrieval output

Expected behavior:

- All existing management capabilities remain available
- Results are visible inside the page instead of terminal-only output
- Actions are blocked or fail clearly when not initialized

## Query Result Model

The key change for the frontend is to return a structured query payload.

Recommended shape:

```python
{
    "ok": True,
    "status": "success",
    "answer": "...",
    "sources": [
        {
            "title": "三国志",
            "content": "...",
            "score": 0.87,
            "rank": 1,
            "file_name": "baihuasanguozhi.txt",
        }
    ],
    "debug": {
        "mode": "milvus",
        "retrieved_count": 5,
        "used_count": 3,
        "notes": "...",
    },
    "error": None,
}
```

This model can be adapted to actual executor constraints, but the interface contract should preserve these concepts:

- `ok` for success/failure
- `status` for short UI display
- `answer` for the final text
- `sources` for evidence cards
- `debug` for optional technical detail
- `error` for inline failure reporting

## Evidence Card Design

Evidence cards are the primary visual enhancement in the Q&A tab.

Each card should display:

- Source title or inferred book name when available
- Passage content used to support the answer
- Rank order
- Optional file name
- Optional similarity or rerank score if it can be obtained cleanly

Design principles:

- Prioritize readability over density
- Keep card ordering aligned with retrieval/rerank importance
- Show enough source text to verify the answer without overwhelming the page
- Avoid exposing raw internal object dumps

## Data Flow

### Initialization

1. User chooses config path and mode
2. UI calls initializer
3. Initializer creates the appropriate executor
4. Executor builds or loads query engine
5. UI stores initialized state and updates status display

### User Q&A Flow

1. User submits a question in `问答助手`
2. UI validates initialization state
3. UI calls structured query method on executor
4. Executor retrieves nodes, optionally captures debug data, and generates answer
5. UI renders:
   - answer text
   - evidence cards
   - optional debug summary

### Management Flow

1. User triggers build/delete/debug actions in `知识库管理`
2. UI validates initialization state
3. UI calls corresponding structured executor method
4. Executor performs action and returns structured result
5. UI renders clear operation outcome

## Required Code Changes

### [gradioui.py](/home/sopuser/code/history_rag/gradioui.py)

Primary refactor target.

Changes:

- Replace multiple `gr.Interface` instances with a single `gr.Blocks` app
- Add tabbed layout
- Add shared state management
- Add structured rendering for answer and evidence
- Add inline status and error surfaces
- Keep initialization and management controls available

### [executor.py](/home/sopuser/code/history_rag/executor.py)

Needs targeted API additions for the UI.

Changes:

- Add structured query method for UI use
- Capture retrieved source data for evidence cards
- Return debug information without depending on console `print`
- Return structured results for build/delete operations
- Keep existing core retrieval/generation behavior intact

### [docs/web_ui.md](/home/sopuser/code/history_rag/docs/web_ui.md)

Update documentation to reflect:

- New two-tab layout
- New Q&A workflow
- Evidence card display
- Management operations and expected outputs

## Error Handling

The UI should explicitly handle and display:

- Initialization failure due to config or model setup issues
- Query before initialization
- Missing file/directory during index build
- Unsupported file type
- Missing or empty index
- Delete operations with zero matches
- Backend exceptions from retrieval or generation

Error handling rules:

- Return errors to the page, not only to terminal
- Keep failure messaging concise and actionable
- Avoid exposing stack traces in the normal UI

## Testing Strategy

Manual validation is sufficient for this scope.

Required checks:

1. Launch the new Gradio UI successfully
2. Verify both tabs render correctly
3. Initialize with a valid config in both supported modes when possible
4. Build index from a sample text file
5. Delete an indexed file and verify feedback
6. Run a normal Q&A query and verify answer display
7. Run a query with evidence display and verify source cards
8. Run a debug query and verify debug details stay readable
9. Verify pre-initialization guard behavior
10. Verify inline error messages for invalid paths or failed actions

## Implementation Boundaries

This design intentionally avoids scope creep.

Out of scope for this iteration:

- Multi-page navigation or routing
- Persistent chat history
- Authentication
- User uploads with long-running task tracking
- Retrieval analytics dashboards
- Graph, timeline, or entity-relation visualization

## Open Decisions Resolved

The following product decisions are already fixed for this iteration:

- Frontend stack: continue with `Gradio`
- Page organization: single page with two tabs
- Q&A visual priority: source evidence cards
- Management scope: preserve full existing management capabilities

## Recommended Implementation Order

1. Refactor `gradioui.py` into a shared-state `gr.Blocks` app
2. Add structured result methods in `executor.py`
3. Render answer and evidence cards in the Q&A tab
4. Convert management actions to inline structured feedback
5. Update `docs/web_ui.md`
6. Run manual UI verification

## Risks

- Existing executor methods currently mix return values with terminal output, so UI adaptation may reveal hidden assumptions
- `PipelineExecutor` behavior may not match `MilvusExecutor` feature parity for evidence/debug output
- Some source metadata fields may not be consistently available across all loaded documents

Mitigations:

- Keep existing executor behavior available where practical
- Normalize missing metadata in the UI adapter layer
- Start with a minimal but stable debug payload rather than exposing internals directly

## Success Criteria

The feature is considered successful when:

- The UI launches as a single Gradio page with two tabs
- Users can initialize and ask questions without using CLI-style command forms
- Answers appear with readable supporting evidence cards
- Management actions remain available and provide inline feedback
- Common failures are visible in the page without relying on terminal inspection
