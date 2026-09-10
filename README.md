# ProjectX

A read-only repository familiarisation and preliminary threat-model workflow:

```text
START -> fetch_tree -> read_threatmodel -> familiarise -> fill_threat_model -> reflect
                                         ^                                   |
                                         |------ improve, below cap ----------|
                                                                             |-> END
```

From this project directory:

```bash
uv run projectx --lang Python --repo /path/to/repository
```

`--lang` is passed to the checklist, familiarisation, threat-model, and reflection prompts.
The first node always fetches the CLI-selected repository's directory tree locally.
The next node reads the actual `ThreatModel` JSON Schema and the tree to produce
an inspection checklist. Familiarisation uses that checklist to select files
dynamically; a `ToolNode` executes reads and sends the results back to the model.
The writer fills the existing schema using the overview **and actual excerpts**.
The reflection node then critiques that report against the source evidence.
It uses the existing `Reflection` schema (`missing` and `superfluous`), wrapped
inside a `ReflectionDecision` that adds a strict Boolean `needs_revision` and
a decision `reason`. Reflection critiques the report; it does not rewrite it.

When the model requests an improvement, a conditional edge returns directly to
`familiarise`, which receives the previous report, latest critique, earlier critique
history, overview, collected excerpts, and read failures. It gathers any additional
evidence needed, then the writer revises the report and the critic reviews it again.
The tree and checklist are produced only once per run.

`MAX_REVISIONS = 3` in `graph_builder.py` means **one initial draft plus at most
three revisions** (four reports and four critiques). Revision 0 is the initial
draft. The counter increments only when a valid replacement report is produced;
file reads and tool messages do not increment it. The critic can end the loop
earlier by returning `needs_revision=false`, but cannot override the numeric cap.

The reader's root comes from `--repo`, not a hardcoded directory or a model argument.
The model chooses `file_path` (relative to that root), `start_line`, and `max_lines`.
`AnswerQuestion` finishes the familiarisation step; it is an output schema, not a
filesystem function. The graph prints its Mermaid diagram, including the
reflection node's conditional loop and exit.
The file-reading loop runs inside `familiarise`, so its internal turns are not
separate nodes in that outer diagram.

The result of `graph.invoke(...)` contains `checklist`, `familiarisation`,
`evidence`, `read_errors`, and a validated `threat_model`. It also contains the
latest `reflection`, `needs_revision`, `reflection_reason`, `revision_count`,
and `history`. Each history entry snapshots that iteration's report, critique,
decision, overview, evidence IDs, and read errors. Actual excerpts accumulate in
`evidence`; their source IDs remain stable across passes.

`stop_reason` is `approved` when the critic requests no further revision, or
`max_revisions` when the cap stops a still-requested improvement. `approved` means
the critic accepted this preliminary discovery report or saw no useful in-scope
next step, **not** that the repository is secure or fully assessed. The final
critique remains available even when the cap is reached.

The CLI currently invokes
the graph without printing that result. Invalid or missing structured responses
from either writer or critic stop the graph; there is no automatic schema-correction
retry. Critique-driven improvements happen only after a valid report and review.

To inspect just the tree, without model calls or LangSmith tracing:

```bash
uv run projectx --lang Python --repo /path/to/repository --preview
```

Normal runs read selected text files; preview still lists names only. Each saved
excerpt includes its relative path, line range, exact content, a file-snapshot
SHA-256, and a source ID. The overview and report prompts request source citations,
but citations and interpretations still need human review. Pydantic validates
structure, not truth. Unread files, runtime behaviour, deployed settings, and
unperformed tests or advisory searches must remain unknown/not assessed.

Defaults are deliberately bounded: at most 8 read calls and 10 model turns per
familiarisation pass (`graph_builder.py`); at most 100 lines and 2,000 source characters
per excerpt, with files larger than 256 KiB refused (`tool_executor.py`). The run
fails if the model exceeds its budget or the initial pass finishes without a
successful nonempty read. Revision passes can finish without new reads when the
existing evidence suffices, avoiding unnecessary model/tool work. Refused reads
are returned to the model as tool errors; they do not become
source evidence. These limits do not guarantee whole-repository coverage or impose
a wall-clock deadline on provider requests.
They are not tokenizer-aware context limits either: large trees, checklists,
reports, critiques, and accumulated source evidence must fit the chosen model's
context window. The commented Ollama alternative has `num_ctx=8096`, which may be
too small for larger runs. Provider context handling and model adherence need live
evaluation; no model-call or wall-clock performance guarantee is implied.
Lines longer than the excerpt character limit are available only as prefixes;
line-based pagination cannot recover the remainder of a single long line.

The reader is a restricted `ReadFileTool` subclass. It refuses paths outside the
selected root, symlinks, non-regular/binary files, common generated directories,
and known secret/credential paths such as `.env`. It does not interpret `.gitignore`
or detect every possible embedded secret. Use a curated, secret-free repository
copy; keep it unchanged during a run. Path rules are not an OS-level sandbox.

Your selected OpenRouter configurations and commented Ollama alternatives remain
in the two existing chain modules. `reflection_chain.py` reuses the writer's model
configuration. Normal runs send the tree, checklist, reports, critiques, and selected source
excerpts to the configured cloud model providers and potentially LangSmith when
tracing is enabled; only use repositories approved for that sharing. Configure
provider credentials in your environment or `.env`. No repository code is executed.

Offline tests (no model calls):

```bash
uv run python -m unittest discover -s tests -v
```
