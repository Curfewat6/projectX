# ProjectX

A linear, read-only repository familiarisation and initial threat-model workflow:

```text
START -> fetch_tree -> familiarise -> fill_threat_model -> END
```

From this project directory:

```bash
uv run projectx --lang Python --repo /path/to/repository
```

`--lang` is passed to the familiarisation prompt and the threat-model prompt.
The first node fetches the selected repository's directory tree locally. The next
two nodes each invoke their configured LLM chain once. The CLI prints the overview
and the validated `ThreatModel` as JSON. Invalid or missing structured responses
stop the graph; this version does not retry them automatically.

To inspect just the tree, without model calls or LangSmith tracing:

```bash
uv run projectx --lang Python --repo /path/to/repository --preview
```

Only directory and file names are inspected. README text and source contents are
not read, so the resulting report is preliminary and must mark unassessed details
explicitly. The tree has depth/entry limits and exclusions described in
`tool_executor.py`.

The existing Ollama configuration and commented provider alternatives remain in
the two chain modules. Normal runs send the tree and overview to the configured
model providers; configure their credentials in your environment or `.env`.

Offline tests (no model calls):

```bash
uv run python -m unittest discover -s tests -v
```
