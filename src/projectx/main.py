import argparse
from pathlib import Path

from langsmith import tracing_context

from projectx.graph_builder import build_graph
from projectx.tool_executor import repository_tree_tool


def repository_directory(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(
            f"Repository directory does not exist: {value}"
        )
    return path.resolve()


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        required=True,
        help="Programming language of the code being reviewed. We do not auto detect.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        type=repository_directory,
        help="Path to the local repository the agent should familiarise itself with.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print the repository tree locally without calling either model.",
    )
    args = parser.parse_args(argv)
    return args


def main():
    command_line_arguments = get_args()

    if command_line_arguments.preview:
        # Preview stays local, including when LangSmith tracing is configured.
        with tracing_context(enabled=False):
            print(
                repository_tree_tool.invoke(
                    {"repository_path": str(command_line_arguments.repo)}
                )
            )
        return

    print(
        "[Info] Running: fetch tree -> read threat-model template -> "
        "familiarise (file reads) -> fill threat model -> reflect "
        "(up to 3 revisions)",
        flush=True,
    )
    graph = build_graph()
    result = graph.invoke(
        {
            "repository_path": str(command_line_arguments.repo),
            "programming_language": command_line_arguments.lang,
        }
    )


if __name__ == "__main__":
    main()
