import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode

load_dotenv()


def get_repository_tree(
    repository_path: str,
    max_depth: int = 4,
    max_entries: int = 500,
) -> str:
    """Return a repository's directory tree as text, without reading file contents.

    The root is depth 0; its immediate children are depth 1. Limits apply to
    displayed entries, with markers wherever traversal stops. Symlinks are
    listed without following them. Common generated directories are excluded;
    custom .gitignore rules are not interpreted. Filesystem errors propagate
    to the caller.
    """
    if max_depth < 1 or max_entries < 1:
        raise ValueError("max_depth and max_entries must be positive integers.")

    root = Path(repository_path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {root}")

    excluded_directories = {
        ".git", ".venv", "venv", "__pycache__", "node_modules",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", "build", "dist",
    }
    lines = [f"{root.name}/" if root.name else root.as_posix()]
    entry_count = 0

    def visit(directory: Path, prefix: str, depth: int) -> bool:
        nonlocal entry_count
        children = []
        for path in directory.iterdir():
            is_link = path.is_symlink()
            is_directory = not is_link and path.is_dir()
            if is_directory and path.name in excluded_directories:
                continue
            children.append((path, is_directory, is_link))
        children.sort(key=lambda child: (
            not child[1], child[0].name.casefold(), child[0].name,
        ))

        for index, (path, is_directory, is_link) in enumerate(children):
            if entry_count >= max_entries:
                return False
            is_last = index == len(children) - 1
            branch = "└── " if is_last else "├── "
            label = path.name + ("/" if is_directory else "")
            if is_link:
                label += " [symlink]"
            elif is_directory and depth == max_depth:
                label += " [not expanded: depth limit]"
            lines.append(prefix + branch + label)
            entry_count += 1

            if is_directory and depth < max_depth:
                child_prefix = prefix + ("    " if is_last else "│   ")
                if not visit(path, child_prefix, depth + 1):
                    return False
        return True

    if not visit(root, "", 1):
        lines.append(f"... [tree truncated at {max_entries} entries]")
    return "\n".join(lines)


repository_tree_tool = StructuredTool.from_function(
    get_repository_tree, name="get_repo_tree"
)

# Available for a future model-directed tool loop. The linear graph invokes
# repository_tree_tool directly with the repository selected on the CLI.
execute_tools = ToolNode([repository_tree_tool])

if __name__ == "__main__":
    print(
        f"[INFO] NOT RUNNING Main.py.\n[Info] Now running {os.path.basename(__file__)}"
    )
