import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Keep this test module offline even when run by itself in a tracing-enabled shell.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

with patch("dotenv.load_dotenv", return_value=False):
    from projectx.tool_executor import (
        MAX_FILE_BYTES,
        MAX_READ_CHARS,
        make_read_file_tool,
    )


class RepositoryReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "repository"
        self.root.mkdir()
        self.reader = make_read_file_tool(str(self.root))
        network = patch(
            "socket.socket.connect", side_effect=AssertionError("No network in tests")
        )
        network.start()
        self.addCleanup(network.stop)

    def read(self, file_path, **kwargs):
        return self.reader.invoke(
            {
                "name": "read_file",
                "args": {"file_path": file_path, **kwargs},
                "id": "read-1",
                "type": "tool_call",
            },
            config={"callbacks": []},
        )

    def test_read_returns_line_numbers_and_source_artifact(self):
        source = "first line\nsecond line\nthird line\n"
        (self.root / "README.md").write_text(source, encoding="utf-8")

        message = self.read("README.md")

        self.assertIsInstance(message, ToolMessage)
        self.assertEqual(message.tool_call_id, "read-1")
        self.assertIn("1: first line\n2: second line\n3: third line", message.content)
        self.assertEqual(
            message.artifact,
            {
                "path": "README.md",
                "start_line": 1,
                "end_line": 3,
                "content": source,
                "truncated": False,
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
            },
        )

    def test_range_and_character_limits_preserve_exact_excerpt(self):
        source = "one\ntwo\nthree\nfour\n"
        (self.root / "source.py").write_text(source, encoding="utf-8")
        message = self.read("source.py", start_line=2, max_lines=2)
        self.assertEqual(message.artifact["content"], "two\nthree\n")
        self.assertEqual(message.artifact["start_line"], 2)
        self.assertEqual(message.artifact["end_line"], 3)
        self.assertTrue(message.artifact["truncated"])

        (self.root / "long.txt").write_text(
            "a" * (MAX_READ_CHARS + 50), encoding="utf-8"
        )
        long_message = self.read("long.txt")
        self.assertEqual(long_message.artifact["content"], "a" * MAX_READ_CHARS)
        self.assertEqual(long_message.artifact["end_line"], 1)
        self.assertTrue(long_message.artifact["truncated"])
        self.assertIn("final displayed line may be partial", long_message.content)

    def test_dynamic_roots_cannot_be_overridden_by_model(self):
        second_root = self.base / "second"
        second_root.mkdir()
        (self.root / "same.txt").write_text("first", encoding="utf-8")
        (second_root / "same.txt").write_text("second", encoding="utf-8")
        second_reader = make_read_file_tool(str(second_root))

        self.assertEqual(self.read("same.txt").artifact["content"], "first")
        self.assertIn("1: second", second_reader.invoke({"file_path": "same.txt"}))
        self.assertEqual(
            set(self.reader.args_schema.model_fields),
            {"file_path", "start_line", "max_lines"},
        )
        with self.assertRaises(ValidationError):
            self.reader.invoke({"file_path": "same.txt", "root_dir": str(second_root)})

    def test_default_and_maximum_line_counts(self):
        (self.root / "many_lines.txt").write_text("line\n" * 120, encoding="utf-8")
        self.assertEqual(self.read("many_lines.txt").artifact["end_line"], 60)
        maximum = self.read("many_lines.txt", max_lines=100)
        self.assertEqual(maximum.artifact["end_line"], 100)
        self.assertTrue(maximum.artifact["truncated"])

    def test_factory_requires_an_existing_directory(self):
        (self.root / "source.py").write_text("source", encoding="utf-8")
        with self.assertRaises(NotADirectoryError):
            make_read_file_tool(str(self.root / "source.py"))
        with self.assertRaises(FileNotFoundError):
            make_read_file_tool(str(self.root / "missing"))

    def test_rejects_absolute_paths_and_traversal(self):
        outside = self.base / "outside.txt"
        outside.write_text("not repository content", encoding="utf-8")
        for request in (
            str(outside),
            "../outside.txt",
            "folder/../../outside.txt",
            "..\\outside.txt",
        ):
            with self.subTest(request=request), self.assertRaises(ToolException):
                self.read(request)

    def test_rejects_file_and_directory_symlinks_including_internal_links(self):
        (self.root / "real.txt").write_text("source", encoding="utf-8")
        nested = self.root / "real_dir"
        nested.mkdir()
        (nested / "source.py").write_text("source", encoding="utf-8")
        (self.root / "file_link.txt").symlink_to(self.root / "real.txt")
        (self.root / "dir_link").symlink_to(nested, target_is_directory=True)
        (self.root / "outside_link").symlink_to(self.base, target_is_directory=True)
        for request in (
            "file_link.txt",
            "dir_link/source.py",
            "outside_link/repository/real.txt",
        ):
            with self.subTest(request=request), self.assertRaises(ToolException):
                self.read(request)

    def test_rejects_known_sensitive_and_generated_paths(self):
        cases = (
            ".env",
            ".env.local",
            "config/credentials.json",
            "secret.yaml",
            "keys/server.pem",
            "keys/server.key",
            ".ssh/id_rsa",
            ".aws/config",
            ".docker/config.json",
            "node_modules/example/index.js",
            ".venv/lib/source.py",
            ".git/config",
        )
        for request in cases:
            location = self.root / request
            location.parent.mkdir(parents=True, exist_ok=True)
            location.write_text("synthetic fixture", encoding="utf-8")
            with self.subTest(request=request), self.assertRaises(ToolException):
                self.read(request)

    def test_rejects_directories_binary_non_utf8_and_oversized_files(self):
        (self.root / "directory").mkdir()
        (self.root / "binary.dat").write_bytes(b"hello\x00world")
        (self.root / "not_utf8.txt").write_bytes(b"\xff\xfe")
        (self.root / "too_large.txt").write_bytes(b"x" * (MAX_FILE_BYTES + 1))
        for request in (
            "directory",
            "binary.dat",
            "not_utf8.txt",
            "too_large.txt",
            "missing.txt",
        ):
            with self.subTest(request=request), self.assertRaises(ToolException):
                self.read(request)

    def test_rejects_fifo_without_waiting_for_a_writer(self):
        os.mkfifo(self.root / "pipe")
        with self.assertRaises(ToolException):
            self.read("pipe")

    def test_rejects_replacement_of_selected_root(self):
        self.root.rename(self.base / "original_repository")
        self.root.mkdir()
        (self.root / "source.py").write_text("replacement", encoding="utf-8")
        with self.assertRaisesRegex(ToolException, "root changed"):
            self.read("source.py")

    def test_empty_file_and_out_of_range(self):
        (self.root / "empty.txt").write_text("", encoding="utf-8")
        message = self.read("empty.txt")
        self.assertEqual(message.artifact["content"], "")
        self.assertEqual(message.artifact["end_line"], 0)
        self.assertFalse(message.artifact["truncated"])
        with self.assertRaises(ToolException):
            self.read("empty.txt", start_line=2)

    def test_argument_validation(self):
        for arguments in (
            {"start_line": 0},
            {"start_line": True},
            {"max_lines": 0},
            {"max_lines": 101},
            {"max_lines": "5"},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValidationError):
                self.read("source.py", **arguments)

    def test_tool_node_returns_artifact_on_success_and_none_on_error(self):
        (self.root / "README.md").write_text("hello", encoding="utf-8")
        node = ToolNode([self.reader], handle_tool_errors=True)
        request = AIMessage(
            content="",
            tool_calls=[
                {"name": "read_file", "args": {"file_path": "README.md"}, "id": "good"},
                {"name": "read_file", "args": {"file_path": ".env"}, "id": "bad"},
            ],
        )
        builder = StateGraph(MessagesState)
        builder.add_node("tools", node)
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        messages = builder.compile().invoke({"messages": [request]})["messages"][1:]
        self.assertEqual(messages[0].status, "success")
        self.assertEqual(messages[0].artifact["content"], "hello")
        self.assertEqual(messages[1].status, "error")
        self.assertIsNone(messages[1].artifact)


if __name__ == "__main__":
    unittest.main()
