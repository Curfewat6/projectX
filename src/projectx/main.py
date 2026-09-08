import argparse
from pathlib import Path


def repository_directory(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(
            f"Repository directory does not exist: {value}"
        )
    return path.resolve()


def get_args():
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
        help="Print the overview tool's output locally without calling the model.",
    )
    args = parser.parse_args()
    return args


def get_threat_model(lang):
    from projectx.threat_model_chain import threat_model_prompt_template as base_prompt

    threat_model_prompt = base_prompt.partial(programming_language=lang)
    return threat_model_prompt


def main():
    command_line_arguments = get_args()
    user_supplied_programming_language = command_line_arguments.lang

    from projectx.familiaraisation_chain import (
        build_repository_overview,
        familiarise_repository,
    )

    if command_line_arguments.preview:
        print(build_repository_overview(command_line_arguments.repo))
        return

    print("We are BACK")
    print("[Info] Starting repository familiarisation...")
    overview = familiarise_repository(
        command_line_arguments.repo,
        user_supplied_programming_language,
    )
    print(overview.text)


if __name__ == "__main__":
    main()
