"""Allow `python -m llm_clustering` to execute the CLI."""

from llm_clustering.main import main


def run() -> None:
    """Delegate to the existing CLI entry point."""
    main()


if __name__ == "__main__":
    run()


