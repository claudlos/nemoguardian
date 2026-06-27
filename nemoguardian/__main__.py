"""Allow `python -m nemoguardian` to invoke the CLI."""

from nemoguardian.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
