"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """wide-but-small."""


if __name__ == "__main__":
    main(prog_name="widebutsmall")  # pragma: no cover
