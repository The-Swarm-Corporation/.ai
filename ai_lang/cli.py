import click
from pathlib import Path
from loguru import logger
import sys
import os
from ai_lang.ai_lang import process_ai_file_sync

# Configure logger
logger.remove()
logger.add(
    sys.stderr, format="{time} {level} {message}", level="INFO"
)


@click.group()
def cli():
    """DotAI - Natural Language Programming Language CLI"""
    pass


@cli.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option(
    "--output", "-o", help="Output directory for generated files"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without executing",
)
def run(filepath: str, debug: bool, output: str, dry_run: bool):
    """Execute a .ai file and generate the requested files/code"""
    if debug:
        logger.remove()
        logger.add(
            sys.stderr,
            format="{time} {level} {message}",
            level="DEBUG",
        )

    if output:
        os.makedirs(output, exist_ok=True)
        os.chdir(output)

    logger.info(f"Processing {filepath}")

    try:
        if dry_run:
            logger.info("Dry run - showing what would be executed:")
            with open(filepath) as f:
                for line in f:
                    logger.info(f"Would process: {line.strip()}")
            return

        results = process_ai_file_sync(filepath)
        logger.info(f"Successfully processed {len(results)} requests")

        for result in results:
            if result["success"]:
                logger.info(f"✓ {result['request']}")
            else:
                logger.error(
                    f"✗ {result['request']}: {result.get('error')}"
                )

    except Exception as e:
        logger.exception(f"Failed to process file: {e}")
        sys.exit(1)


@cli.command()
def version():
    """Show the installed version"""
    import pkg_resources

    version = pkg_resources.get_distribution("dotai").version
    click.echo(f"DotAI version {version}")


@cli.command()
@click.argument("filepath", type=click.Path())
def init(filepath: str):
    """Initialize a new .ai file with example content"""
    if Path(filepath).exists():
        logger.error(f"File {filepath} already exists")
        sys.exit(1)

    example = """# Example .ai file
Create a Python script that prints "Hello World"

Create a JSON file containing a list of 3 colors with their RGB values

Create a markdown file with a simple todo list
"""

    with open(filepath, "w") as f:
        f.write(example)
    logger.info(f"Created example .ai file at {filepath}")


if __name__ == "__main__":
    cli()
