"""
Unified CLI command runner for Portfolio Manager.
Provides short-cut arguments and an interactive selection menu.
"""

import argparse
import subprocess
import sys

import questionary


def run_command(cmd: list[str]) -> None:
    """Executes a command using subprocess, sharing the parent stdin/stdout."""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed with exit code {e.returncode}: {' '.join(cmd)}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n\n👋 Process interrupted.")
        sys.exit(0)


def download() -> None:
    """Runs pm.download using uv."""
    print("🚀 Downloading stock data from Yahoo Finance...\n")
    run_command(["uv", "run", "-m", "pm.download"])


def update() -> None:
    """Runs pm.update using uv."""
    print("🔄 Updating fund sheets...\n")
    run_command(["uv", "run", "-m", "pm.update"])


def portfolio() -> None:
    """Runs pm.portfolio using uv."""
    print("📊 Generating portfolio csv files...\n")
    run_command(["uv", "run", "-m", "pm.portfolio"])


def app() -> None:
    """Runs the Streamlit dashboard using uv."""
    print("💻 Starting Streamlit App...\n")
    run_command(["uv", "run", "streamlit", "run", "app.py"])


def show_menu() -> None:
    """Displays an interactive selection menu and runs the selected option."""
    runners = [
        "Download stock data    (pm.download)",
        "Update fund sheets     (pm.update)",
        "Generate portfolio     (pm.portfolio)",
        "Run Streamlit app      (app.py)",
    ]
    choices = [
        questionary.Choice(title=f"{i}: {runner}", value=i) for i, runner in enumerate(runners)
    ]
    choice = questionary.select(
        "Select runner:",
        choices=choices,
    ).ask()

    if choice == 0:
        download()
    elif choice == 1:
        update()
    elif choice == 2:
        portfolio()
    elif choice == 3:
        app()
    else:
        print("👋 Selection cancelled. Goodbye!")

    sys.exit(0)


def main() -> None:
    """Main entrypoint of the runner."""
    parser = argparse.ArgumentParser(
        description="Simplify execution of Portfolio Manager commands."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["download", "update", "portfolio", "app"],
        help="Specific command to run (download, update, portfolio, app). "
        "If omitted, runs in interactive menu mode.",
    )

    args = parser.parse_args()

    if args.command == "download":
        download()
    elif args.command == "update":
        update()
    elif args.command == "portfolio":
        portfolio()
    elif args.command == "app":
        app()
    else:
        while True:
            show_menu()


if __name__ == "__main__":
    main()
