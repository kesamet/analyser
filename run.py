"""
Unified CLI command runner for Portfolio Manager.
Provides short-cut arguments and an interactive selection menu.
"""

import argparse
import subprocess
import sys


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
    print("\n" + "=" * 50)
    print("       PORTFOLIO MANAGER CLI RUNNER       ")
    print("=" * 50)
    print("  1) Download stock data    (pm.download)")
    print("  2) Update fund sheets     (pm.update)")
    print("  3) Generate portfolio     (pm.portfolio)")
    print("  4) Run Streamlit app      (app.py)")
    print("  5) Exit")
    print("=" * 50)

    try:
        choice = input("Select an option (1-5): ").strip()
        print()
        if choice == "1":
            download()
        elif choice == "2":
            update()
        elif choice == "3":
            portfolio()
        elif choice == "4":
            app()
        elif choice == "5" or choice.lower() in ("q", "exit", "quit"):
            print("Goodbye!")
            sys.exit(0)
        else:
            print("⚠️ Invalid choice. Please enter 1, 2, 3, 4, or 5.")
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
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
