"""Render the terminal screenshot in the README from real compare-runs output.

    python tools/render_readme_media.py

Writes docs/assets/terminal.svg from the checked-in H100 reliability example, so the
image always shows what the current code prints.
"""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console
from rich.terminal_theme import MONOKAI

from inference_doctor.quality import load_and_compare_directories
from inference_doctor.quality_report import render_quality_report

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "inference-ci-reliability-001"


def main() -> None:
    report = load_and_compare_directories(
        EXAMPLE / "baseline", EXAMPLE / "candidate", metric="texts_per_second"
    )
    console = Console(
        file=io.StringIO(), record=True, width=104, force_terminal=True, color_system="truecolor"
    )
    console.print(
        "[dim]$[/dim] inference-doctor compare-runs --baseline baseline/ "
        "--candidate candidate/ --metric texts_per_second"
    )
    render_quality_report(report, console)
    target = ROOT / "docs" / "assets" / "terminal.svg"
    console.save_svg(str(target), title="inference-doctor", theme=MONOKAI)
    print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
