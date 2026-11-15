#!/usr/bin/env python3
"""Build DOCX reports, refresh README metrics, and regenerate the results notebook."""

import shutil
from pathlib import Path

from config import (
    METRICS_SUMMARY_PATH,
    NOTEBOOK_PATH,
    PROJECT_DIR,
    README_PATH,
    REPORT_PATH,
    ensure_runtime_dirs,
    resolve_report_template,
)
from generate_report_docx import build_report
from report_metrics import build_readme_metrics_markdown, load_summary

README_METRICS_START = "<!-- METRICS_TABLE_START -->"
README_METRICS_END = "<!-- METRICS_TABLE_END -->"

# Legacy notebook path at repo root (some README / submission layouts)
NOTEBOOK_PATH_LEGACY = PROJECT_DIR / "PneumoniaMNIST_Analiza_i_Modelowanie.ipynb"


def _check_prerequisites() -> None:
    missing = []
    if not METRICS_SUMMARY_PATH.is_file():
        missing.append(f"metrics: {METRICS_SUMMARY_PATH} (run run_pipeline.py)")
    try:
        resolve_report_template()
    except FileNotFoundError as exc:
        missing.append(str(exc))
    if missing:
        raise FileNotFoundError(
            "Nie można zbudować raportu — brakuje:\n  - " + "\n  - ".join(missing)
        )


def refresh_readme_metrics_table() -> None:
    summary = load_summary()
    table_md = build_readme_metrics_markdown(summary)
    readme_path = README_PATH
    text = readme_path.read_text(encoding="utf-8")
    if README_METRICS_START not in text or README_METRICS_END not in text:
        print("[WARN] README nie ma znaczników METRICS_TABLE — pomijam aktualizację tabeli.")
        return
    before, rest = text.split(README_METRICS_START, 1)
    _, after = rest.split(README_METRICS_END, 1)
    new_text = f"{before}{README_METRICS_START}\n{table_md}\n{README_METRICS_END}{after}"
    readme_path.write_text(new_text, encoding="utf-8")
    print(f"[INFO] Updated metrics table in {readme_path.resolve()}")


def mirror_notebook_to_root() -> None:
    if not NOTEBOOK_PATH.is_file():
        return
    shutil.copy2(NOTEBOOK_PATH, NOTEBOOK_PATH_LEGACY)
    print(f"[INFO] Notebook copy: {NOTEBOOK_PATH_LEGACY.resolve()}")


def main() -> None:
    ensure_runtime_dirs()
    _check_prerequisites()

    build_report()
    refresh_readme_metrics_table()

    from build_notebook import build_notebook

    build_notebook()
    mirror_notebook_to_root()

    for p in (REPORT_PATH, NOTEBOOK_PATH):
        if p.is_file():
            print(f"[OK] {p.resolve()} ({p.stat().st_size // 1024} KiB, mtime={p.stat().st_mtime:.0f})")


if __name__ == "__main__":
    main()