"""Run main.ipynb's training cells with a Windows-safe worker entry point."""
import json
from pathlib import Path


if __name__ == "__main__":
    notebook_path = Path(__file__).resolve().parents[1] / "main.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    scope = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        # The experiment saves all images and metrics; display is for Jupyter.
        if index == 10:
            continue
        print(f"Executing main.ipynb cell {index}", flush=True)
        exec(compile("".join(cell["source"]), f"{notebook_path}:cell{index}", "exec"), scope)
