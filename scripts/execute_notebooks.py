"""Execute every checked-in notebook using this Python interpreter."""

import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from scripts.export_derivations import export_derivations

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Run notebooks from scratch and retain their outputs for GitHub viewing."""
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    os.environ.setdefault("IPYTHONDIR", str(ROOT / ".matplotlib" / "ipython"))
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name="python3")
        # Avoid accidentally executing a globally registered unrelated Python.
        manager.kernel_spec.argv = [
            sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
        ]
        client = NotebookClient(notebook, km=manager, timeout=180,
                                resources={"metadata": {"path": str(ROOT)}},
                                record_timing=False)
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        count = sum(c.cell_type == "code" for c in notebook.cells)
        print(f"PASS {path.name}: {count} code cells executed without errors")
    export_derivations()


if __name__ == "__main__":
    main()
