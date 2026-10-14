"""Execute with the same interpreter used to launch this script."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks/01_eda_and_modeling.ipynb'
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name='python3')
manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
NotebookClient(notebook, timeout=180, kernel_manager=manager,
               resources={'metadata': {'path': str(root)}}).execute()
nbformat.validate(notebook)
nbformat.write(notebook, path)
print('Executed:', path)
