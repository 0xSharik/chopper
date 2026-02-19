import sys
import os

project_root = r"d:\chopper\project\insilico_lab"
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    print("Attempting to import src.gui.md_tab...")
    from src.gui import md_tab
    print("SUCCESS: Imported src.gui.md_tab")
except Exception as e:
    print(f"ERROR: Failed to import src.gui.md_tab: {e}")

try:
    print("Attempting to import src.gui.app...")
    from src.gui import app
    print("SUCCESS: Imported src.gui.app")
except Exception as e:
    print(f"ERROR: Failed to import src.gui.app: {e}")
