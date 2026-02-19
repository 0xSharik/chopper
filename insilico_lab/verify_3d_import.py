
try:
    import py3Dmol
    print("✅ py3Dmol imported successfully")
except ImportError as e:
    print(f"❌ py3Dmol import failed: {e}")

try:
    from stmol import showmol
    print("✅ stmol imported successfully")
except ImportError as e:
    print(f"❌ stmol import failed: {e}")
