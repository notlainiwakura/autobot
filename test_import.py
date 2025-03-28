import sys
print("Python path:", sys.path)
try:
   from confluence_client import ConfluenceClient
   print("Import successful")
except ImportError as e:
   print(f"Import error: {e}")
