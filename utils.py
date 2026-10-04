from pathlib import Path

def ensure_path(file_obj):
    if file_obj is None: raise ValueError('Please upload a file or choose a demo sample.')
    if isinstance(file_obj,str): return file_obj
    if hasattr(file_obj,'name'): return file_obj.name
    return str(file_obj)
