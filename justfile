set shell := ["powershell.exe", "-Command"] # Windows
# set shell := ["bash", "-c"] # Linux/Mac

test:
    uv run pytest
    # uv run --python 3.10 pytest
    