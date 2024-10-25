from pathlib import Path

path = Path(__file__)

breakpoint()
print(path.resolve().parent.parent)