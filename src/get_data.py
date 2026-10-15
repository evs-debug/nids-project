import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = ["KDDTrain+.txt", "KDDTest+.txt"]
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = RAW_DIR / name
        if target.exists():
            print(f"{name} already present, skipping")
            continue
        urllib.request.urlretrieve(BASE + name, target)
        print(f"downloaded {name}")


if __name__ == "__main__":
    main()
