"""Verify the source and configuration inventory."""
import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    sys.path.insert(0, str(REPOSITORY / "src"))
    from jev.workflow import verify_sources
    try:
        count = verify_sources()
    except (ValueError, KeyError, OSError) as error:
        print("JEV: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps(dict(status="PASS", preserved_files=count)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
