"""CLI wrapper around benchmark report command."""
from sensor_classifier.cli import main

if __name__ == "__main__":
    import sys

    main(["report", *sys.argv[1:]])
