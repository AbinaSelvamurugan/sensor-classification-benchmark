"""CLI wrapper around benchmark predict command."""
from sensor_classifier.cli import main

if __name__ == "__main__":
    import sys

    main(["predict", *sys.argv[1:]])
