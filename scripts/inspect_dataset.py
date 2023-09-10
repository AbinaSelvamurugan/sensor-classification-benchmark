"""CLI wrapper around benchmark inspect command."""
from sensor_classifier.cli import main

if __name__ == "__main__":
    import sys

    main(["inspect", *sys.argv[1:]])
