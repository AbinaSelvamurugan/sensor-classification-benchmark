"""CLI wrapper around benchmark evaluate command."""
from sensor_classifier.cli import main

if __name__ == "__main__":
    import sys

    main(["evaluate", *sys.argv[1:]])
