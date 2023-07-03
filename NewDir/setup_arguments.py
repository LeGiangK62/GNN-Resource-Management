import argparse


def setup_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test", type=int, default=730,
                        help="Just a test argument")

    return parser.parse_args()

