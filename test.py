import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description="Image Attribute Modification")

    parser.add_argument(
        "--semantic_editting",
        type=lambda x: x == "True",
        default=False,
        help="Whether to use semantic editting",
    )

    return parser


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    print(args)
