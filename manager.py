from argparse import ArgumentParser, Namespace


def main(arguments: Namespace):
    pass


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()

    main(arguments=args)
