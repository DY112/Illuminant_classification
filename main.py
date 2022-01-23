from argparse import ArgumentParser
from solver import Solver

def main(config):
    solver = Solver(config)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--')

    config = parser.parse_args()
    main(config)