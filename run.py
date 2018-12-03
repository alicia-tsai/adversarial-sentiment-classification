import argparse

from data_loader import DataLoader
from classifier import train_classifier


def main():
    parser = argparse.ArgumentParser(description='Sentiment classifier argument parser.')
    parser.add_argument('--build_data', action='store_true',
                        help='build and save data sets (only needed for the first time).')
    parser.add_argument('--alg', choices=['CNN', 'BiLSTM', 'BiRNN'], required=True,
                        help='algorithm to train the sentiment classifier')
    parser.add_argument('--small_subsets', action='store_true',
                        help='train and evaluate on smaller subsets of the data.')
    parser.add_argument('--outfile', type=str,
                        help='output file name to save trained model.')
    args = parser.parse_args()

    # build and save data for the first time
    if args.build_data:
        DataLoader.build_data()

    # load data from file
    data_loader = DataLoader()
    data_loader.load_data()

    # train model
    train_classifier(alg=args.alg, data_loader=data_loader, small_subsets=args.small_subsets, outfile=args.outfile)


if __name__ == '__main__':
    main()
