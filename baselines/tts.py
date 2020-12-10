from argparse import ArgumentParser
from pathlib import Path
import random


def split(data_dir: Path, val_share: float, test_share: float):
    with open(data_dir / 'ratings_final.txt') as ratings_file:
        ratings = list(ratings_file)

    all_indexes = set(range(len(ratings)))
    test_size = int(test_share * len(ratings))
    val_size = int(val_share * len(ratings))
    test_indexes = random.sample(all_indexes, test_size)
    val_indexes = random.sample(all_indexes - set(test_indexes), val_size)
    train_indexes = sorted(set(all_indexes) - set(val_indexes) - set(test_indexes))
    val_indexes.sort()
    test_indexes.sort()
    train_ratings = [ratings[i] for i in train_indexes]
    val_ratings = [ratings[i] for i in val_indexes]
    test_ratings = [ratings[i] for i in test_indexes]

    with open(data_dir / 'ratings_final_train.txt', 'w') as train_file:
        train_file.write(''.join(train_ratings))
    with open(data_dir / 'ratings_final_val.txt', 'w') as val_file:
        val_file.write(''.join(val_ratings))
    with open(data_dir / 'ratings_final_test.txt', 'w') as test_file:
        test_file.write(''.join(test_ratings))


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('data_dir', type=Path)
    arg_parser.add_argument('--val_share', type=float, default=.2)
    arg_parser.add_argument('--test_share', type=float, default=.2)
    args = arg_parser.parse_args()
    split(args.data_dir, args.val_share, args.test_share)
