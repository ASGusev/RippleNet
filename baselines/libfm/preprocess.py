import csv
import sys
from pathlib import Path


def load_rating_list(path):
    with path.open() as ratings_file:
        ratings_reader = csv.reader(ratings_file, delimiter='\t')
        ratings_list = [tuple(map(int, r)) for r in ratings_reader]
    return ratings_list


def rating_list_to_open_ke_format(rating_list_path, out_path, n_users):
    rating_list = load_rating_list(rating_list_path)
    rating_list = [r[:2] for r in rating_list if r[2] == 1]
    with open(out_path, 'w') as out_file:
        out_file.write(f'{len(rating_list)}\n')
        for user, item in rating_list:
            out_file.write(f'{user} {item + n_users} 0\n')


def preprocess_dataset_for_open_ke(dataset_dir):
    all_ratings = load_rating_list(dataset_dir / 'ratings_final.txt')
    users, items = set(), set()
    for user, item, _ in all_ratings:
        users.add(user)
        items.add(item)
    users, items = sorted(users), sorted(items)
    n_users = len(users)
    n_items = len(items)

    item_entity_indexes = [i + n_users for i in items]
    with open(dataset_dir / 'entity2id.txt', 'w') as entities_file:
        entities_file.write(f'{n_users + len(items)}\n')
        for user in users:
            entities_file.write(f'{user}\t{user}\n')
        for i in item_entity_indexes:
            entities_file.write(f'{i}\t{i}\n')
    with open(dataset_dir / 'relation2id.txt', 'w') as relations_file:
        relations_file.write('1\n0\t0\n')
    with open(dataset_dir / 'type_constrain.txt', 'w') as constraint_file:
        constraint_file.write('1\n')
        heads = "\t".join(map(str, users))
        tails = "\t".join(map(str, item_entity_indexes))
        constraint_file.write(f'0\t{n_users}\t{heads}\n')
        constraint_file.write(f'0\t{n_items}\t{tails}\n')

    rating_list_to_open_ke_format(dataset_dir / 'ratings_final_train.txt', dataset_dir / 'train2id.txt', n_users)
    rating_list_to_open_ke_format(dataset_dir / 'ratings_final_val.txt', dataset_dir / 'valid2id.txt', n_users)
    rating_list_to_open_ke_format(dataset_dir / 'ratings_final_test.txt', dataset_dir / 'test2id.txt', n_users)


def prepare_lib_fm_task():
    pass  # TODO: implement


if __name__ == '__main__':
    preprocess_dataset_for_open_ke(Path(sys.argv[1]))
