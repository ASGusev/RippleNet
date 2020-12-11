import csv
from pathlib import Path
from argparse import ArgumentParser


import torch


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


def preprocess_dataset_for_open_ke(args):
    dataset_dir = args.dataset_dir
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

    with open(dataset_dir / 'config.txt', 'w') as config_file:
        config_file.write(f'{n_users} {n_items}\n')


def serialize_emb_lib_fm(embedding, offset):
    return ' '.join(f'{index}:{value}' for index, value in enumerate(embedding, offset))


def rating_list_to_libfm(rating_list_path, out_path, n_users, n_items, user_embeddings, item_embeddings):
    rating_list = load_rating_list(rating_list_path)
    offset_user_emb, offset_item_emb = n_users + n_items, n_users + n_items + user_embeddings.shape[1]
    user_emb_reps = [serialize_emb_lib_fm(user_embedding, offset_user_emb) for user_embedding in user_embeddings]
    item_emb_reps = [serialize_emb_lib_fm(item_embedding, offset_item_emb) for item_embedding in item_embeddings]
    with out_path.open('w') as out_file:
        for user, item, label in rating_list:
            out_file.write(f'{label} {user}:1 {n_users + item}:1 {user_emb_reps[user]} {item_emb_reps[item]}\n')


def prepare_libfm_task(args):
    dataset_dir = args.dataset_dir
    with open(dataset_dir / 'config.txt') as config_file:
        n_users, n_items = map(int, config_file.read().split())
    trans_r_embeddings = torch.load(dataset_dir / 'trans_r.pt')['ent_embeddings.weight'].numpy()
    user_embeddings = trans_r_embeddings[:n_users]
    item_embeddings = trans_r_embeddings[n_users:]
    rating_list_to_libfm(dataset_dir / 'ratings_final_train.txt', dataset_dir / 'train.libfm',
                         n_users, n_items, user_embeddings, item_embeddings)
    rating_list_to_libfm(dataset_dir / 'ratings_final_val.txt', dataset_dir / 'val.libfm',
                         n_users, n_items, user_embeddings, item_embeddings)
    rating_list_to_libfm(dataset_dir / 'ratings_final_test.txt', dataset_dir / 'test.libfm',
                         n_users, n_items, user_embeddings, item_embeddings)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    subparsers = arg_parser.add_subparsers()

    open_ke_parser = subparsers.add_parser('open_ke')
    open_ke_parser.add_argument('dataset_dir', type=Path)
    open_ke_parser.set_defaults(func=preprocess_dataset_for_open_ke)

    lib_fm_parser = subparsers.add_parser('lib_fm')
    lib_fm_parser.add_argument('dataset_dir', type=Path)
    lib_fm_parser.set_defaults(func=prepare_libfm_task)

    arguments = arg_parser.parse_args()
    arguments.func(arguments)
