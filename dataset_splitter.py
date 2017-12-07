import numpy as np
import argparse
import sys
import os
from shutil import copy2

def copy_dataset(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    os.mkdir(os.path.join(output_directory, 'train'));
    os.mkdir(os.path.join(output_directory, 'test'));

    for path, _, file_list in os.walk(input_directory):
        if file_list == []:
            continue

        label = os.path.basename(path)

        file_list = [os.path.join(path, x) for x in file_list]

        random_set = np.random.permutation(len(file_list))
        train_list = random_set[:round(len(random_set)*0.8)]
        test_list = random_set[-(len(file_list) - len(train_list))::]

        train_images = [file_list[index] for index in train_list]
        test_images = [file_list[index] for index in test_list]
        
        train_dir = os.path.join(output_directory, 'train', label)
        test_dir = os.path.join(output_directory, 'test', label)
        os.mkdir(train_dir)
        os.mkdir(test_dir)

        for image in train_images:
            copy2(image, train_dir)

        for image in test_images:
            copy2(image, test_dir)
        

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--input_dir")
    a.add_argument("--out_dir")

    args = a.parse_args()
    if args.input_dir is None or args.out_dir is None:
        a.print_help()
        sys.exit(1)

    copy_dataset(args.input_dir, args.out_dir)