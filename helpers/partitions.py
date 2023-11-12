from src.dataloader import create_partition
import src.config as config
import argparse
import os

# Get arguments from command line

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
parser.add_argument('--val_ratio', type=float)
parser.add_argument('--test_ratio', type=float)
parser.add_argument('--output_path', type=str, default=config.INPUT_PATH)

INPUT_PATH = parser.parse_args().input_path
VAL_RATIO = parser.parse_args().val_ratio
TEST_RATIO = parser.parse_args().test_ratio
OUTPUT_PATH = parser.parse_args().output_path

folder_name = f'train_{(1 - VAL_RATIO - TEST_RATIO) * 100:.0f}_{VAL_RATIO * 100:.0f}_{TEST_RATIO * 100:.0f}'
output_partition = os.path.join(OUTPUT_PATH, folder_name)

create_partition(INPUT_PATH, VAL_RATIO, TEST_RATIO, output_partition)
