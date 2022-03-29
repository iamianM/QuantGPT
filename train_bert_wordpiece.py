import os
import random
import argparse
import shutil
from glob import glob
from pathlib import Path

from lm_dataformat import Reader
from tokenizers import Tokenizer, models
from tqdm import tqdm

# parser

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, help="Path to where your files are located. Files ending in .zst are treated as \
                    archives, all others as raw text.")
parser.add_argument("--output_dir", type=str, default="tokenizers", help="Where to put the tokenizer")
parser.add_argument("--file_type", type=str, choices=["xz", "txt"], default="xz", help="Extension of file to parse")
parser.add_argument("--min_frequency", type=int, default=2, help="min frequency of word count")
args = parser.parse_args()

# main script

data_path = Path(args.base_dir)
archives = glob(str(data_path / f"*.{args.file_type}"))

out_path = Path(args.output_dir)

if os.path.exists(out_path):
    shutil.rmtree(out_path)

if not out_path.is_dir():
    out_path.mkdir()

    for arch in tqdm(archives):
        name = os.path.basename(arch).split(".")[0] + ".txt"
        fp = out_path / name

        if args.file_type == 'xz':
            g = Reader(arch).stream_data()

            with open(fp, "w") as f:
                for s in g:
                    f.write(s)
                    f.write("\n\n")
        elif args.file_type == 'txt':
            shutil.copyfile(str(arch), str(fp))

data_files = glob(str(out_path / "*.txt"))
data_files = random.sample(data_files, int(0.2 * len(data_files)))

# Initialize an empty tokenizer
tokenizer = Tokenizer(models.WordPiece(max_input_chars_per_word=3))

# And then train
tokenizer.train(
    data_files,
)

# Save the files
tokenizer.save(args.out + '/' + args.name)
