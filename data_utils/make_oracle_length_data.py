import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    choices=[
        "cnn_dm",
        "xsum",
    ],
    help="which dataset is going to be processed.",
)
args = parser.parse_args()

with open("data/{}/test.bpe.target".format(args.dataset), "r") as f:
    data = f.readlines()
data = [s[:-1] for s in data]
tgt_length_list = [len(s.split(" ")) for s in data]

os.makedirs("data/desired_lengths/{}".format(args.dataset), exist_ok=True)

with open("data/desired_lengths/{}/test.oracle".format(args.dataset), "w") as f:
    f.writelines(["{}\n".format(length) for length in tgt_length_list])

with open("data/desired_lengths/{}/test.oracle+20".format(args.dataset), "w") as f:
    f.writelines(["{}\n".format(length+20) for length in tgt_length_list])
with open("data/desired_lengths/{}/test.oracle+10".format(args.dataset), "w") as f:
    f.writelines(["{}\n".format(length+10) for length in tgt_length_list])
with open("data/desired_lengths/{}/test.oracle-10".format(args.dataset), "w") as f:
    f.writelines(["{}\n".format(length-10) for length in tgt_length_list])
with open("data/desired_lengths/{}/test.oracle-20".format(args.dataset), "w") as f:
    f.writelines(["{}\n".format(length-20) for length in tgt_length_list])
