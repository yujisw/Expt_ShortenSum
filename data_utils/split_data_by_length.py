import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lower", help="The lower limit of target length when splitting", type=int)
parser.add_argument("--upper", help="The upper limit of target length when splitting", type=int)
args = parser.parse_args()

assert args.lower is not None or args.upper is not None
print("start to make split data whose target is {}<= & <={} in length".format(args.lower if args.lower else "None", args.upper if args.upper else "None"))

for mode in ["train", "val", "test"]:
    print("splitting {} data...".format(mode))

    with open("data/cnn_dm/{}.bpe.source".format(mode), "r") as f:
        source = f.readlines()
    source = [s[:-1] for s in source]
    with open("data/cnn_dm/{}.bpe.target".format(mode), "r") as f:
        target = f.readlines()
    target = [s[:-1] for s in target]

    if args.lower is None:
        indices_split = [i for i, sent in enumerate(target) if len(sent.split(' '))<=args.upper]
    elif args.upper is None:
        indices_split = [i for i, sent in enumerate(target) if len(sent.split(' '))>=args.lower]
    else:
        indices_split = [i for i, sent in enumerate(target) if len(sent.split(' '))>=args.lower and len(sent.split(' '))<=args.upper]
    print(len(indices_split))

    target_split = [sent+"\n" for i, sent in enumerate(target) if i in indices_split]
    source_split = [sent+"\n" for i, sent in enumerate(source) if i in indices_split]
    print(len(target_split))
    print(len(source_split))
    
    length_range_description = "{}-{}".format(args.lower if args.lower else "", args.upper if args.upper else "")
    print("creating {0}.{1}.bpe.source & {0}.{1}.bpe.target...".format(mode, length_range_description))
    with open("data/cnn_dm/{}.{}.bpe.source".format(mode, length_range_description), "w") as f:
        f.writelines(source_split)
    with open("data/cnn_dm/{}.{}.bpe.target".format(mode, length_range_description), "w") as f:
        f.writelines(target_split)
