import os
from datasets import load_dataset

xsum = load_dataset("xsum")

os.makedirs("data/xsum", exist_ok=True)

for split in ["test", "validation", "train"]:
    print(split, "collecting...")
    source_list = xsum[split]["document"]
    target_list = xsum[split]["summary"]
    split = "val" if split=="validation" else split
    with open("data/xsum/{}.source".format(split), "w") as source_file:
        source_file.writelines([s.replace("\n", " ")+"\n" for s in source_list])
    with open("data/xsum/{}.target".format(split), "w") as target_file:
        target_file.writelines([s.replace("\n", " ")+"\n" for s in target_list])
