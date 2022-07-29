with open("data/cnn_dm/test.bpe.target", "r") as f:
    data = f.readlines()
data = [s[:-1] for s in data]
tgt_length_list = [len(s.split(" ")) for s in data]
with open("data/desired_lengths/test.oracle", "w") as f:
    f.writelines(["{}\n".format(length) for length in tgt_length_list])