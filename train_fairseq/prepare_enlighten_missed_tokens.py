import argparse
import numpy as np
import torch
from bart_with_extractor import ProposedModel
from mytask import ProposalTask

parser = argparse.ArgumentParser()
parser.add_argument("--train-dest-dir", help="path to directory where the checkpoint is saved.", type=str)
parser.add_argument(
    "--dataset",
    required=True,
    choices=[
        "cnn_dm",
        "xsum",
    ],
    help="which dataset is going to be processed.",
)
parser.add_argument(
    "--beam-args",
    choices=[
        "xsum",
        "cnn",
        "least",
    ],
    default="least",
    help="args for beam search.",
)
parser.add_argument("--miss-threshold", default=0.4, help="max threshold of rouge-1.", type=float)
parser.add_argument("--max-freq", default=3, help="max frequency of tokens to be enlightened.", type=int)
args = parser.parse_args()

model = ProposedModel.from_pretrained(
    args.train_dest_dir,
    checkpoint_file="checkpoint_best.pt",
    data_name_or_path=args.train_dest_dir,
)
model.model.encoder.use_topk_result = True
model = model.eval()
if torch.cuda.is_available():
    model = model.cuda().half()

with open("data/{}/test.source".format(args.dataset)) as src, \
open("data/desired_lengths/{}/test.oracle".format(args.dataset)) as oracle, \
open("data/{}/test.target.tokenized".format(args.dataset)) as ref, \
open("data/{}/test.target".format(args.dataset)) as ref_row, \
open("{}/test_{}.hypo_{}_args.tokenized".format(args.train_dest_dir, args.dataset, args.beam_args)) as gen, \
open("{}/test_{}.hypo_{}_args".format(args.train_dest_dir, args.dataset, args.beam_args)) as gen_row:
    src_list = src.readlines()
    oracle_list = oracle.readlines()
    ref_list = ref.readlines()
    ref_row_list = ref_row.readlines()
    gen_list = gen.readlines()
    gen_row_list = gen_row.readlines()
src_list = [sent[:-1] for sent in src_list]
oracle_list = [int(line[:-1]) for line in oracle_list]
ref_list = [sent[:-1].split(" ") for sent in ref_list]
ref_row_list = [sent[:-1] for sent in ref_row_list]
gen_list = [sent[:-1].split(" ") for sent in gen_list]
gen_row_list = [sent[:-1] for sent in gen_row_list]

bad_ids = []
for i in range(len(ref_list)):
    score = len(set(gen_list[i]) & set(ref_list[i])) / len(set(ref_list[i]))
    if score<args.miss_threshold and len(model.encode(src_list[i]))<1024 and len(ref_list[i])<100:
        # print(i, score)
        bad_ids.append(i)
print("bad count:", len(bad_ids))

with open("{}/test_{}_missed{}.source".format(args.train_dest_dir, args.dataset, args.miss_threshold), "w") as src_missed, \
open("{}/test_{}_missed{}.oracle".format(args.train_dest_dir, args.dataset, args.miss_threshold), "w") as oracle_missed, \
open("{}/test_{}_missed{}.target.tokenized".format(args.train_dest_dir, args.dataset, args.miss_threshold), "w") as ref_missed, \
open("{}/test_{}_missed{}.hypo_{}_args.tokenized".format(args.train_dest_dir, args.dataset, args.miss_threshold, args.beam_args), "w") as gen_missed:
    src_missed.writelines([sent+"\n" for i, sent in enumerate(src_list) if i in bad_ids])
    oracle_missed.writelines([str(l)+"\n" for i, l in enumerate(oracle_list) if i in bad_ids])
    ref_missed.writelines([" ".join(tokens)+"\n" for i, tokens in enumerate(ref_list) if i in bad_ids])
    gen_missed.writelines([" ".join(tokens)+"\n" for i, tokens in enumerate(gen_list) if i in bad_ids])

enlighten_indices_list = []
for data_id in bad_ids:
    # print("id:", data_id)
    k = len(ref_list[data_id])
    # print("k:", k)
    count = k
    source = src_list[data_id]

    src_tokens = model.encode(source)

    gen_tokens = model.encode(gen_row_list[data_id])
    ref_tokens = model.encode(ref_row_list[data_id])

    candidate_ids = set(src_tokens.tolist()) & set(ref_tokens.tolist()) - set(gen_tokens.tolist())
    candidate_id_list = np.array(sorted(list(candidate_ids)))

    # for can_id in candidate_id_list:
    #     print(can_id, src_tokens.bincount()[can_id], model.bpe.decode(model.task.source_dictionary.string(torch.tensor([can_id]))))
    
    candidate_indices = []
    for i in candidate_id_list[(src_tokens.bincount()[candidate_id_list]<=args.max_freq).tolist()]:
        candidate_indices.extend(np.argwhere(src_tokens==i)[0].tolist())
    candidate_indices = np.sort(candidate_indices)
    # print("count:", len(candidate_indices), "\n")
    # print(candidate_indices, "\n")
    enlighten_indices_list.append(candidate_indices)

assert len(bad_ids) == len(enlighten_indices_list)

with open("{}/test_{}_missed{}_{}.enlighten_indices_{}_args".format(args.train_dest_dir, args.dataset, args.miss_threshold, args.max_freq, args.beam_args), "w") as enlighten_missed:
    enlighten_missed.writelines([" ".join([str(idx) for idx in enlighten_indices])+"\n" for enlighten_indices in enlighten_indices_list])
