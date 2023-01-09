# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.models.bart import BARTModel
import argparse

from bart_with_extractor import ProposedModel
from mytask import ProposalTask

XSUM_KWARGS = dict(beam=6, lenpen=1.0, max_len_b=60, min_len=10, no_repeat_ngram_size=3)
CNN_KWARGS = dict(beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
LEAST_KWARGS = dict(beam=4, no_repeat_ngram_size=3)
LEAST_XSUM_KWARGS = dict(beam=6, no_repeat_ngram_size=3)
BEAM_ARGS={
    "xsum": XSUM_KWARGS,
    "cnn": CNN_KWARGS,
    "least": LEAST_KWARGS,
    "least_xsum": LEAST_XSUM_KWARGS,
}

@torch.no_grad()
def generate(bart, infile, desired_length_filepath, outfile="bart_hypo.txt", bsz=1, n_obs=None, **eval_kwargs):
    count = 1

    # if n_obs is not None: bsz = min(bsz, n_obs)

    with open(infile) as source, open(outfile, "w") as fout, open(desired_length_filepath) as desired_length_file:
        sline = source.readline().strip()
        slines = [sline]
        desired_length = desired_length_file.readline().strip()
        desired_lengths = [int(desired_length)]
        for sline, desired_length in zip(source, desired_length_file):
            if n_obs is not None and count > n_obs:
                break
            if count % bsz == 0:
                hypotheses_batch = bart.sample(slines, desired_lengths=desired_lengths, **eval_kwargs)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + "\n")
                    fout.flush()
                slines = []
                desired_lengths = []

            slines.append(sline.strip())
            desired_lengths.append(int(desired_length))
            count += 1

        if slines != []:
            hypotheses_batch = bart.sample(slines, desired_lengths=desired_lengths, **eval_kwargs)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + "\n")
                fout.flush()


def main():
    """
    Usage::

         python examples/bart/summarize.py \
            --model-dir $HOME/bart.large.cnn \
            --model-file model.pt \
            --src $HOME/data-bin/cnn_dm/test.source
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        default="bart.large.cnn/",
        help="path containing model file and src_dict.txt",
    )
    parser.add_argument(
        "--model-file",
        default="checkpoint_best.pt",
        help="where in model_dir are weights saved",
    )
    parser.add_argument(
        "--src", default="test.source", help="text to summarize", type=str
    )
    parser.add_argument(
        "--out", default="test.hypo", help="where to save summaries", type=str
    )
    parser.add_argument("--bsz", default=1, help="where to save summaries", type=int)
    parser.add_argument(
        "--n", default=None, help="how many examples to summarize", type=int
    )
    # parser.add_argument(
    #     "--xsum-kwargs",
    #     action="store_true",
    #     default=False,
    #     help="if true use XSUM_KWARGS else CNN_KWARGS",
    # )

    # Here are the additional arguments.
    parser.add_argument(
        "--use-proposal",
        action="store_true",
        default=False,
        help="if true use ProposedModel else BARTModel",
    )
    parser.add_argument(
        "--desired-length", default="test.oracle", help="desired lengths to summaries", type=str
    )
    parser.add_argument(
        "--beam-args",
        choices=[
            "xsum",
            "cnn",
            "least",
            "least_xsum",
        ],
        default="least",
        help="args for beam search.",
    )
    parser.add_argument(
        "--topk-eps",
        default=0.001,
        type=float,
        metavar="D",
        help="topk's epsilon"
    )
    parser.add_argument(
        "--topk-randperm",
        action="store_true",
        default=False,
        help="if true randomly permutate topk score",
    )

    args = parser.parse_args()
    eval_kwargs = BEAM_ARGS[args.beam_args]
    # eval_kwargs = XSUM_KWARGS if args.xsum_kwargs else CNN_KWARGS
    print(eval_kwargs)
    if args.model_dir == "pytorch/fairseq":
        model = torch.hub.load("pytorch/fairseq", args.model_file)
    if args.use_proposal:
        model = ProposedModel.from_pretrained(
            args.model_dir,
            checkpoint_file=args.model_file,
            data_name_or_path=args.model_dir,
        )
        model.model.encoder.extractor.topk_eps = args.topk_eps
        print("model's topk_eps:", model.model.encoder.extractor.topk_eps)
        model.model.encoder.extractor.topk_randperm = args.topk_randperm
        print("model's topk_randperm:", model.model.encoder.extractor.topk_randperm)
        # foo = torch.hub.load("pytorch/fairseq", "transformer.wmt16.en-de", checkpoint_file="model.pt",  tokenizer="moses", bpe="subword_nmt")
        # model.task.build_dataset_for_inference = foo.task.build_dataset_for_inference
    else:
        model = BARTModel.from_pretrained(
            args.model_dir,
            checkpoint_file=args.model_file,
            data_name_or_path=args.model_dir,
        )
        foo = torch.hub.load("pytorch/fairseq", "transformer.wmt16.en-de", checkpoint_file="model.pt",  tokenizer="moses", bpe="subword_nmt")
        model.task.build_dataset_for_inference = foo.task.build_dataset_for_inference

    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda().half()
    generate(
        model, args.src, args.desired_length, bsz=args.bsz, n_obs=args.n, outfile=args.out, **eval_kwargs
    )


if __name__ == "__main__":
    main()