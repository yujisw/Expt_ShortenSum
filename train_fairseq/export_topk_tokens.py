import argparse
import torch
from bart_with_extractor import ProposedModel
from mytask import ProposalTask

@torch.no_grad()
def get_topk_token_set(model, source, k, use_proposal=False):
    tokens = model.encode(source)
    k = min(k, len(tokens))
    desired_length = torch.LongTensor([k])
    if torch.cuda.is_available():
        tokens = tokens.cuda()
        desired_length = desired_length.cuda()
    if use_proposal:
        encoder_out, (features, masked_x, x) = model.extract_topk_result(tokens, desired_length)
    else:
        features = model.extract_features(tokens)
    token_score = features.sum(axis=-1).data[0]
    threshold = sorted(features.sum(dim=-1).ravel())[-k].data
    topk_tokens = [token.item() for token, score in zip(tokens, token_score) if score >= threshold]

    tokens_str = [model.bpe.decode(
        model.task.source_dictionary.string(torch.tensor([t]))) for t in tokens]
    tokens_str[0] = "[BOS]"
    tokens_str[-1] = "[EOS]"
    tokens_str = [token.replace("\n", "[NewLine]") for token in tokens_str]
    tokens_str_md = [" **{}**".format(token.replace(" ", "")) if score >= threshold else token for token, score in zip(tokens_str, token_score)]

    return set(topk_tokens), "".join(tokens_str_md)

def calc_faithful_score(topk_token_set, gen_token_set, k):
    return len(topk_token_set & gen_token_set) / k

@torch.no_grad()
def calc_overall_faithful_score(model, src, gen, desired_length_file, bolded_out, score_out, use_proposal=False):
    count = 0
    scores = []
    with open(src) as src, open(gen) as gen, open(desired_length_file) as dl, open(bolded_out, "w") as bolded_out, open(score_out, "w") as score_out:
        for src_line, gen_line, desired_length in zip(src, gen, dl):
            desired_length = int(desired_length)
            topk_token_set, bolded_src = get_topk_token_set(model, src_line, desired_length)
            bolded_out.write(bolded_src + "\n")
            gen_token_set = set([token.item() for token in model.encode(gen_line)])
            score = calc_faithful_score(topk_token_set, gen_token_set, desired_length)
            score_out.write("{}: {}\n".format(count, score))
            scores.append(score)
            count += 1
        score_out.write("overall score: {}\n".format(sum(scores)/count))
    return sum(scores)/count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        default="bart.large.cnn/",
        help="path containing model file and src_dict.txt",
    )
    parser.add_argument(
        "--use-proposal",
        action="store_true",
        default=False,
        help="if true use ProposedModel else BARTModel",
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
        "--gen", default="test.hypo", help="generated summaries", type=str
    )
    parser.add_argument(
        "--desired-length", default="test.oracle", help="desired lengths to summaries", type=str
    )
    parser.add_argument(
        "--bolded-out", default="test.bolded_src", help="where to save bolded src to emphasize topk tokens", type=str
    )
    parser.add_argument(
        "--score-out", default="test.faithful_score", help="where to save faithful scores", type=str
    )
    parser.add_argument(
        "--topk-eps",
        default=0.001,
        type=float,
        metavar="D",
        help="topk's epsilon"
    )

    args = parser.parse_args()
    if args.use_proposal:
        model = ProposedModel.from_pretrained(
            args.model_dir,
            checkpoint_file=args.model_file,
            data_name_or_path=args.model_dir,
        )
        model.model.encoder.extractor.topk_eps = args.topk_eps
        print("model's topk_eps:", model.model.encoder.extractor.topk_eps)
        # foo = torch.hub.load("pytorch/fairseq", "transformer.wmt16.en-de", checkpoint_file="model.pt",  tokenizer="moses", bpe="subword_nmt")
        # model.task.build_dataset_for_inference = foo.task.build_dataset_for_inference
        model.model.encoder.use_topk_result = True
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
    
    faithful_score = calc_overall_faithful_score(
        model,
        args.src,
        args.gen,
        args.desired_length,
        args.bolded_out,
        args.score_out
    )
    print(faithful_score)

if __name__ == "__main__":
    main()
