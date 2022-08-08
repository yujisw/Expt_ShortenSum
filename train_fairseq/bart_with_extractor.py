# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension
"""

import copy
import logging
from typing import List, Dict
from numpy import extract, inner

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params

# from fairseq.models.bart.hub_interface import BARTHubInterface

from typing import Optional
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor

from fairseq.models.fairseq_encoder import EncoderOut

from differentiable_topk import SortedTopK_custom
from my_hub_interface import MyHubInterface
from soft_topk_attention import SoftTopKMultiHeadAttention

logger = logging.getLogger(__name__)


@register_model("proposed_model")
class ProposedModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        logger.info("PrintDebug: ProposedModel __init__ called.")

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

        logger.info("encoder dictionary eos: {}".format(self.encoder.dictionary.eos()))

        extractor = Extractor(args, self.encoder.dictionary.eos())
        self.encoder = Encoder_Extractor(args, encoder, extractor)

    @staticmethod
    def add_args(parser):
        super(ProposedModel, ProposedModel).add_args(parser)
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            help="Apply spectral normalization on the classification head",
        )

        # Here are the arguments for the Extractor.
        parser.add_argument(
            "--extract-num",
            help="The number of extracted tokens",
            type=int
        )
        parser.add_argument(
            "--use-differentiable-topk",
            action="store_true",
            help="Use differentiable top-k operator to mask unimportant tokens."
        )
        parser.add_argument(
            "--apply-formula-to-extract-num",
            action="store_true",
            help="Apply the formula to extract_num. (extract_num = alpha * target_length + beta)"
        )
        parser.add_argument(
            "--alpha-for-extract-num",
            help="Alpha for the formula of extract_num.",
            type=float
        )
        parser.add_argument(
            "--beta-for-extract-num",
            help="Beta for the formula of extract_num.",
            type=int
        )
        parser.add_argument(
            "--when-to-extract",
            choices=[
                "before_attention",
                "during_attention",
                "after_attention",
                "after_fc",
            ],
            default="before_attention",
            help="When the model extracts.",
        )


        # Here are the arguments for the extracting method.
        parser.add_argument(
            "--token-scoring-fn",
            choices=[
                "inner_product",
                "linear",
                "self_attention",
            ],
            default="self_attention",
            help="Token scoring function to use for the Extractor.",
        )

    @property
    def supported_targets(self):
        return {"self"}

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        tgt_lengths=None, # [bsz] (the same shape as src_lengths)
        features_only=False,
        classification_head_name=None,
        token_embeddings=None,
        **kwargs,
    ):
        # if classification_head_name is not None:
        #     features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            token_embeddings=token_embeddings,
            **kwargs,
        )

        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            **kwargs,
        )

        # if classification_head_name is not None:
        #     sentence_representation = x[
        #         src_tokens.eq(self.encoder.dictionary.eos()), :
        #     ].view(x.size(0), -1, x.size(-1))[:, -1, :]
        #     x = self.classification_heads[classification_head_name](
        #         sentence_representation
        #     )
        return x, extra

    @classmethod
    def from_pretrained( # この関数は訓練時には呼ばれず、generate時に呼ばれる。
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        **kwargs,
    ):
        from fairseq import hub_utils

        # Load args, task, and models from trained checkpoint
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return MyHubInterface(x["args"], x["task"], x["models"][0])

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = BARTClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            do_spectral_norm=self.args.spectral_norm_classification_head,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + "." if name != "" else ""
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict["encoder.encoder.embed_tokens.weight"].size(0)
        if (
            loaded_dict_size == len(self.encoder.dictionary) + 1
            and "<mask>" not in self.encoder.dictionary
        ):
            truncate_emb("encoder.encoder.embed_tokens.weight")
            truncate_emb("decoder.embed_tokens.weight")
            truncate_emb("encoder.encoder.output_projection.weight")
            truncate_emb("decoder.output_projection.weight")

        # When continued pretraining on new set of languages for mbart,
        # add extra lang embeddings at the end of embed_tokens.
        # Note: newly added languages are assumed to have been added at the end.
        # ここは multilingual_denoisingやらないから呼ばれない。ややこしいのでコメントアウト。
        # if self.args.task == "multilingual_denoising" and loaded_dict_size < len(
        #     self.encoder.dictionary
        # ):
        #     logger.info(
        #         "Adding extra language embeddings not found in pretrained model for "
        #         "continued pretraining of MBART on new set of languages."
        #     )
        #     loaded_mask_token_embedding = state_dict["encoder.embed_tokens.weight"][
        #         -1, :
        #     ]

        #     num_langids_to_add = len(self.encoder.dictionary) - loaded_dict_size
        #     embed_dim = state_dict["encoder.embed_tokens.weight"].size(1)

        #     new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
        #     nn.init.normal_(new_lang_embed_to_add, mean=0, std=embed_dim ** -0.5)
        #     new_lang_embed_to_add = new_lang_embed_to_add.to(
        #         dtype=state_dict["encoder.embed_tokens.weight"].dtype,
        #     )

        #     state_dict["encoder.embed_tokens.weight"] = torch.cat(
        #         [
        #             state_dict["encoder.embed_tokens.weight"][
        #                 : loaded_dict_size - 1, :
        #             ],
        #             new_lang_embed_to_add,
        #             loaded_mask_token_embedding.unsqueeze(0),
        #         ]
        #     )
        #     state_dict["decoder.embed_tokens.weight"] = torch.cat(
        #         [
        #             state_dict["decoder.embed_tokens.weight"][
        #                 : loaded_dict_size - 1, :
        #             ],
        #             new_lang_embed_to_add,
        #             loaded_mask_token_embedding.unsqueeze(0),
        #         ]
        #     )

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting", prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v

class Encoder_Extractor(nn.Module):
    def __init__(self, args, encoder, extractor):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.extractor = extractor
        self.dictionary = encoder.dictionary
        self.reorder_encoder_out = encoder.reorder_encoder_out

    def max_positions(self):
        """Return the max input length allowed by the task."""
        return None

    def forward_torchscript(self, net_input: Dict[str, Tensor]):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        if torch.jit.is_scripting():
            return self.forward(
                src_tokens=net_input["src_tokens"],
                src_lengths=net_input["src_lengths"],
                tgt_lengths=net_input["tgt_lengths"],
            )
        else:
            return self.forward_non_torchscript(net_input)

    @torch.jit.unused
    def forward_non_torchscript(self, net_input: Dict[str, Tensor]):
        encoder_input = {
            k: v for k, v in net_input.items() if k != "prev_output_tokens"
        }
        return self.forward(**encoder_input)

    def forward(
        self,
        src_tokens,
        src_lengths,
        tgt_lengths=None, # [bsz] (the same shape as src_lengths)
        token_embeddings=None,
        **kwargs,
    ):
        # if classification_head_name is not None:
        #     features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            token_embeddings=token_embeddings,
            **kwargs,
        )

        # ここに提案手法を組み込む
        extractor_out, new_padding_mask = self.extractor(
            encoder_out.encoder_out,
            src_tokens,
            encoder_out.encoder_padding_mask,
            tgt_lengths,
        )
        extractor_out = EncoderOut(
            encoder_out=extractor_out,  # T x B x C
            encoder_padding_mask=new_padding_mask,  # B x T
            encoder_embedding=encoder_out.encoder_embedding,  # B x T x C
            encoder_states=encoder_out.encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )
        return extractor_out

class Extractor(nn.Module):
    """Encoder layer block.

    In the original paper each opesration (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, eos_idx):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        logger.info("normalize_before")
        logger.info(self.normalize_before)
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

        # The additional arguments below
        self.eos_idx = eos_idx
        logger.info("eos_idx: {}".format(self.eos_idx))

        # settings for the method of extracting
        self.when_to_extract = getattr(args, "when_to_extract", "")
        assert self.when_to_extract != ""
        logger.info("Extract {}".format(self.when_to_extract))
        if self.when_to_extract == "during_attention":
            pass
        else:
            self.extract = self.extract_using_normal_topk
            self.use_differentiable_topk = getattr(args, "use_differentiable_topk", False)
            if self.use_differentiable_topk:
                logger.info("Use differentiable top-k operator.")
                self.topk_ope = SortedTopK_custom(epsilon=0.001, max_iter=200)
                self.extract = self.extract_using_differentiable_topk
            else:
                logger.info("Use normal top-k operator (not differentiable).")
                self.extract = self.extract_using_normal_topk

            # settings for extract_num
            self.apply_formula_to_extract_num = getattr(args, "apply_formula_to_extract_num", False)
            if self.apply_formula_to_extract_num:
                self.alpha_for_extract_num = getattr(args, "alpha_for_extract_num", -1)
                self.beta_for_extract_num = getattr(args, "beta_for_extract_num", -1)
                assert self.alpha_for_extract_num!=-1 and self.beta_for_extract_num!=-1, "Specify ALPHA or BETA when applying a formula to extract_num."
                logger.info("extract_num are determined by the following formula.")
                logger.info("extract_num = {} * target_length + {}".format(self.alpha_for_extract_num, self.beta_for_extract_num))
            else:
                self.extract_num = getattr(args, "extract_num", 0)
                assert self.extract_num!=0, "Specify extract_num when using a fixed value for it. It should be a positive integer."
                logger.info("For different target lengths, extract_num is fixed at {}.".format(self.extract_num))

            # setting for token scoring
            self.token_scoring_fn = getattr(args, "token_scoring_fn", "")
            assert self.token_scoring_fn != ""
            logger.info("Calculate token's scores with {}".format(self.token_scoring_fn))
            if self.token_scoring_fn=="linear":
                self.linear_for_token_scores = quant_noise(
                    nn.Linear(self.embed_dim, 1), p=self.quant_noise, block_size=self.quant_noise_block_size
                )
                self.activation_for_token_scores = nn.Softmax(dim=0)
            elif self.token_scoring_fn=="self_attention":
                self.self_attn_for_token_scores = MultiheadAttention(
                    self.embed_dim,
                    args.encoder_attention_heads,
                    dropout=args.attention_dropout,
                    self_attention=True,
                    q_noise=self.quant_noise,
                    qn_block_size=self.quant_noise_block_size,
                )
                self.dropout_module_for_token_scores = FairseqDropout(
                    args.dropout, module_name=self.__class__.__name__
                )
                self.self_attn_layer_norm_for_token_scores = LayerNorm(self.embed_dim)
                self.linear_for_token_scores = quant_noise(
                    nn.Linear(self.embed_dim, 1), p=self.quant_noise, block_size=self.quant_noise_block_size
                )
                self.activation_for_token_scores = nn.Softmax(dim=0)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        when_to_extract = getattr(args, "when_to_extract", "")
        if when_to_extract == "during_attention":
            return SoftTopKMultiHeadAttention(
                args,
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )
        else:
            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def get_token_scores(self, x, src_tokens, padding_mask, attn_mask):
        """
        Calculating token's scores using inner product.
        """
        if self.token_scoring_fn=="inner_product":
            # 文表現を得る
            sentence_representation = x[
                    src_tokens.permute(1,0).eq(self.eos_idx), :
                ].view(-1, x.size(1), x.size(-1))[-1, :, :] # [1, batch_size, embed_dim]
            # calc similarity between each token vec and sentence representation by inner product
            # we replace vecs of padding token into "-inf" so that their topk_result becomes 0.5, which means padding tokens are not chosen.
            inner_products = torch.sum(sentence_representation * x, 2) # [seq_len, batch_size]
            return inner_products
        elif self.token_scoring_fn=="linear":
            x = self.linear_for_token_scores(x).squeeze(-1)
            x = self.activation_for_token_scores(x)
            return x
        elif self.token_scoring_fn=="self_attention":
            residual = x
            x, _ = self.self_attn_for_token_scores(
                query=x,
                key=x,
                value=x,
                key_padding_mask=padding_mask,
                attn_mask=attn_mask,
            ) # [seq_len, bsz, embed_dim] -> [seq_len, bsz, embed_dim]

            x = self.dropout_module_for_token_scores(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.self_attn_layer_norm_for_token_scores(x)
            residual = x
            # argsみたらFalseだったからこれはコメントアウト。
            # if self.normalize_before:
            #     x = self.final_layer_norm(x)

            x = self.linear_for_token_scores(x).squeeze(-1) # [seq_len, bsz]
            x = self.activation_for_token_scores(x) # [seq_len, bsz]
            return x

    def extract_using_normal_topk(self, x, src_tokens, padding_mask, attn_mask, tgt_lengths):
        """
        Input:
            x: [seq_len, batch_size, embed_dim]
            src_tokens: [batch_size, seq_len]
            padding_mask: [batch_size, seq_len]
        Output:
            extracted_x: [extract_num, batch_size, embed_dim]
            new_padding_mask: [batch_size, extract_num]
        """
        if self.apply_formula_to_extract_num:
            extract_num = int(tgt_lengths.max().item() * self.alpha_for_extract_num + self.beta_for_extract_num)
        else:
            extract_num = self.extract_num
        # get token's scores
        token_scores = self.get_token_scores(x, src_tokens, padding_mask, attn_mask).masked_fill(padding_mask.permute(1,0), float("-inf")) # [seq_len, batch_size]
        # get indices of top-k high similarity
        topk_high_indices = torch.topk(token_scores, min(extract_num, x.size(0)), dim=0).indices
        topk_high_indices = torch.sort(topk_high_indices, dim=0).values # restore the original order
        # make transform matrix for extracting important token vecs
        binary_extract_matrix = torch.nn.functional.one_hot(topk_high_indices, num_classes=x.shape[0]).permute(1,2,0).to(x.dtype)
        # calculate extracted token vecs & new padding mask
        extracted_x = torch.bmm(x.permute(1, 2, 0), binary_extract_matrix).permute(2,0,1) # x:[B, C, L], bem:[B, L, extract_num] = [B, C, extract_num]
        new_padding_mask = torch.bmm(padding_mask.unsqueeze(1).to(x.dtype), binary_extract_matrix).to(torch.bool).squeeze(1)
        return extracted_x, new_padding_mask, topk_high_indices

    def extract_using_differentiable_topk(self, x, src_tokens, padding_mask, attn_mask, tgt_lengths):
        """
        Input:
            x: [seq_len, batch_size, embed_dim]
            src_tokens: [batch_size, seq_len]
            padding_mask: [batch_size, seq_len]
        Output:
            extracted_x: [extract_num, batch_size, embed_dim]
            new_padding_mask: [batch_size, extract_num]
        """
        if self.apply_formula_to_extract_num:
            extract_num = int(tgt_lengths.max().item() * self.alpha_for_extract_num + self.beta_for_extract_num)
        else:
            extract_num = self.extract_num
        # logger.info("extract_num for this minibatch is {}. (minibatch size is {})".format(extract_num, x.size(1)))
        # get token's scores
        token_scores = self.get_token_scores(x, src_tokens, padding_mask, attn_mask) # [seq_len, batch_size]
        # get indices of top-k high similarity
        topk_result, _ = self.topk_ope(token_scores.permute(1,0), k=min(extract_num, x.size(0)))
        # calculate extracted token vecs (Note: topk_result of padding tokens are replaced into 0.)
        masked_x = (x.permute(2,1,0) * (topk_result.sum(axis=-1).masked_fill(padding_mask, 0.))).permute(2,1,0).to(x.dtype) # x:[B, C, L], bem:[B, L, extract_num] = [B, C, extract_num]
        return masked_x, padding_mask, topk_result

    def forward(self, x, src_tokens, encoder_padding_mask, tgt_lengths, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.
            tgt_lengths (LongTensor): long tensor of shape `(batch).`

        Returns:
            encoded output of shape `(extract_num, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        if self.when_to_extract == "before_attention":
            x, new_padding_mask, topk_result = self.extract(x, src_tokens, encoder_padding_mask, attn_mask, tgt_lengths)
        residual = x

        # argsみたらFalseだったからこれはコメントアウト。
        # if self.normalize_before:
        #     x = self.self_attn_layer_norm(x)
        
        if self.when_to_extract == "during_attention":
            x, topk_result = self.self_attn( # This topk_result is attn_weights
                tgt_lengths=tgt_lengths,
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            new_padding_mask = encoder_padding_mask
        else:
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.when_to_extract == "after_attention":
            x, new_padding_mask, topk_result = self.extract(x, src_tokens, encoder_padding_mask, attn_mask, tgt_lengths)
        residual = x
        # argsみたらFalseだったからこれはコメントアウト。
        # if self.normalize_before:
        #     x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.when_to_extract == "after_fc":
            x, new_padding_mask, topk_result = self.extract(x, src_tokens, encoder_padding_mask, attn_mask, tgt_lengths)

        return x, new_padding_mask


class BARTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

        if do_spectral_norm:
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture("proposed_model", "proposed_model_large")
def proposed_model_large_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 11)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)


@register_model_architecture("proposed_model", "proposed_model_base")
def proposed_model_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    proposed_model_large_architecture(args)

