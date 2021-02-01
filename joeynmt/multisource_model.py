# coding: utf-8
"""
Module to represents whole models
"""

import numpy as np

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from joeynmt.initialization import initialize_model
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from joeynmt.multisource_decoders import Decoder, RecurrentDecoder, TransformerDecoder
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.search import beam_search, greedy, multisource_greedy, multisource_beam_search
from joeynmt.vocabulary import Vocabulary
from joeynmt.multisource_batch import Batch
from joeynmt.helpers import ConfigurationError


class Model(nn.Module):
    """
    Base Model class
    """

    def __init__(self,
                 encoder_tcp: Encoder,
                 encoder_tsl: Encoder,
                 decoder: Decoder,
                 src_embed_tcp: Embeddings,
                 # trg_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary) -> None:  # Use same vocabulary for translation and gloss
        """
        Create a new encoder-decoder model

        :param encoder_tcp: encoder of transcription
        :param encoder_tsl: encoder of translation
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        self.src_embed_tcp = src_embed_tcp
        self.trg_embed = trg_embed
        self.trg_embed = trg_embed
        self.encoder_tcp = encoder_tcp
        self.encoder_tsl = encoder_tsl
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]

    # pylint: disable=arguments-differ
    def forward(self, src_tcp: Tensor,src_tsl: Tensor, trg_input: Tensor, src_tcp_mask: Tensor, src_tsl_mask: Tensor, 
                src_lengths_tcp: Tensor, src_lengths_tsl: Tensor, trg_mask: Tensor = None) -> (
        Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_lengths: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """
        # src lengths should be the longer on of tsl and tcp
        encoder_tcp_output, encoder_tcp_hidden = self.encode_tcp(src=src_tcp,
                                                     src_length=src_lengths_tcp,
                                                     src_mask=src_tcp_mask)
        encoder_tsl_output, encoder_tsl_hidden = self.encode_tsl(src=src_tsl,
                                                     src_length=src_lengths_tsl,
                                                     src_mask=src_tsl_mask)
        unroll_steps = trg_input.size(1)
        # -------------------return two output and two hidden------------------------

        return self.decode(encoder_tcp_output=encoder_tcp_output,
                           encoder_tsl_output=encoder_tsl_output,
                           encoder_tcp_hidden=encoder_tcp_hidden,
                           encoder_tsl_hidden=encoder_tsl_hidden,
                           src_tcp_mask=src_tcp_mask, src_tsl_mask=src_tsl_mask, trg_input=trg_input,
                           unroll_steps=unroll_steps,
                           trg_mask=trg_mask)

    def encode_tcp(self, src: Tensor, src_length: Tensor, src_mask: Tensor) \
        -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder_tcp(self.src_embed_tcp(src), src_length, src_mask)

    def encode_tsl(self, src: Tensor, src_length: Tensor, src_mask: Tensor) \
        -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder_tsl(self.trg_embed(src), src_length, src_mask)

    def decode(self, encoder_tcp_output: Tensor, encoder_tsl_output: Tensor, encoder_tcp_hidden: Tensor, encoder_tsl_hidden: Tensor,
               src_tcp_mask: Tensor, src_tsl_mask: Tensor, trg_input: Tensor,
               unroll_steps: int, decoder_hidden: Tensor = None,
               trg_mask: Tensor = None) \
        -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(trg_embed=self.trg_embed(trg_input),
                            encoder_tcp_output=encoder_tcp_output,
                            encoder_tsl_output=encoder_tsl_output,
                            encoder_tcp_hidden=encoder_tcp_hidden,
                            encoder_tsl_hidden=encoder_tsl_hidden,
                            src_tcp_mask=src_tcp_mask,
                            src_tsl_mask=src_tsl_mask,
                            unroll_steps=unroll_steps,
                            hidden=decoder_hidden,
                            trg_mask=trg_mask)

    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module) \
            -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        # pylint: disable=unused-variable

        out, hidden, att_probs, _ = self.forward(
            src_tcp=batch.src_tcp, src_tsl=batch.src_tsl, trg_input=batch.trg_input,
            src_tcp_mask=batch.src_tcp_mask, src_tsl_mask=batch.src_tsl_mask, 
            src_lengths_tcp=batch.src_lengths_tcp, src_lengths_tsl=batch.src_lengths_tsl, 
            trg_mask=batch.trg_mask)

        # compute log probs
        log_probs = F.log_softmax(out, dim=-1)

        # compute batch loss
        batch_loss = loss_function(log_probs, batch.trg)
        # return batch loss = sum over all elements in batch that are not pad
        return batch_loss

    def run_batch(self, batch: Batch, max_output_length: int, beam_size: int,
                  beam_alpha: float) -> (np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        # print('beam_size')
        # print(beam_size)
        encoder_tcp_output, encoder_tcp_hidden = self.encode_tcp(
            batch.src_tcp, batch.src_lengths_tcp,
            batch.src_tcp_mask)
        encoder_tsl_output, encoder_tsl_hidden = self.encode_tsl(
            batch.src_tsl, batch.src_lengths_tsl,
            batch.src_tsl_mask)

        # if maximum output length is not globally specified, adapt to src len
        # output_length should equal to src_lengths for each example
        if max_output_length is None:
            max_output_length = int(max(batch.src_lengths_tcp.cpu().numpy()) * 1.0)

        # greedy decoding
        if beam_size < 2:
            stacked_output, stacked_attention_scores = multisource_greedy(
                    encoder_tcp_output=encoder_tcp_output,
                    encoder_tsl_output=encoder_tsl_output,
                    encoder_tcp_hidden=encoder_tcp_hidden,
                    encoder_tsl_hidden=encoder_tsl_hidden, eos_index=self.eos_index,
                    src_tcp_mask=batch.src_tcp_mask, src_tsl_mask=batch.src_tsl_mask, embed=self.trg_embed,
                    bos_index=self.bos_index, decoder=self.decoder,
                    max_output_length=max_output_length)
            # batch, time, max_src_length
        else:  # beam size
            stacked_output, stacked_attention_scores = \
                    multisource_beam_search(
                        size=beam_size, encoder_tcp_output=encoder_tcp_output,
                        encoder_tsl_output=encoder_tsl_output,
                        encoder_tcp_hidden=encoder_tcp_hidden,
                        encoder_tsl_hidden=encoder_tsl_hidden,
                        src_tcp_mask=batch.src_tcp_mask, src_tsl_mask=batch.src_tsl_mask, embed=self.trg_embed,
                        max_output_length=max_output_length,
                        alpha=beam_alpha, eos_index=self.eos_index,
                        pad_index=self.pad_index,
                        bos_index=self.bos_index,
                        decoder=self.decoder)

        return stacked_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return "%s(\n" \
               "\tencoder_tcp=%s,\n" \
               "\tencoder_tsl=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed_tcp=%s,\n" \
               "\ttrg_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder_tcp, self.encoder_tsl, 
                   self.decoder, self.src_embed_tcp, self.trg_embed, self.trg_embed)


def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """
    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = trg_vocab.stoi[PAD_TOKEN]

#--------How to define input of embeddings?--------------
    print(cfg)
    src_embed_tcp = Embeddings(
        **cfg["encoder_tcp"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)
    trg_embed = Embeddings(
        **cfg["encoder_tsl"]["embeddings"], vocab_size=len(trg_vocab),
        padding_idx=trg_padding_idx)

    # this ties source and target embeddings
    # for softmax layer tying, see further below
    if cfg.get("tied_embeddings", False):
        if src_vocab.itos == trg_vocab.itos:
            # share embeddings for src and trg
            trg_embed = src_embed
        else:
            raise ConfigurationError(
                "Embedding cannot be tied since vocabularies differ.")
    else:
        trg_embed = Embeddings(
            **cfg["decoder"]["embeddings"], vocab_size=len(trg_vocab),
            padding_idx=trg_padding_idx)

    # build encoder
    enc_tcp_dropout = cfg["encoder_tcp"].get("dropout", 0.)
    enc_tsl_dropout = cfg["encoder_tsl"].get("dropout", 0.)
    enc_emb_tcp_dropout = cfg["encoder_tcp"]["embeddings"].get("dropout", enc_tcp_dropout)
    enc_emb_tsl_dropout = cfg["encoder_tsl"]["embeddings"].get("dropout", enc_tsl_dropout)
    if cfg["encoder_tcp"].get("type", "recurrent") == "transformer":
        assert cfg["encoder_tcp"]["embeddings"]["embedding_dim"] == \
               cfg["encoder_tcp"]["hidden_size"], \
               "for transformer, emb_size must be hidden_size"

        encoder_tcp = TransformerEncoder(**cfg["encoder_tcp"],
                                     emb_size=src_embed_tcp.embedding_dim,
                                     emb_dropout=enc_emb_tcp_dropout)
    else:
        encoder_tcp = RecurrentEncoder(**cfg["encoder_tcp"],
                                   emb_size=src_embed_tcp.embedding_dim,
                                   emb_dropout=enc_emb_tcp_dropout)

    if cfg["encoder_tsl"].get("type", "recurrent") == "transformer":
        assert cfg["encoder_tsl"]["embeddings"]["embedding_dim"] == \
               cfg["encoder_tsl"]["hidden_size"], \
               "for transformer, emb_size must be hidden_size"

        encoder_tsl = TransformerEncoder(**cfg["encoder_tsl"],
                                     emb_size=trg_embed.embedding_dim,
                                     emb_dropout=enc_emb_tsl_dropout)
    else:
        encoder_tsl = RecurrentEncoder(**cfg["encoder_tsl"],
                                   emb_size=trg_embed.embedding_dim,
                                   emb_dropout=enc_emb_tsl_dropout)


    # build decoder
    dec_dropout = cfg["decoder"].get("dropout", 0.)
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    #----------------how to convey sum of the two encoder output to decoder?-------
    if cfg["decoder"].get("type", "recurrent") == "transformer":
        decoder = TransformerDecoder(
            **cfg["decoder"], encoder_tcp=encoder_tcp,
                    encoder_tsl=encoder_tsl,  vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
    else:
        decoder = RecurrentDecoder(
            **cfg["decoder"], encoder_tcp_output=encoder_tcp_output,
                    encoder_tsl_output=encoder_tsl_output,
                    encoder_tcp_hidden=encoder_tcp_hidden,
                    encoder_tsl_hidden=encoder_tsl_hidden,  vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)

    model = Model(encoder_tcp=encoder_tcp, encoder_tsl=encoder_tsl, decoder=decoder,
                  src_embed_tcp=src_embed_tcp, trg_embed=trg_embed,
                  src_vocab=src_vocab, trg_vocab=trg_vocab)

    # tie softmax layer with trg embeddings
    if cfg.get("tied_softmax", False):
        if trg_embed.lut.weight.shape == \
                model.decoder.output_layer.weight.shape:
            # (also) share trg embeddings and softmax layer:
            model.decoder.output_layer.weight = trg_embed.lut.weight
        else:
            raise ConfigurationError(
                "For tied_softmax, the decoder embedding_dim and decoder "
                "hidden_size must be the same."
                "The decoder must be a Transformer.")

    # custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model
