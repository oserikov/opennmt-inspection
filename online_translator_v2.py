#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse

import sys

import torch

from onmt.utils.misc import get_logger
from onmt.translate.translator import build_translator

from tqdm import tqdm

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts
import json
import io

import os

class FakeOpt(object):
    def __init__(self,
            beam_size=5,
            min_length=10,
            max_length=100,
            stepwise_penalty=False,
            block_ngram_repeat=0,
            ignore_when_blocking=[],
            replace_unk=True,
            model=None,
            dump_layers=-1,
            verbose=False,
            report_bleu=False,
            batch_size=30,
            n_best=1,
            dump_beam='',
            gpu=-1,
            alpha=0,
            beta=0,
            length_penalty='none',
            coverage_penalty='none',
            data_type='text'):
        self.alpha = alpha
        self.beta = beta
        self.length_penalty = length_penalty
        self.coverage_penalty = coverage_penalty
        self.beam_size = beam_size
        self.n_best = n_best
        self.max_length = max_length
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self.dump_beam = dump_beam
        self.dump_layers = dump_layers
        self.report_bleu = report_bleu
        self.data_type = data_type
        self.replace_unk = replace_unk
        self.batch_size = batch_size
        self.gpu = gpu
        self.verbose = verbose
        self.model = model

# Translator cache
# TODO use of global variables is generally unhealthy and we should
# move this into a class.
translator = None
translator_model_name = None

def init_model(model, use_gpu=False):
    opt = FakeOpt(model=model, gpu=0 if use_gpu else -1)

    translator = build_translator(opt, report_score=False, logger=get_logger(), use_output=False)
    return translator

def translate(translator, sentences, modifications):
    # def intervene(layer_data, sentence_index, index):
    #     for token, layer, neuron, value in modifications[sentence_index]:
    #         if layer == index:
    #             print('Succesfully flipping %d %d %d %f' % (token, layer, neuron, value))
    #             layer_data[token][0][neuron] = value
    #     return layer_data

    def intervene(layer_data, layer_index):
        rnn_size = layer_data.shape[2]
        start_range = layer_index * rnn_size
        end_range = start_range + rnn_size

        for mod in modifications:
            neuron, activation = mod
            if neuron >= start_range and neuron < end_range:
                layer_data[:,:,neuron-start_range] = activation

        return layer_data

    modified = []
    dumps = []

    # NB. Some of this is kind of hacky with passing streams and things
    # and also returning them; it may be good to go back later and try to dedupe
    # all the plumbing. Everything should presently work though.
    for i, source in enumerate(sentences):
        stream = io.StringIO()
        layer_dump, scores, predictions = translator.translate(src_data_iter=[source],
                             src_dir='',
                             batch_size=1,
                             attn_debug=False,
                             intervention=lambda l, j: intervene(l, j),
                             out_file=stream)
        translation = stream.getvalue()

        sys.stdout.flush()
        modified.append(translation)
        dumps.append(layer_dump)

    return modified, dumps
