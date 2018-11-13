#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse

import torch

from onmt.utils.misc import get_logger
from onmt.translate.translator import build_translator

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts

import os


def main(opt):
    translator = build_translator(opt, report_score=True, logger=logger, use_output=True)

    if len(opt.neurons.strip()) == 0:
        neurons_to_ablate = []
    else:
        neurons_to_ablate = [int(x) for x in opt.neurons.split(" ")]
    print(neurons_to_ablate)

    def intervene(layer_data, layer_index):
        rnn_size = layer_data.shape[2]
        start_range = layer_index * rnn_size
        end_range = start_range + rnn_size
        neurons = [n-start_range for n in neurons_to_ablate if n >= start_range and n < end_range]

        layer_data[:,:,neurons] = 0

        return layer_data

    translator.translate(src_path=opt.src,
                         tgt_path=opt.tgt,
                         src_dir=opt.src_dir,
                         batch_size=opt.batch_size,
                         attn_debug=opt.attn_debug,
                         intervention=lambda l, i: intervene(l, i),
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ablate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    parser.add_argument('-neurons-to-ablate', dest='neurons', type=str, default="")

    opt = parser.parse_args()
    logger = get_logger(opt.log_file)
    main(opt)
