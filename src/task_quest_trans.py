"""
Transforms a question to an asserted prop.
Reads loc2props.csv and prepares train/test data.
"""
from helper import *
from models import *
from csv_utils import *
from time import strftime, localtime, time
from random import shuffle
import re
import json
from nltk.tokenize import word_tokenize
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import linecache
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("-rev", choices=["basic", "copy", "rule", "baseline"], 
                    required=True)
parser.add_argument("-n_epochs", type=int, default=10)
parser.add_argument("-rev_len", type=int, default=sys.maxsize)
parser.add_argument("-rev_enc_dim", type=int, default=128)
parser.add_argument("-beam_size", type=int, default=4)
parser.add_argument("-max_prop_len", type=int, default=200)
parser.add_argument("-copy_weight", default="auto") 
parser.add_argument("-copy_mech", default="attn_nm", \
                    choices=["bilinear", "ff", "attn_nm"]) 
parser.add_argument("-learn_rate", type=float, default=0.001)
parser.add_argument("-n_folds", type=int, default=5)
args = parser.parse_args()

data_path = {"train": "../data/quest_us2016.csv",
             "test": "../data/quest_mm2012.csv"}
glove_dir = "../rsc"

def get_prefix(args):
    prefix = "MM2012-{}".format(strftime("%Y%m%d_%H%M%S", localtime()))
    prefix += "-MTH{}".format(args.rev)

    if args.rev in ["basic", "copy"]:
        prefix += "-END{}-RML{}-BM{}".format(
                args.rev_enc_dim,
                "max" if args.rev_len > 99999 else args.rev_len, 
                args.beam_size)
        if args.rev.startswith("copy"):
            prefix += "-CW{}-CB{}".format(args.copy_weight, args.copy_mech)
        prefix += "-LR{}".format(args.learn_rate)

    return prefix

def get_tokens(text_nlp):
    tokens, poses, res_nps = [], [], []
    for sent in text_nlp["sentences"]:
        for token in sent["tokens"]:
            tokens.append(token["originalText"].lower())
            poses.append(token["pos"])

        np_level = wordno = 0
        nps = set([])
        npes = set([])
        for token in sent["parse"].split():
            if token.startswith("("):
                if token.startswith("(NP") or np_level > 0:
                    np_level += 1
            elif token.endswith(")"):
                if np_level > 0:
                    n_bracks = len(token.split(")")) - 1
                    np_level = max(0, np_level - n_bracks)
                    if np_level == 0:
                        npes.add(wordno)
                    else:
                        nps.add(wordno)
                wordno += 1

        for wordno, word in enumerate(sent["tokens"]):
            if wordno in nps: res_nps.append("NP")
            elif wordno in npes: res_nps.append("NPE")
            else: res_nps.append(-1)

    #print("text:", text)
    #print("tokens:", tokens)
    #print("pos:", poses)
    #print("np:", res_nps)
    #print("")

    return tokens, poses, res_nps

def get_glove_idxs(words):
    idxs = []
    for word in words:
        if word in word2idx:
            idx = word2idx[word]
        else:
            if word in word2idx_glove:
                v = np.array(linecache.getline(f"{glove_dir}/glove.840B.300d.txt",
                                               word2idx_glove[word]+1)\
                                      .split(" ")[1:], 
                             dtype=float)
            else:
                v = np.random.uniform(-.25, .25, 300)
            wembs_w.append(v)
            idx = len(word2idx)
            word2idx[word] = idx
        idxs.append(idx)
    return idxs

def accuracy(out_idxs_true, out_idxs_pred):
    words_ref = [str(idx) for idx in out_idxs_true]
    words_hyp = [str(idx) for idx in out_idxs_pred]

    bleu = get_bleu([words_ref], words_hyp)
    match = 1. if words_ref == words_hyp else 0.

    return bleu, match


prefix = get_prefix(args)
os.makedirs("../logs", exist_ok=True)
logger = get_logger(f"../logs/{prefix}.txt")

# Prepare data
word2idx = {}
word2idx_glove = {word.strip(): idx for idx, word in \
        enumerate(open(f"{glove_dir}/glove_words.txt"))}
wembs_w = []  # Will be cast to np.array later

data = defaultdict(list)
for split in ["train", "test"]:
    if args.rev in ["rule", "baseline"] and split == "train": continue

    print(f"Loading {split}...")
    for row in tqdm(iter_csv_header(data_path[split])):
        if "Question" not in row["ya"]: continue

        in_tokens, in_poses, in_nps = \
                get_tokens(json.loads(row["loctext_processed"]))
        out_tokens, _, _ = \
                get_tokens(json.loads(row["proptext_processed"]))

        inst = {"in_tokens": in_tokens,
                "in_poses": in_poses,
                "in_nps": in_nps,
                "out_tokens": out_tokens}
        if args.rev in ["basic", "copy"]:
            inst["in_idxs"] = get_glove_idxs(in_tokens)
            inst["out_idxs"] = get_glove_idxs(["[SOP]"] + out_tokens + ["[EOP]", "[EOG]"])
        data[split].append(inst)

    #data[split] = data[split][:100]
    print(f"n_{split}: {len(data[split])}")
print("voca:", len(word2idx))
wembs_w = float_tensor(np.array(wembs_w))


if args.rev in ["basic", "copy"]:
    revisor = Revisor2(args, wembs_w, None, word2idx)
    if torch.cuda.is_available():
        revisor.cuda(torch.device("cuda"))

    shuffle(data["train"])
    data_train = data["train"][int(len(data["train"])/5):]
    data_val = data["train"][:int(len(data["train"])/5)]
    for epoch in range(args.n_epochs):
        # Train
        train_losses = []
        for inst in tqdm(data_train):
            loss, _ = revisor.revise(inst["in_idxs"], inst["out_idxs"], "train")
            if not np.isnan(loss):
                train_losses.append(loss)

        # Validation
        val_bleus = []
        with torch.no_grad():
            for inst in tqdm(data_val):
                _, out_idxs_pred = revisor.revise(inst["in_idxs"], None, "test")
                bleu, match = accuracy(inst["out_idxs"], out_idxs_pred)
                val_bleus.append(bleu)

        # Test
        with torch.no_grad():
            test_bleus, test_matches = [], []
            for inst in tqdm(data["test"]):
                _, out_idxs_pred = revisor.revise(inst["in_idxs"], None, "test")

                bleu, match = accuracy(inst["out_idxs"], out_idxs_pred)
                test_bleus.append(bleu)
                test_matches.append(match)

        logger.info("[Epoch {}] train_loss={:.3f}, val_bleu={:.3f}, test_bleu={:.3f}, test_match={:.3f}".format(epoch+1, np.mean(train_losses), np.mean(val_bleus), np.mean(test_bleus), np.mean(test_matches)))

elif args.rev == "rule":
    revisor = QuestionTransformer()
    test_bleus, test_matches = [], []
    test_bleus_well, test_matches_well = [], []
    patt_matched, patt_matched_well = [], []
    for inst in data["test"]:
        out_text_pred, pattern_name = \
                revisor.transform(inst["in_tokens"], inst["in_poses"], 
                                  inst["in_nps"])
        out_tokens_pred = out_text_pred.strip().split(" ")

        bleu, match = accuracy(inst["out_tokens"], out_tokens_pred)
        test_bleus.append(bleu)
        test_matches.append(match)
        patt_matched.append(int(pattern_name is not None))

        if revisor.is_well_formed(inst["in_tokens"][0]):
            test_bleus_well.append(bleu)
            test_matches_well.append(match)
            patt_matched_well.append(int(pattern_name is not None))

    logger.info(("test_bleu={:.3f}, test_match={:.3f}, "
                 "matched={}/{} ({:.3f})").format(
                np.mean(test_bleus), np.mean(test_matches),
                sum(patt_matched), len(patt_matched), np.mean(patt_matched)))
    logger.info(("test_bleu_well={:.3f}, test_match_well={:.3f}, "
                 "matched_well={}/{} ({:.3f})").format(
                np.mean(test_bleus_well), np.mean(test_matches_well),
                sum(patt_matched_well), len(patt_matched_well),
                np.mean(patt_matched_well)))

elif args.rev == "baseline":
    test_bleus, test_matches = [], []
    for inst in data["test"]:
        bleu, match = accuracy(inst["out_tokens"], inst["in_tokens"])
        test_bleus.append(bleu)
        test_matches.append(match)

    logger.info("test_bleu={:.3f}, test_match={:.3f}".format(np.mean(test_bleus), np.mean(test_matches)))


