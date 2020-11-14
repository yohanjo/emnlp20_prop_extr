import subprocess
import requests
import json
import pickle
import os
import numpy as np
import torch
from csv_utils import *
import logging
from collections import defaultdict, Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import sentence_bleu
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


bleu_fn = bleu_score.SmoothingFunction().method3

def get_bleu(refs, hyp):
    bleu = sentence_bleu(refs, hyp, smoothing_function=bleu_fn)
    return bleu



ARG_IDS_PATH = "../data/arg_ids.txt"
DATA_DIR = "../data"
FOLDS_PATH = DATA_DIR + "/folds.csv"
BIGRAM_PATH = DATA_DIR + "/spch_2gram_idx.txt"  # Spch bigram
VOCA_GLOVE_PATH = DATA_DIR + "/voca_glove.csv"
WORD_EMBS_PATH = DATA_DIR + "/voca_glove_emb.txt"
POS_IDX_PATH = DATA_DIR + "/pos_idx.txt"
POS2_IDX_PATH = DATA_DIR + "/pos2_idx.txt"
LEMMA_IDX_PATH = DATA_DIR + "/lemma_idx.txt"
NER_IDX_PATH = DATA_DIR + "/ner_idx.txt"
SPCH_2GRAM_PATH = DATA_DIR + "/spch_2gram_idx.txt" 
WEMB_DIM = 300
MODEL_DIR = "../models"


def list_split(pred, l):
    res = [[]]
    for e in l:
        if pred(e): res.append([])
        else: res[-1].append(e)
    res = [sub_l for sub_l in res if len(sub_l) > 0]
    return res




def accuracy(conf_mat):
    """Returns: {"prec": #, "recl": #, "f1": #}."""
    res = dict()
    n_rows, n_cols = conf_mat.shape
    res["prec"] = [conf_mat[c, c]/conf_mat[:, c].sum() \
                            if conf_mat[:, c].sum() != 0 else 0 \
                        for c in range(n_cols)]
    res["recl"] = [conf_mat[r, r]/conf_mat[r].sum() \
                            if conf_mat[r].sum() != 0 else 0 \
                        for r in range(n_rows)]
    res["f1"] = [2*p*r / (p+r) if p != 0 and r != 0 else 0 \
                        for p, r in zip(res["prec"], res["recl"])]
    return res



def get_logger(path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    for handler in [logging.FileHandler(path, mode="w"),
                    logging.StreamHandler()]:
        logger.addHandler(handler)

    return logger

def cuda(tensor):
    if torch.cuda.is_available(): tensor = tensor.cuda()
    return tensor


def long_tensor(v):
    tensor = torch.LongTensor(v)
    if torch.cuda.is_available(): tensor = tensor.cuda()
    return tensor

def float_tensor(v):
    tensor = torch.FloatTensor(v)
    if torch.cuda.is_available(): tensor = tensor.cuda()
    return tensor

def zero_tensor(*size):
    tensor = torch.zeros(size)
    if torch.cuda.is_available(): tensor = tensor.cuda()
    return tensor


def pickle_dump(obj, path): 
    """Platform-safe pickle.dump. Identical to pickle.dump, except that
    pickle.dump may raise an error on Mac for a big file (>2GB)."""
    max_bytes = 2 ** 31 - 1
    out_bytes = pickle.dumps(obj)
    n_bytes = len(out_bytes)
    with open(path, "wb") as f:
        for idx in range(0, n_bytes, max_bytes):
            f.write(out_bytes[idx:(idx + max_bytes)])

def pickle_load(path):
    """Platform-safe pickle.load. Identical to pickle.load, except that
    pickle.load may raise an error on Mac for a big file."""
    max_bytes = 2 ** 31 - 1
    input_size = os.path.getsize(path)
    obj_bytes = bytearray(0)
    with open(path, "rb") as f:
        for _ in range(0, input_size, max_bytes):
            obj_bytes += f.read(max_bytes) 
    return pickle.loads(obj_bytes)


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


