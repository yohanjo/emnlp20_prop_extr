"""
BERT classifier for source identification in reported speech.
"""
from csv_utils import *
from pytorch_transformers import BertForSequenceClassification, BertForTokenClassification, BertTokenizer, AdamW
from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle
from time import time
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import argparse
from tqdm import tqdm
from nltk.translate import bleu_score
from nltk.translate.bleu_score import sentence_bleu
import json

bleu_fn = bleu_score.SmoothingFunction().method3

parser = argparse.ArgumentParser()
parser.add_argument("-mode", required=True)
parser.add_argument("-n_epochs", type=int, default=10)
parser.add_argument("-data", default="../data/rspch_parc3.csv")
parser.add_argument("-learn_rate", default=1e-5)
parser.add_argument("-adam_epsilon", default=1e-8, type=float)
args = parser.parse_args()

def long_tensor(v):
    tensor = torch.LongTensor(v)
    if torch.cuda.is_available(): tensor = tensor.cuda()
    return tensor

def float_tensor(v):
    tensor = torch.FloatTensor(v)
    if torch.cuda.is_available(): tensor = tensor.cuda()
    return tensor

def tokenize(text, pad):
    text = text.strip()
    text_tokens = tokenizer.tokenize(text)[:510]
    if pad:
        text_tokens = ["[CLS]"] + text_tokens + ["[SEP]"]

    input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    return input_ids

def accuracy_source(mask_true, mask_pred):
    assert len(mask_true) == len(mask_pred)
    
    sources_true, sources_pred = [], []
    for mask, sources in [(mask_true, sources_true), (mask_pred, sources_pred)]:
        for widx, label in enumerate(mask):
            if label == 0:
                sources.append([])
                sources[-1].append(widx)
            elif label == 1:
                if len(sources) == 0 or mask[widx-1] == 2:
                    sources.append([])
                sources[-1].append(widx)

    acc = {
        "prec": np.mean([int(src_pred in sources_true) \
                         for src_pred in sources_pred]) if sources_pred else 0,
        "recl": np.mean([int(src_true in sources_pred) \
                         for src_true in sources_true]) if sources_true else 0,
        "prec_rlx": np.mean([int(any(len(set(src_pred) & set(src_true)) > 0 \
                                     for src_true in sources_true)) \
                        for src_pred in sources_pred]) if sources_pred else 0,
        "recl_rlx": np.mean([int(any(len(set(src_pred) & set(src_true)) > 0 \
                                     for src_pred in sources_pred)) \
                        for src_true in sources_true]) if sources_true else 0,
    }
    acc["f1"] = 2 * acc["prec"] * acc["recl"] / (acc["prec"] + acc["recl"]) \
                if acc["prec"] > 0 and acc["recl"] > 0 else 0
    acc["f1_rlx"] = 2 * acc["prec_rlx"] * acc["recl_rlx"] / (acc["prec_rlx"] + acc["recl_rlx"]) \
                if acc["prec_rlx"] > 0 and acc["recl_rlx"] > 0 else 0
    return acc


def accuracy_content(mask_true, mask_pred, text_tokens):
    acc = {}

    ys_true = []
    ys_pred = []
    for y_t, y_p in zip(mask_true, mask_pred):
        ys_true.append(y_t)
        ys_pred.append(y_p)

    acc["prec"], acc["recl"], acc["f1"], _ = \
        precision_recall_fscore_support(ys_true, ys_pred, average="macro")

    # BLEU
    ref = " ".join(tokenizer.convert_ids_to_tokens(
                    [token for label, token in \
                     zip(mask_true, text_tokens) if label != 2]))
    hyp = " ".join(tokenizer.convert_ids_to_tokens(
                    [token for label, token in \
                     zip(mask_pred, text_tokens) if label != 2]))
    acc["bleu"] = sentence_bleu([ref], hyp, smoothing_function=bleu_fn)

    #print("Ref:", ref)
    #print("Hyp:", hyp)
    #print("")

    return acc
 
label2idx = {"B": 0, "I": 1, "O": 2}
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print(f"Loading {args.data}...")
split2data = defaultdict(list)
for row in iter_csv_header(args.data):
    #if len(split2data[row["split"]]) == 100: continue

    if row[args.mode] == "": continue

    row["text_tokens"] = []
    row["label"] = []
    mask = json.loads(row[f"{args.mode}_mask"])
    for word, label in zip(row["text"].split(" "), 
                           json.loads(row[f"{args.mode}_mask"])):
        tokens = tokenize(word, False)
        row["text_tokens"].extend(tokens)
        row["label"].extend([label2idx[label]] + ([label2idx["I"] if label=="B" else label2idx[label]] * (len(tokens)-1)))
        if len(row["text_tokens"]) == 510: break

    row["text_tokens"] = [tokenizer.convert_tokens_to_ids("[CLS]")] + \
                            row["text_tokens"] + \
                         [tokenizer.convert_tokens_to_ids("[SEP]")]
    row["label"] = [label2idx["O"]] + row["label"] + [label2idx["O"]]

    assert len(row["text_tokens"]) == len(row["label"])

    split2data[row["split"]].append(row)

print(", ".join([f"n_{split}={len(data)}" for split, data in split2data.items()]))

print("Bert...")
max_test_accs = defaultdict(list)

model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.to(torch.device("cuda"))
optimizer = AdamW(model.parameters(),
                       lr=args.learn_rate, 
                       eps=args.adam_epsilon)

start_time = time()

max_val_accs = None
max_test_accs = None
for epoch in range(args.n_epochs):
    print("Epoch:", epoch+1)

    model.train()
    train_loss = 0
    for row in tqdm(split2data["train"]):
        outputs = model(long_tensor(row["text_tokens"]).unsqueeze(0),
                        labels=long_tensor(row["label"]).unsqueeze(0))
        loss, logits = outputs[:2]
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("train_loss:", train_loss)

    model.eval()
    val_accs = defaultdict(lambda: None)
    test_accs = defaultdict(lambda: None)
    for split, data, accs in [("val", split2data["val"], val_accs), 
                              ("test", split2data["test"], test_accs)]:
        accs_inst = defaultdict(list)
        loss_sum = 0
        with torch.no_grad():
            for row in tqdm(data):
                outputs = model(long_tensor(row["text_tokens"]).unsqueeze(0),
                      labels=long_tensor(row["label"]).unsqueeze(0))
                loss, logits = outputs[:2]
                loss_sum += loss.item()

                label_pred = logits[0].argmax(dim=1).cpu().numpy().tolist()

                if args.mode == "source":
                    acc_inst = accuracy_source(row["label"], label_pred)
                else:
                    acc_inst = accuracy_content(row["label"], label_pred, row["text_tokens"])
                for met, score in acc_inst.items():
                    accs_inst[met].append(score)

        for met, scores in accs_inst.items():
            accs[met] = np.mean(scores)

        print("{}_loss={}, {}".format(
                split,
                loss_sum, 
                ", ".join([f"{split}_{met}={score}" \
                            for met, score in accs.items()])))

    # Save max acc
    if max_val_accs is None or \
            val_accs["f1"] > max_val_accs["f1"]:
        max_val_accs = val_accs.copy()
        max_test_accs = test_accs.copy()
        print("(Updating max...)")

print("\n=================================================================")
print(", ".join([f"max_val_{met}={score}" for met, score in max_val_accs.items()]))
print(", ".join([f"max_test_{met}={score}" for met, score in max_test_accs.items()]))
print("Total time: {:.1f}m".format((time() - start_time) / 60))

#for fold in range(args.n_folds):
#    for row in fold2data[fold]:
#        print("Sentence:", row["text"])
#        print("True:", row["source_label"])
#        print("Pred:", row["source_label_pred_best"])
#        print("")
#
