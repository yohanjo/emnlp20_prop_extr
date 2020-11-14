"""
BERT classifier for source identification in reported speech.
"""
from csv_utils import *
from pytorch_transformers import BertForSequenceClassification, BertForTokenClassification, BertTokenizer, AdamW
from itertools import product
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle
from time import time
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n_folds", default=5)
parser.add_argument("-n_epochs", type=int, default=10)
parser.add_argument("-data", default="../data/rspch_us2016.csv")
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

def match(ref, tgt):
    r = 0
    mask = []
    while r <= (len(ref) - len(tgt)):
        if ref[r] == tgt[0]:
            # Check if ref[r:] == tgt
            if all(ref[r+t] == tgt[t] for t in range(len(tgt))):
                mask.append(1)
                mask.extend([2] * (len(tgt) - 1))
                r += len(tgt)
            else:
                mask.append(0)
                r += 1
        else:
            mask.append(0)
            r += 1
    mask.extend([0] * (len(ref) - len(mask)))
    return mask

def accuracy_source(source, text_tokens, label_pred):
    """source: source_token [24, 21, 11]
       logits: (seq_len, n_labels)"""
    sources_pred = []

    add = False
    for text_token, label in zip(text_tokens, label_pred):
        if label == 1: 
            sources_pred.append([text_token])
            add = True
        elif label == 2 and add: 
            sources_pred[-1].append(text_token)
            add = True
        else:
            add = False

    acc = {
        "prec_str": np.mean([int(source_pred == source) for source_pred in \
                                sources_pred]) if sources_pred else 0,
        "recl_str": float(source in sources_pred),
        "prec_rlx": np.mean([len(set(source_pred) & set(source)) > 0 \
                                for source_pred in sources_pred]) \
                            if sources_pred else 0,
        "recl_rlx": any(set(source_pred) & set(source) for source_pred in \
                                sources_pred) if sources_pred else 0
    }
    return acc

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print(f"Loading {args.data}...")
fold2data = defaultdict(list)
for row in iter_csv_header(args.data):
    if row["is_source"] == "no":
        continue
    fold2data[int(row["fold"])].append(row)


print("Tokenizing...")
for data in fold2data.values():
    for row in data:
        text_tokens = tokenize(row["text"], True)
        source_tokens = tokenize(row["source"], False)
        content_tokens = tokenize(row["content"], False)

        source_label = match(text_tokens, source_tokens) 
        content_label = match(text_tokens, content_tokens)

        row["text_tokens"] = text_tokens
        row["source_tokens"] = source_tokens
        row["content_tokens"] = content_tokens

        row["source_label"] = source_label
        row["content_label"] = content_label

        #print("Text:", row["text"])
        #print("Source:", row["source"])
        #print("Label:", row["source_label"])
        #print("")


print("Bert...")
start_time = time()
max_val_accs = defaultdict(list)
max_test_accs = defaultdict(list)
for test_fold in range(args.n_folds):
    print("Fold", test_fold)

    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=3)
    model.to(torch.device("cuda"))
    optimizer = AdamW(model.parameters(),
                           lr=args.learn_rate, 
                           eps=args.adam_epsilon)


    test_data = fold2data[test_fold]
    val_fold = (test_fold + 1) % args.n_folds
    val_data = fold2data[val_fold]
    train_data = [row for fold, data in fold2data.items() for row in data \
                    if fold not in [test_fold, val_fold]]

    acc = defaultdict(lambda: None)
    for epoch in range(args.n_epochs):
        print("Epoch:", epoch+1)

        model.train()
        train_loss = val_loss = test_loss = 0
        for row in train_data:
            outputs = model(long_tensor(row["text_tokens"]).unsqueeze(0),
                            labels=long_tensor(row["source_label"]).unsqueeze(0))
            loss, logits = outputs[:2]
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("train_loss:", train_loss)

        model.eval()
        with torch.no_grad():
            accs = defaultdict(list)
            for row in val_data:
                outputs = model(long_tensor(row["text_tokens"]).unsqueeze(0),
                      labels=long_tensor(row["source_label"]).unsqueeze(0))
                loss, logits = outputs[:2]
                val_loss += loss.item()

                label_pred = logits[0].argmax(dim=1).cpu().numpy().tolist()
                val_acc = accuracy_source(row["source_tokens"], 
                                          row["text_tokens"], label_pred)
                for metric, mode in product(["prec", "recl"], ["str", "rlx"]):
                    accs[f"{metric}_{mode}"].append(val_acc[f"{metric}_{mode}"])
            for mode in ["str", "rlx"]:
                acc[f"val_prec_{mode}"] = np.mean(accs[f"prec_{mode}"])
                acc[f"val_recl_{mode}"] = np.mean(accs[f"recl_{mode}"])
                acc[f"val_f1_{mode}"] = 2 * acc[f"val_prec_{mode}"] * \
                            acc[f"val_recl_{mode}"] / \
                            (acc[f"val_prec_{mode}"] + acc[f"val_recl_{mode}"])

            print("val_loss={}, {}".format(
                    val_loss, 
                    ", ".join([f"val_{metric}_{mode}=" + \
                                    str(acc[f"val_{metric}_{mode}"]) \
                                    for mode in ["str", "rlx"] for metric in \
                                    ["prec", "recl", "f1"]])))

            accs = defaultdict(list)
            for row in test_data:
                outputs = model(long_tensor(row["text_tokens"]).unsqueeze(0),
                      labels=long_tensor(row["source_label"]).unsqueeze(0))
                loss, logits = outputs[:2]
                test_loss += loss.item()

                label_pred = logits[0].argmax(dim=1).cpu().numpy().tolist()
                row["source_label_pred"] = label_pred

                test_acc = accuracy_source(row["source_tokens"], 
                                          row["text_tokens"], label_pred)
                for metric, mode in product(["prec", "recl"], ["str", "rlx"]):
                    accs[f"{metric}_{mode}"].append(test_acc[f"{metric}_{mode}"])
            for mode in ["str", "rlx"]:
                acc[f"test_prec_{mode}"] = np.mean(accs[f"prec_{mode}"])
                acc[f"test_recl_{mode}"] = np.mean(accs[f"recl_{mode}"])
                acc[f"test_f1_{mode}"] = 2 * acc[f"test_prec_{mode}"] * \
                        acc[f"test_recl_{mode}"] / \
                        (acc[f"test_prec_{mode}"] + acc[f"test_recl_{mode}"])

            print("test_loss={}, {}".format(
                    test_loss, 
                    ", ".join([f"test_{metric}_{mode}=" + \
                                    str(acc[f"test_{metric}_{mode}"]) \
                                for mode in ["str", "rlx"] for metric in \
                                ["prec", "recl", "f1"]])))

            for mode in ["str", "rlx"]:
                if acc[f"max_val_f1_{mode}"] is None or \
                        acc[f"val_f1_{mode}"] > acc[f"max_val_f1_{mode}"]:
                    for m in ["prec", "recl", "f1"]:
                        acc[f"max_val_{m}_{mode}"] = acc[f"val_{m}_{mode}"]
                        acc[f"max_test_{m}_{mode}"] = acc[f"test_{m}_{mode}"]

                    if mode == "str":
                        for row in test_data:
                            row["source_label_pred_best"] = \
                                            row["source_label_pred"]

    print(", ".join([f"max_val_{metric}_{mode}=" + \
                            str(acc[f"max_val_{metric}_{mode}"]) \
            for mode in ["str", "rlx"] for metric in ["prec", "recl", "f1"]]))
    print(", ".join([f"max_test_{metric}_{mode}=" + \
                            str(acc[f"max_test_{metric}_{mode}"]) \
            for mode in ["str", "rlx"] for metric in ["prec", "recl", "f1"]]))

    for metric, mode in product(["prec", "recl", "f1"], ["str", "rlx"]):
        max_val_accs[f"{metric}_{mode}"].append(
                acc[f"max_val_{metric}_{mode}"])
    for metric, mode in product(["prec", "recl", "f1"], ["str", "rlx"]):
        max_test_accs[f"{metric}_{mode}"].append(
                acc[f"max_test_{metric}_{mode}"])

print("=================================================================")
print(", ".join([f"final_val_{metric}_{mode}=" + \
                        str(np.mean(max_val_accs[f"{metric}_{mode}"])) \
        for mode in ["str", "rlx"] for metric in ["prec", "recl", "f1"]]))
print(", ".join([f"final_test_{metric}_{mode}=" + \
                        str(np.mean(max_test_accs[f"{metric}_{mode}"])) \
        for mode in ["str", "rlx"] for metric in ["prec", "recl", "f1"]]))
print("Total time: {:.1f}m".format((time() - start_time) / 60))

print("\n=================================================================")
for fold in range(args.n_folds):
    for row in fold2data[fold]:
        print("Sentence:", row["text"])
        print("True:", row["source_label"])
        print("Pred:", row["source_label_pred_best"])
        print("")

