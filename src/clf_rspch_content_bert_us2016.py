"""
BERT classifier for content identification for reported speech
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
import sys
from nltk.translate import bleu_score
from nltk.translate.bleu_score import sentence_bleu

bleu_fn = bleu_score.SmoothingFunction().method3

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

    model = BertForTokenClassification.from_pretrained(
                                'bert-base-uncased', num_labels=3)
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
                            labels=long_tensor(row["content_label"]).unsqueeze(0))
            loss, logits = outputs[:2]
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("train_loss:", train_loss)

        model.eval()
        with torch.no_grad():
            ys_true, ys_pred, bleu = [], [], []
            for row in val_data:
                outputs = model(long_tensor(row["text_tokens"]).unsqueeze(0),
                      labels=long_tensor(row["content_label"]).unsqueeze(0))
                loss, logits = outputs[:2]
                val_loss += loss.item()

                label_pred = logits[0].argmax(dim=1).cpu().numpy().tolist()

                # Accuracy
                for y_t, y_p in zip(row["content_label"], label_pred):
                    ys_true.append(y_t)
                    ys_pred.append(y_p)

                # BLEU
                ref = " ".join(tokenizer.convert_ids_to_tokens(
                                [row["text_tokens"][w] for w, label in \
                                 enumerate(row["content_label"]) if label != 0]))
                hyp = " ".join(tokenizer.convert_ids_to_tokens(
                                [row["text_tokens"][w] for w, label in \
                                 enumerate(label_pred) if label != 0]))
                bleu.append(sentence_bleu([ref], hyp, smoothing_function=bleu_fn))

            acc["val_bleu"] = np.mean(bleu)
            acc["val_prec"], acc["val_recl"], acc["val_f1"], _ = \
                    precision_recall_fscore_support(ys_true, ys_pred,
                                                    average="macro")
            print("val_loss={}, {}".format(
                    val_loss, 
                    ", ".join(["val_{}={}".format(metric, acc[f"val_{metric}"])\
                                    for metric in ["prec", "recl", "f1", "bleu"]])))

            ys_true, ys_pred, bleu = [], [], []
            for row in test_data:
                outputs = model(long_tensor(row["text_tokens"]).unsqueeze(0),
                      labels=long_tensor(row["content_label"]).unsqueeze(0))
                loss, logits = outputs[:2]
                test_loss += loss.item()

                label_pred = logits[0].argmax(dim=1).cpu().numpy().tolist()
                row["content_label_pred"] = label_pred
                for y_t, y_p in zip(row["content_label"], label_pred):
                    ys_true.append(y_t)
                    ys_pred.append(y_p)

                # BLEU
                ref = " ".join(tokenizer.convert_ids_to_tokens(
                                [row["text_tokens"][w] for w, label in \
                                 enumerate(row["content_label"]) if label != 0]))
                hyp = " ".join(tokenizer.convert_ids_to_tokens(
                                [row["text_tokens"][w] for w, label in \
                                 enumerate(label_pred) if label != 0]))
                bleu.append(sentence_bleu([ref], hyp, smoothing_function=bleu_fn))

                #print("Ref:", ref)
                #print("Hyp:", hyp)
                #print("")
            acc["test_bleu"] = np.mean(bleu)
            acc["test_prec"], acc["test_recl"], acc["test_f1"], _ = \
                    precision_recall_fscore_support(ys_true, ys_pred,
                                                    average="macro")
            print("test_loss={}, {}".format(
                    test_loss, 
                    ", ".join(["test_{}={}".format(
                                    metric, acc[f"test_{metric}"])\
                                    for metric in ["prec", "recl", "f1", "bleu"]])))

            if acc[f"max_val_f1"] is None or \
                    acc[f"val_f1"] > acc[f"max_val_f1"]:
                for m in ["prec", "recl", "f1", "bleu"]:
                    acc[f"max_val_{m}"] = acc[f"val_{m}"]
                    acc[f"max_test_{m}"] = acc[f"test_{m}"]

                for row in test_data:
                    row["content_label_pred_best"] = row["content_label_pred"]

    print(", ".join(["max_val_{}={}".format(
                        metric, acc[f"max_val_{metric}"]) \
                        for metric in ["prec", "recl", "f1", "bleu"]]))
    print(", ".join(["max_test_{}={}".format(
                        metric, acc[f"max_test_{metric}"]) \
                        for metric in ["prec", "recl", "f1", "bleu"]]))

    for metric in ["prec", "recl", "f1", "bleu"]:
        max_val_accs[metric].append(acc[f"max_val_{metric}"])
    for metric in ["prec", "recl", "f1", "bleu"]:
        max_test_accs[metric].append(acc[f"max_test_{metric}"])

print("=================================================================")
print(", ".join(["final_val_{}={}".format(
                    metric, np.mean(max_val_accs[metric])) \
                    for metric in ["prec", "recl", "f1", "bleu"]]))
print(", ".join(["final_test_{}={}".format(
                    metric, np.mean(max_test_accs[metric])) \
                    for metric in ["prec", "recl", "f1", "bleu"]]))
print("Total time: {:.1f}m".format((time() - start_time) / 60))

print("\n=================================================================")
for fold in range(args.n_folds):
    for row in fold2data[fold]:
        print("Sentence:", row["text"])
        print("True:", row["content_label"])
        print("Pred:", row["content_label_pred_best"])
        print("")

