"""
CRF for source identification in reported speech.
"""
from csv_utils import *
import numpy as np
import argparse
from collections import defaultdict, Counter
from itertools import product
import json
import re
from nltk.translate import bleu_score
from nltk.translate.bleu_score import sentence_bleu
from sklearn_crfsuite.metrics import flat_f1_score
from nltk.tree import Tree
import sklearn_crfsuite
from time import time

bleu_fn = bleu_score.SmoothingFunction().method3

parser = argparse.ArgumentParser()
parser.add_argument("-mode", required=True)
parser.add_argument("-data", default="../data/rspch_parc3.csv")
parser.add_argument("-print_top_feats", action="store_true")
parser.add_argument("-print_top_trans", action="store_true")
parser.add_argument("-print_result", action="store_true")
parser.add_argument("-save_model", action="store_true")
args = parser.parse_args()

feat_as_str = True

algorithms = ["lbfgs", "pa"]
feat_combs = [["word", "ner", "prev_1gram", "prev_2gram", "next_1gram", 
               "next_2gram", "pos", "subj"]]
lbfgs_c1s = [0, 0.05, 0.1, 0.2]
lbfgs_c2s = [0, 0.05, 0.1, 0.2]
pa_cs = [0.5, 1, 2, 4]

#lbfgs_c1s = [0.05]
#lbfgs_c2s = [0]
#pa_cs = [0.5]

class CRF(object):
    def __init__(self):
        self.voca_bio = ["B", "I", "O"]
        self.bio_idx = {"B": 0, "I": 1, "O": 2}

        self.feats = None

    def init_crf(self, params):
        """
            params: dict of parameters.
        """
        self.params = params
        self.crf = sklearn_crfsuite.CRF(**params)

    def get_feats(self, sents):

        clause_mask = []
        for sent in sents:
            clause_mask_ = ["O"] * len(sent["tokens"])
            parse = sent["parse"].copy(deep=True)
            for p in range(len(parse.leaves())):
                parse[parse.leaf_treeposition(p)] = p
            for clause_tree in parse.subtrees(lambda st: st.label() == "S"):
                clause_mask_[clause_tree.leaves()[0]] = "B"
                clause_mask_[clause_tree.leaves()[-1]] = "E"
            clause_mask += clause_mask_

        subj_mask = []
        for sent in sents:
            subj_mask_ = [0] * len(sent["tokens"])
            for dep in sent["enhancedDependencies"]:
                if dep["dep"].startswith("nsubj"):
                    subj_mask_[dep["dependent"] - 1] = 1
            subj_mask += subj_mask_

        tokens = [token for sent in sents for token in sent["tokens"]]

        feats = []
        for t, token in enumerate(tokens):
            feat = {}
            feat["bias"] = 1

            if "word" in self.feats:
                feat["word"] = token["originalText"].lower()
            
            if "pos" in self.feats:
                feat["pos"] = token["pos"][:2]

            if "ner" in self.feats:
                feat["ner"] = token["ner"]

            if "prev_1gram" in self.feats:
                feat["prev_1gram"] = tokens[t-1]["originalText"].lower() if t-1 >= 0 else "<S>"

            if "prev_2gram" in self.feats:
                bigram = (tokens[t-2]["originalText"].lower() if t-2 >= 0 else "<S>") + \
                         (tokens[t-1]["originalText"].lower() if t-1 >= 0 else "<S>")
                feat["prev_2gram"] = bigram

            if "next_1gram" in self.feats:
                feat["next_1gram"] = tokens[t+1]["originalText"].lower() if t+1 < len(tokens) else "</S>"

            if "next_2gram" in self.feats:
                bigram = (tokens[t+1]["originalText"].lower() if t+1 < len(tokens) else "</S>") + \
                         (tokens[t+2]["originalText"].lower() if t+2 < len(tokens) else "</S>")
                feat["next_2gram"] = bigram

            if "boc" in self.feats:
                feat["boc"] = int(clause_mask[t] == "B")

            if "eoc" in self.feats:
                feat["eoc"] = int(clause_mask[t] == "E")

            if "subj" in self.feats:
                feat["subj"] = subj_mask[t]

            feats.append(feat)

        return feats
 
    def train(self, X, y):
        self.crf.fit(X, y)

    def predict(self, sents):
        X = self.get_feats(sents)
        y = self.crf.predict([X]) # [[tag, ...]]
        tags_pred = [self.bio_idx[t] for t in y[0]]
        return tags_pred

    def test(self, X):
        y_pred = self.crf.predict(X)
        return y_pred

    def print_trans(self, trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    def print_feats(self, state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr))

    def get_top_feats(self, state_features):
        top_feats = defaultdict(list)
        for l in ["B", "I", "O"]:
            for (attr, label), weight in state_features:
                if l != label: continue
                top_feats[l].append(attr)
                if len(top_feats[label]) == N_TOP_FEATS: break
        return top_feats
                
    def print_top_trans(self):
        print("Top likely transitions:")
        self.print_trans(Counter(self.crf.transition_features_).most_common(4))

        print("\nTop unlikely transitions:")
        self.print_trans(Counter(self.crf.transition_features_).most_common()[-5:])

    def print_top_feats(self):
        print("Top positive:")
        self.print_feats(Counter(self.crf.state_features_)\
                            .most_common(N_TOP_FEATS))

        print("\nTop negative:")
        self.print_feats(Counter(self.crf.state_features_)\
                            .most_common()[::-1][:N_TOP_FEATS])


def accuracy_source(mask_true, mask_pred):
    assert len(mask_true) == len(mask_pred)
    
    sources_true, sources_pred = [], []
    for mask, sources in [(mask_true, sources_true), (mask_pred, sources_pred)]:
        for widx, label in enumerate(mask):
            if label == "B":
                sources.append([])
                sources[-1].append(widx)
            elif label == "I":
                if len(sources) == 0 or mask[widx-1] == "O":
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
 

print(f"Loading {args.data}...")
split2data = defaultdict(list)
for row in iter_csv_header(args.data):
    row["text_nlp"] = json.loads(row["text_nlp"])

    row["words"] = []
    for sent in row["text_nlp"]["sentences"]:
        sent["parse"] = Tree.fromstring(sent["parse"])
        for token in sent["tokens"]:
            row["words"].append(token["originalText"])
    row["label"] = json.loads(row[f"{args.mode}_mask"])
    row["target"] = row[args.mode]

    #print("Sentence:", row["text"])
    #print("Target:", row["target"])
    #print("Label:", row["label"])
    #print("")
    
    split2data[row["split"]].append(row)

print(", ".join([f"n_{split}={len(data)}" for split, data in split2data.items()]))

md = CRF()
perf = defaultdict(lambda: defaultdict(list))
feats_analy = {}
for feats in feat_combs:
    print("Features: {}".format(feats))

    md.feats = feats

    # Train
    X_train = []
    y_train = []
    for row in split2data["train"]:
        X_loc = md.get_feats(row["text_nlp"]["sentences"])
        X_train.append(X_loc)
        y_train.append(row["label"])

    max_val_acc = None
    max_test_accs = None
    for algorithm in algorithms:
        if algorithm == "lbfgs": 
            crf_config_iter = product(lbfgs_c1s, lbfgs_c2s)
        else: 
            crf_config_iter = product(pa_cs)

        for crf_config in crf_config_iter:
            start_time = time()

            print("Model: {}, config: {}".format(algorithm, crf_config))
            if algorithm == "lbfgs": 
                params = {"algorithm": algorithm, "c1": crf_config[0], 
                          "c2": crf_config[1]}
            else:
                params = {"algorithm": algorithm, "c": crf_config[0]}

            md.init_crf(params)
            md.train(X_train, y_train)

            print(" - n_features:", len(md.crf.state_features_) + \
                    len(md.crf.transition_features_))

            # Val & test
            val_accs, test_accs = {}, {}
            for split, accs in [("val", val_accs),
                                ("test", test_accs)]: 
                X, y = [], []
                for row in split2data[split]:
                    X_loc = md.get_feats(row["text_nlp"]["sentences"])
                    X.append(X_loc)
                    y.append(row["label"])
                y_pred = md.test(X)

                if args.mode == "content":
                    accs["f1"] = flat_f1_score(y, y_pred, average="macro")

                    bleus = []
                    for row, label_pred in zip(split2data[split], y_pred):
                        words = [t["originalText"].lower() for sent in \
                                    row["text_nlp"]["sentences"] \
                                    for t in sent["tokens"]]
                        ref = [word for w, word in enumerate(words) if row["label"][w] != "O"]
                        hyp = [word for w, word in enumerate(words) if label_pred[w] != "O"]
                        bleus.append(sentence_bleu([ref], hyp, smoothing_function=bleu_fn))
                    accs["bleu"] = np.mean(bleus)

                else:
                    accs_inst = defaultdict(list)
                    for row, label_pred in zip(split2data[split], y_pred):
                        acc_inst = accuracy_source(row["label"], label_pred)
                        for met, score in acc_inst.items():
                            accs_inst[met].append(score)
                    for met, scores in accs_inst.items():
                        accs[met] = np.mean(scores)

                print(f" - {split}: " + \
                      ", ".join(["{}={:.3f}".format(met, score) \
                                for met, score in accs.items()]))

            if max_val_acc is None or val_accs["f1"] > max_val_acc:
                max_val_acc = val_accs["f1"]
                max_test_accs = test_accs.copy()
    
            print(" - " + ", ".join(["{}_{}={}".format(split, met, score) \
                        for split, accs in [("val", val_accs), ("test", test_accs)] \
                        for met, score in accs.items()]))
            print(" - Time: {:.1f} mins".format((time() - start_time) / 60))


    print(", ".join(["max_test_{}={}".format(met, score) \
                        for met, score in max_test_accs.items()]))

    print("=====================================================")


""" 
if args.print_top_trans:
    md.print_top_trans()

if args.print_top_feats:
    md.print_top_feats()


# Print performance
os.makedirs(os.path.dirname(EVAL_PATH), exist_ok=True)
with open(EVAL_PATH, "w") as f:
    out_csv = csv.writer(f)
    out_csv.writerow(["features", "classifier_config"] + \
                     [f"{met}_{sfx}" for met in metrics \
                        for sfx in (["avg"] + [f"f{fold}" for fold in folds])] +\
                     ["B_feats", "I_feats", "O_feats"])
    for config, metric_fold_acc in perf.items():
        feats_str = config[0]
        row = list(config)
        for metric in metrics:
            row.append(np.mean(metric_fold_acc[metric]))
            row.extend(metric_fold_acc[metric])
        row.extend([", ".join(feats_analy[(feats_str, crf_config)]["B"]),
                    ", ".join(feats_analy[(feats_str, crf_config)]["I"]),
                    ", ".join(feats_analy[(feats_str, crf_config)]["O"])])
        out_csv.writerow(row)

"""                          
