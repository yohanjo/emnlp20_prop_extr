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
parser.add_argument("-n_folds", default=5)
parser.add_argument("-mode", required=True)
parser.add_argument("-data", default="../data/rspch_us2016.csv")
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

def get_label(text, text_nlp, target):
    # Masking target characters
    mask = [0] * len(text)
    for m in re.finditer(re.escape(target), text):
        mask[m.start()] = 1
        for i in range(m.start()+1, m.end()):
            mask[i] = 2

    label = []
    targets = []
    for sent in text_nlp["sentences"]:
        for token in sent["tokens"]:
            token_mask = mask[token["characterOffsetBegin"]:token["characterOffsetEnd"]]
            if 1 in token_mask:  # Beginning
                label.append("B")
                targets.append([token["originalText"]])
            elif 2 in token_mask:  # Inside
                label.append("I")
                targets[-1].append(token["originalText"])
            else:  # Other
                label.append("O")
    return label, targets

def accuracy_source(source, words, label_pred):
    """source: ["Senate", "Clinton"],
        words: ["Yesterday", "senate", "clinton", "said", ...],
        label_pred: ["O", "B", "I", "O", ...]"""
    sources_pred = []

    add = False
    for word, tag in zip(words, label_pred):
        if tag == "B":
            sources_pred.append([word])
            add = True
        elif tag == "I" and add:
            sources_pred[-1].append(word)
            add = True
        else:
            add = False

    acc = {
        "prec": np.mean([int(source_pred == source) for source_pred in \
                                sources_pred]) if sources_pred else 0,
        "recl": float(source in sources_pred),
        "prec_rlx": np.mean([len(set(source_pred) & set(source)) > 0 \
                                for source_pred in sources_pred]) \
                            if sources_pred else 0,
        "recl_rlx": any(set(source_pred) & set(source) for source_pred in \
                                sources_pred) if sources_pred else 0
    }
    acc["f1"] = 2 * acc["prec"] * acc["recl"] / (acc["prec"] + acc["recl"]) \
                if acc["prec"] > 0 and acc["recl"] > 0 else 0
    acc["f1_rlx"] = 2 * acc["prec_rlx"] * acc["recl_rlx"] / (acc["prec_rlx"] + acc["recl_rlx"]) \
                if acc["prec_rlx"] > 0 and acc["recl_rlx"] > 0 else 0
    return acc
 

print(f"Loading {args.data}...")
fold2data = defaultdict(list)
for row in iter_csv_header(args.data):
    if row["is_source"] != "yes": continue
    row["text_nlp"] = json.loads(row["text_nlp"])

    row["words"] = []
    for sent in row["text_nlp"]["sentences"]:
        sent["parse"] = Tree.fromstring(sent["parse"])
        for token in sent["tokens"]:
            row["words"].append(token["originalText"])
    row["label"], targets = get_label(row["text"], row["text_nlp"], row[args.mode].strip())
    row["target"] = targets[0]

    #print("Sentence:", row["text"])
    #print("Target:", row["target"])
    #print("Label:", row["label"])
    #print("")
    
    fold2data[int(row["fold"])].append(row)

md = CRF()
perf = defaultdict(lambda: defaultdict(list))
feats_analy = {}
for feats in feat_combs:
    print("Features: {}".format(feats))
    md.feats = feats

    scores = defaultdict(list)
    for test_fold in range(args.n_folds):
        print("Fold", test_fold)
        test_data = fold2data[test_fold]
        val_fold = (test_fold + 1) % args.n_folds
        val_data = fold2data[val_fold]
        train_data = [row for fold, data in fold2data.items() for row in data \
                        if fold not in [test_fold, val_fold]]


        # Train
        X_train = []
        y_train = []
        for row in train_data:
            X_loc = md.get_feats(row["text_nlp"]["sentences"])
            X_train.append(X_loc)
            y_train.append(row["label"])

        score = {}
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

                # Val
                X_val = []
                y_val = []
                for row in val_data:
                    X_loc = md.get_feats(row["text_nlp"]["sentences"])
                    X_val.append(X_loc)
                    y_val.append(row["label"])
                y_val_pred = md.test(X_val)
                if args.mode == "content":
                    score["f1_val"] = flat_f1_score(y_val, y_val_pred, average="macro")
                    # BLEU
                    bleu = []
                    for row, y_pred in zip(val_data, y_val_pred):
                        words = [t["originalText"].lower() for sent in \
                                    row["text_nlp"]["sentences"] \
                                    for t in sent["tokens"]]
                        ref = [word for w, word in enumerate(words) if row["label"][w] != "O"]
                        hyp = [word for w, word in enumerate(words) if y_pred[w] != "O"]
                        bleu.append(sentence_bleu([ref], hyp, smoothing_function=bleu_fn))
                    score["bleu_val"] = np.mean(bleu)
                else:
                    accs = defaultdict(list)
                    for row, label_pred in zip(val_data, y_val_pred):
                        acc = accuracy_source(row["target"], row["words"], label_pred)
                        for key, val in acc.items():
                            accs[f"{key}_val"].append(val)
                    for key, vals in accs.items():
                        score[key] = np.mean(vals)

                # Test
                X_test = []
                y_test = []
                for row in test_data:
                    X_loc = md.get_feats(row["text_nlp"]["sentences"])
                    X_test.append(X_loc)
                    y_test.append(row["label"])
                y_test_pred = md.test(X_test)
                if args.mode == "content":
                    score["f1_test"] = flat_f1_score(y_test, y_test_pred, average="macro")
                    # BLEU
                    bleu = []
                    for row, y_pred in zip(test_data, y_test_pred):
                        words = [t["originalText"].lower() for sent in \
                                    row["text_nlp"]["sentences"] \
                                    for t in sent["tokens"]]
                        ref = [word for w, word in enumerate(words) if row["label"][w] != "O"]
                        hyp = [word for w, word in enumerate(words) if y_pred[w] != "O"]
                        bleu.append(sentence_bleu([ref], hyp, smoothing_function=bleu_fn))
                    score["bleu_test"] = np.mean(bleu)
                else:
                    accs = defaultdict(list)
                    for row, label_pred in zip(test_data, y_test_pred):
                        acc = accuracy_source(row["target"], row["words"], label_pred)
                        for key, val in acc.items():
                            accs[f"{key}_test"].append(val)
                    for key, vals in accs.items():
                        score[key] = np.mean(vals)


                if "max_f1_val" not in score or score["f1_val"] > score["max_f1_val"]:
                    for key, val in [(key, val) for key, val in score.items() if not key.startswith("max")]:
                        score[f"max_{key}"] = score[key]
    
                print(" - " + ", ".join(["{}={:.3f}".format(key, val) \
                                            for key, val in score.items() \
                                            if not key.startswith("max")]))
                print(" - Time: {:.6f} mins".format((time() - start_time) / 60))
        
                #print(", ".join(["{}: {:.3f}".format(m, s) \
                #                    for m, s in score.items()]))
        print(", ".join(["{}: {:.3f}".format(m, s) \
                            for m, s in [(m, s) for m, s in score.items() \
                                                if m.startswith("max")]]))

        for metric in score.keys():
            if not metric.startswith("max"): continue
            scores[metric].append(score[metric])



    print(", ".join(["final_{}={}".format(m, np.mean(scores[m])) for m in scores.keys()]))
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
