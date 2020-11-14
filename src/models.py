from helper import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from queue import PriorityQueue
import re

N_TOP_FEATS = 30  # Num of top features to display


class Revisor2(object):
    def __init__(self, args, wembs, voca_word, word_idx):
        self.args = args
        self.max_loc_len = args.rev_len
        self.enc_hid_dim = args.rev_enc_dim
        self.enc_out_dim = 2 * args.rev_enc_dim
        self.beam_size = args.beam_size
        self.max_prop_len = args.max_prop_len
        self.voca_word = voca_word
        self.decoder_type = args.rev
        self.word_idx = word_idx

        # Encoder
        self.encoder = Encoder(wembs, self.enc_hid_dim, bidir=True) 

        # Decoder
        if args.rev == "basic":
            print("AttnDecoder")
            self.decoder = AttnDecoder(wembs, self.enc_out_dim)
        elif args.rev == "copy":
            print("CopyDecoder")
            self.decoder = CopyDecoder(wembs, self.enc_out_dim, 
                                       self.enc_out_dim,
                                       args.copy_weight, args.copy_mech)
        else:
            raise Exception("Invalid revisor:", args.rev)
        self.nll_fn = nn.NLLLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
                [{"params": self.encoder.parameters()}, 
                 {"params": self.decoder.parameters()}],
                lr=args.learn_rate)

    def cuda(self, device):
        self.encoder.to(device)
        self.decoder.to(device)

    def nn_engines(self):
        return [self.encoder, self.decoder]

    def save_engines(self, save_dir, fold):
        print("Saving revisor engines...")
        torch.save(self.encoder.state_dict(), 
                   f"{save_dir}/revisor_encoder_f{fold}.pt")
        torch.save(self.decoder.state_dict(),
                   f"{save_dir}/revisor_decoder_f{fold}.pt")

    def load_engines(self, fold):
        self.encoder.load_state_dict(torch.load(
                   f"{MODEL_DIR}/revisor_encoder_f{fold}.pt"))
        self.decoder.load_state_dict(torch.load(
                   f"{MODEL_DIR}/revisor_decoder_f{fold}.pt"))

    def revise(self, in_words, out_words, mode):
        """
            Args:
                in_words: words that correspond to enc_outs (for copy mech).
        """
        # Encoding
        enc_outs, _ = self.encoder(long_tensor(in_words))  

        # Decoding
        enc_out_first = enc_outs[0].view(-1, self.enc_hid_dim)[-1]
        enc_out_last = enc_outs[-1].view(-1, self.enc_hid_dim)[0]
        enc_out_comb = torch.cat((enc_out_first, enc_out_last), 0).view(1, -1)
        dec_hid = (enc_out_comb.view(1, 1, -1),
                   zero_tensor(1, 1, self.enc_out_dim))

        # Training (teacher forcing)
        prop_words_pred = []
        loss = 0
        if mode == "train":
            attn_cp = None
            for w, word in enumerate(out_words[:-1]):
                if self.decoder_type == "copy":
                    word_probs, dec_hid, attn_cp = \
                            self.decoder(long_tensor([word]), dec_hid, 
                                         enc_outs, in_words, attn_cp) 
                else:
                    word_probs, dec_hid = self.decoder(long_tensor([word]), 
                                                       dec_hid, enc_outs) 
                nll_loss = self.nll_fn(word_probs.log(), 
                                     long_tensor([out_words[w + 1]]))

                loss += nll_loss

            loss /= len(out_words) - 1  # SOP

            # Optimize
            if not torch.isnan(loss):
                self.optimizer.zero_grad()
                loss.backward()
                params = list(self.encoder.parameters()) + \
                         list(self.decoder.parameters())
                torch.nn.utils.clip_grad_value_(params, 1.)
                self.optimizer.step()


        # Testing or generating (beam search)
        elif len(in_words) <= self.max_loc_len: 
            # Initialize priority queue
            node = BeamSearchNode(dec_hid, None, self.word_idx["[SOP]"], 
                                  0, 1, None)
            in_nodes = [node]
            end_nodes = []

            # Loop over queue
            while True:
                queue = PriorityQueue()
                for node in in_nodes:
                    prev_word = node.prev_word
                    dec_hid = node.dec_hid
                    attn_cp = node.attn_cp

                    # Decode
                    if self.decoder_type == "copy":
                        word_probs, dec_hid, attn_cp = \
                                self.decoder(long_tensor([prev_word]), dec_hid,
                                             enc_outs, in_words, attn_cp) 
                    else:
                        word_probs, dec_hid = self.decoder(
                                                long_tensor([prev_word]),
                                                dec_hid, enc_outs)
                    assert(.8 < word_probs.sum() < 1.2)

                    # Beam search
                    pws, idxs = word_probs.topk(self.beam_size)
                    for k in range(self.beam_size):
                        word_prob = pws[0][k].item()
                        if word_prob == 0: continue
                        word_pred = idxs[0][k].item()

                        node_new = BeamSearchNode(dec_hid, node, word_pred, 
                                                  node.log_p+np.log(word_prob),
                                                  node.length + 1,
                                                  attn_cp) 

                        # Move the node to end_nodes if complete
                        if node_new.prev_word == self.word_idx["[EOG]"] or \
                                node_new.length == self.max_prop_len:
                            end_nodes.append((-node_new.avg_log_p(), node_new))
                        else:
                            queue.put((-node_new.avg_log_p(), node_new))

                if len(end_nodes) >= self.beam_size: break

                in_nodes = []
                for k in range(min(self.beam_size, queue.qsize())):
                    in_nodes.append(queue.get()[1])
 
            # Choose top prop with backtracking
            neg_avg_log_p, node_trace = sorted(end_nodes)[0]
            while node_trace is not None:
                prop_words_pred.insert(0, node_trace.prev_word)
                node_trace = node_trace.prev_node

        else:
            prop_words_pred = in_words + \
                              [word_idx[w] for w in ["[EOP]", "[EOG]"]]

        return loss.item() if type(loss) is Tensor else loss, prop_words_pred



class Encoder(nn.Module):
    def __init__(self, wembs_w, rnn_hid_dim, bidir=True, 
                 linear_wemb=True, linear_femb=True):
        """
            Args:
                wembs_w: Pretrained word embedding weights.
                rnn_hid_dim: RNN hidden dimensionality.
                feat_size: {feat: feat_dim, ...}.
        """
        super(Encoder, self).__init__()

        self.n_direct = 2 if bidir else 1
        
        voca_size, wemb_dim = wembs_w.size()
        self.word2emb = nn.Sequential(nn.Embedding.from_pretrained(wembs_w),
                                      nn.ReLU())

        self.rnn_hid_dim = rnn_hid_dim
        self.rnn = nn.LSTM(wemb_dim, rnn_hid_dim, bidirectional=bidir)

    def init_rnn_hid(self):
        return (cuda(torch.zeros(self.n_direct, 1, self.rnn_hid_dim)),
                cuda(torch.zeros(self.n_direct, 1, self.rnn_hid_dim)))

    def forward(self, sentence):
        """
            Args:
                sentence: [word, ...]. (sent_len)
            Returns:
                rnn_outs: All hids (seq_len x 1 x n_direct * hid_dim)
                rnn_out: Last hid ((n_direct x 1 x hid_dim) * 2) for LSTM
        """
        wembs = self.word2emb(sentence)  # (seq_len x hid_dim)
        rnn_outs, rnn_out = self.rnn(wembs.unsqueeze(1), self.init_rnn_hid())
        return rnn_outs, rnn_out


class BasicEncoder(nn.Module):
    def __init__(self, wembs_w, rnn_hid_dim, feat_dim, bidir=False, 
                 linear_wemb=True, linear_femb=True):
        """
            Args:
                wembs_w: Pretrained word embedding weights.
                rnn_hid_dim: RNN hidden dimensionality.
                feat_size: {feat: feat_dim, ...}.
        """
        super(BasicEncoder, self).__init__()

        self.n_direct = 1 if not bidir else 2
        self.feat_dim = feat_dim
        self.feat_names = list(feat_dim.keys())
        
        voca_size, wemb_dim = wembs_w.size()
        self.word2emb = nn.Sequential(nn.Embedding.from_pretrained(wembs_w),
                                      nn.ReLU())

        self.feat_layers = {}
        if "pos" in feat_dim:
            self.pos2emb = nn.Sequential(
                            nn.Embedding(feat_dim["pos"], feat_dim["pos"]),
                            nn.Dropout(0) if linear_wemb else nn.ReLU())
            self.feat_layers["pos"] = self.pos2emb
        if "dep" in feat_dim:
            self.dep2emb = nn.Sequential(
                            nn.Embedding(feat_dim["dep"], feat_dim["dep"]),
                            nn.Dropout(0) if linear_femb else nn.ReLU())
            self.feat_layers["dep"] = self.dep2emb
        if "sent_bd_begin" in feat_dim:
            self.sbb2emb = nn.Sequential(
                            nn.Embedding(feat_dim["sent_bd_begin"], 
                                         feat_dim["sent_bd_begin"]),
                            nn.Dropout(0) if linear_femb else nn.ReLU())
            self.feat_layers["sent_bd_begin"] = self.sbb2emb
        if "sent_bd_end" in feat_dim:
            self.sbe2emb = nn.Sequential(
                            nn.Embedding(feat_dim["sent_bd_end"], 
                                         feat_dim["sent_bd_end"]),
                            nn.Dropout(0) if linear_femb else nn.ReLU())
            self.feat_layers["sent_bd_end"] = self.sbe2emb
        feat_dim_total = sum(feat_dim.values())

        self.rnn_hid_dim = rnn_hid_dim
        self.rnn = nn.LSTM(wemb_dim + feat_dim_total, rnn_hid_dim, 
                           bidirectional=bidir)

    def init_rnn_hid(self):
        return (cuda(torch.zeros(self.n_direct, 1, self.rnn_hid_dim)),
                cuda(torch.zeros(self.n_direct, 1, self.rnn_hid_dim)))

    def forward(self, sentence, feats):
        """
            Args:
                sentence: [word, ...]. (sent_len)
                feats: {feat: [f_w1, f_w2, ...], ...}.
            Returns:
                rnn_outs: (seq_len x 1 x n_direct * hid_dim)
                rnn_hid: ((n_direct x 1 x hid_dim) * 2) for LSTM
                rnn_out_comb: (1 x n_direct*hid_dim)
        """
        wembs = self.word2emb(sentence)  # (seq_len x hid_dim)
        fembs = {}
        for feat, layer in self.feat_layers.items():
            fembs[feat] = layer(feats[feat])  # (seq_len x feat_dim)

        rnn_ins = [wembs] + [fembs[feat] for feat in self.feat_names]
        rnn_in = torch.cat(tuple(rnn_ins), 1).unsqueeze(1)
                    # (seq_len x 1 x wemb/femb_dim)

        rnn_outs, rnn_hid = self.rnn(rnn_in, self.init_rnn_hid())

        # Combine the last layer of 1st hid and the 1st layer of last hid
        if self.n_direct == 1:
            rnn_out_comb = rnn_hid[0].view(1, -1)
        else:
            rnn_out_first = rnn_outs[0].view(-1, self.rnn_hid_dim)[-1]
            rnn_out_last = rnn_outs[-1].view(-1, self.rnn_hid_dim)[0]
            rnn_out_comb = torch.cat((rnn_out_first, rnn_out_last), 0)\
                                .view(1, -1)

        return rnn_outs, rnn_hid, rnn_out_comb

class CopyDecoder(nn.Module):
    def __init__(self, wembs_w, dec_hid_dim, enc_hid_dim, copy_weight,
                 copy_mech="bilinear"):
        """
            Args:
                wembs_w: Pretrained word embedding weights.
                dec_hid_dim: Decoder hidden dimensionality.
                enc_hid_dim: Encoder hidden dimensionality.
                copy_weight: Weight for copy vs. normal decoding. 
                             If "auto", copy_weight is dynamically calculated.
                             If "none", words_probs_nm and words_probs_cp are
                             combined before softmax.
                copy_mech: How to compute copy attention in 
                           [bilinear, ff, attn_nm].
        """
        super(CopyDecoder, self).__init__()

        # RNN input
        self.voca_size, wemb_dim = wembs_w.size()
        self.word2emb = nn.Sequential(
                            nn.Embedding.from_pretrained(wembs_w),
                            nn.ReLU()) # ReLU turned out important
        rnn_in_dim = wemb_dim + enc_hid_dim

        # RNN
        self.dec_hid_dim = dec_hid_dim
        self.rnn = nn.LSTM(rnn_in_dim, dec_hid_dim)
        self.rnn2word = nn.Sequential(
                         nn.Linear(enc_hid_dim + dec_hid_dim, self.voca_size))
                         #nn.Softmax(dim=1))

        # Copy mechanism
        self.copy_mech = copy_mech
        if copy_mech == "bilinear":
            self.copyattn = nn.Bilinear(enc_hid_dim, dec_hid_dim, 1)
        elif copy_mech == "ff":
            self.copyattn = nn.Sequential(
                                nn.Linear(enc_hid_dim + dec_hid_dim, 64),
                                nn.ReLU(),
                                nn.Linear(64, 1))
        else:
            # Copy attn = normal attn
            pass

        self.copy_weight = copy_weight
        if self.copy_weight == "auto":
            self.rnn2copy_w = nn.Sequential(nn.Linear(dec_hid_dim, 1),
                                            nn.Sigmoid())


    def init_dec_hid(self):
        return (cuda(torch.zeros(1, 1, self.dec_hid_dim)),
                cuda(torch.zeros(1, 1, self.dec_hid_dim)))

    def forward(self, word, dec_hid, enc_outs, sentence, attn_cp_prev):
        """
            Args:
                word: [word]. (1)
                dec_hid: Previous hidden state.
                enc_outs: (seq_len x 1 x enc_hid_dim)
                attn_cp_prev: (seq_len x 1)
            Returns:
                word_probs: log(softmax) over voca. (1 x voca_size)
                dec_hid: Hidden state. ((n_direct x 1 x hid_dim) * 2) for LSTM
                enc_outs: (seq_len x 1 x hid_dim)
        """
        enc_outs = enc_outs.squeeze(dim=1) # (seq_len x enc_hid_dim)
        seq_len, enc_hid_dim = enc_outs.size()

        # Encode inputs
        rnn_in = self.word2emb(word).view(1, 1, -1)
        if attn_cp_prev is None:
            attn_cp_enc_outs = zero_tensor(1, enc_hid_dim)
        else:
            attn_cp_enc_outs = (attn_cp_prev * enc_outs).sum(0) 
                                    # (1 x enc_hid_dim)
        rnn_in = torch.cat((rnn_in, attn_cp_enc_outs.view(1, 1, -1)), 2)

        # RNN 
        dec_out, dec_hid = self.rnn(rnn_in, dec_hid)  
                # dec_out: (1 x 1 x n_direct*hid_dim)
                # dec_hid: ((n_direct x 1 x hid_dim) * 2) for LSTM
        dec_hid_ = dec_hid[0].view(1, -1)  # (1 x hid_dim)

        # Normal attention
        attn_nm_raw = enc_outs.mm(dec_hid_.transpose(0, 1)) / np.sqrt(seq_len)
                        # (seq_len x 1)
        attn_nm = F.softmax(attn_nm_raw, 0)
        attn_enc_outs = attn_nm * enc_outs  # (seq_len x enc_hid_dim)
        context = attn_enc_outs.sum(0).unsqueeze(0) # (1 x hid_dim)
        word_probs_nm_raw = self.rnn2word(torch.cat((context, dec_out[0]), 1))
                                # (1 x voca_size)
        word_probs_nm = F.softmax(word_probs_nm_raw, dim=1) # (1 x voca_size)

        # Copy attention
        if self.copy_mech == "bilinear":
            attn_cp_raw = self.copyattn(enc_outs, dec_hid_.expand(seq_len, -1))
                            # (seq_len x 1)
            attn_cp = F.softmax(attn_cp_raw, 0)
        elif self.copy_mech == "ff":
            attn_in = torch.cat((enc_outs, dec_hid_.expand(seq_len, -1)), 1)
                            # (seq_len x (enc_hid_dim + dec_hid_dim))
            attn_cp_raw = self.copyattn(attn_in) # (seq_len x 1)
            attn_cp = F.softmax(attn_cp_raw, 0)
        else:
            attn_cp_raw = attn_nm_raw
            attn_cp = F.softmax(attn_cp_raw, 0)

        if self.copy_weight != "none":
            word_probs_cp = self.copy_attn_voca(attn_cp, sentence) # (1 x voca_size)
            copy_weight = self.rnn2copy_w(dec_hid_) if self.copy_weight == "auto" \
                                                    else self.copy_weight
            word_probs = copy_weight * word_probs_cp + \
                            (1 - copy_weight) * word_probs_nm # (1 x voca_size)
        else:
            word_probs_cp_raw_exp = self.copy_attn_voca(
                                                attn_cp_raw.tanh().exp(), 
                                                sentence) # (1 x voca_size)
            word_probs_raw_exp = word_probs_nm_raw.exp() + \
                                  word_probs_cp_raw_exp
            word_probs = word_probs_raw_exp / word_probs_raw_exp.sum()


        return word_probs, dec_hid, attn_cp

    def copy_attn_voca(self, attn_cp, sentence):
        """Applies attn_cp to the voca.
            Args:
                attn_cp: (seq_len x 1)
                sentence: [w1, w2, ...] (seq_len)
            Returns:
                res: Probability over voca. (1 x voca_size)
        """
        res = zero_tensor(self.voca_size)
        for w, attn in zip(sentence, attn_cp):
            res[w] += attn[0]
        return res.view(1, -1)



class AttnDecoder(nn.Module):
    def __init__(self, wembs_w, rnn_hid_dim):
        """
            Args:
                wembs_w: Pretrained word embedding weights.
                rnn_hid_dim: RNN hidden dimensionality.
        """
        super(AttnDecoder, self).__init__()

        voca_size, wemb_dim = wembs_w.size()
        self.word2emb = nn.Embedding.from_pretrained(wembs_w)
        self.rnn_hid_dim = rnn_hid_dim
        self.rnn = nn.LSTM(wemb_dim, rnn_hid_dim)

        self.rnn2word = nn.Sequential(
                            nn.Linear(rnn_hid_dim * 2, voca_size),
                            nn.Softmax(dim=1))

    def init_rnn_hid(self):
        return (cuda(torch.zeros(1, 1, self.rnn_hid_dim)),
                cuda(torch.zeros(1, 1, self.rnn_hid_dim)))

    def forward(self, word, rnn_hid, enc_outs):
        """
            Args:
                word: [word]. (1)
                rnn_hid: Previous hidden state.
            Returns:
                word_probs: log(softmax) over voca. (1 x voca_size)
                rnn_hid: Hidden state. ((n_direct x 1 x hid_dim) * 2) for LSTM
                enc_outs: (seq_len x 1 x hid_dim)
        """
        wemb = self.word2emb(word).view(1, 1, -1)
        wemb = F.relu(wemb)
        rnn_out, rnn_hid = self.rnn(wemb, rnn_hid)  
                # rnn_out: (1 x 1 x n_direct*hid_dim)
                # rnn_hid: ((n_direct x 1 x hid_dim) * 2) for LSTM

        # Normal attention
        enc_outs = enc_outs.squeeze(dim=1) # (seq_len x (n_direct * hid_dim))
        seq_len = enc_outs.size()[0]
        attn_w = enc_outs.mm(rnn_hid[0].view(-1, 1)) / np.sqrt(seq_len)
                        # (seq_len x 1)
        attn_w = F.softmax(attn_w, 0)
        attn_enc_outs = attn_w * enc_outs  # (seq_len x (n_direct * hid_dim))
        context = attn_enc_outs.sum(0).unsqueeze(0) # (1 x (n_direct * hid_dim))

        word_probs = self.rnn2word(torch.cat((context, rnn_out[0]), 1))
                        # (1 x voca_size)

        return word_probs, rnn_hid



class BasicDecoder(nn.Module):
    def __init__(self, wembs_w, rnn_hid_dim):
        """
            Args:
                wembs_w: Pretrained word embedding weights.
                rnn_hid_dim: RNN hidden dimensionality.
                n_labels: Num of labels.
        """
        super(BasicDecoder, self).__init__()
        
        voca_size, wemb_dim = wembs_w.size()
        self.word2emb = nn.Embedding.from_pretrained(wembs_w)

        self.rnn_hid_dim = rnn_hid_dim
        self.rnn = nn.LSTM(wemb_dim, rnn_hid_dim)

        self.rnn2word = nn.Sequential(
                            nn.Linear(rnn_hid_dim, voca_size),
                            nn.Softmax(dim=1))

    def init_rnn_hid(self):
        n_direct = 1
        return (cuda(torch.zeros(n_direct, 1, self.rnn_hid_dim)),
                cuda(torch.zeros(n_direct, 1, self.rnn_hid_dim)))

    def forward(self, word, rnn_hid, enc_outs=None):
        """
            Args:
                word: [word]. (1)
                rnn_hid: Previous hidden state.
            Returns:
                word_probs: log(softmax) over voca. (1 x voca_size)
                rnn_hid: Hidden state. ((n_direct x 1 x hid_dim) * 2) for LSTM
                enc_outs: Not used, just for func arguments consistency.
        """
        wemb = self.word2emb(word).view(1, 1, -1)
        wemb = F.relu(wemb)
        rnn_out, rnn_hid = self.rnn(wemb, rnn_hid)  
                # rnn_out: (1 x 1 x n_direct*hid_dim)
                # rnn_hid: ((n_direct x 1 x hid_dim) * 2) for LSTM

        word_probs = self.rnn2word(rnn_out[0])  # (1 x voca_size)
        return word_probs, rnn_hid



class BeamSearchNode(object):
    def __init__(self, dec_hid, prev_node, prev_word, log_p, length, attn_cp):
        self.dec_hid = dec_hid
        self.prev_node = prev_node
        self.prev_word = prev_word
        self.log_p = log_p
        self.length = length 
        self.attn_cp = attn_cp

    def __lt__(self, node):
        return self.length > node.length

    def avg_log_p(self):
        return 0 if self.length == 1 else self.log_p / float(self.length - 1)



class QuestionTransformer(object):
    def __init__(self):
        self.fword_patterns = self.load_patterns()
        self.well_formed_fwords = [
                "why", "where", "when",
                "how", "what", "which", "who",
                "have", "has", "is", "are", "was", "were",
                "can", "will", "should", "would", "could",
                "does", "do", "did"
        ]

    def match_form(self, words, poses, nps):
        """
            Args:
                words: [word, ...]
        """
        form = []
        for word, pos, np in zip(words, poses, nps):
            if word in ["n't", "not", "never"]: pos = "NOT"

            if np in ["NP", "NPE"]:
                form.append("{}//{}##{} ".format(word, pos, np))
            else:
                form.append("{}//{} ".format(word, pos))
        res = "".join(form)
        return res


    def transform(self, words, poses, nps):
        """
            Returns:
                convtext: Transformed text.
                pattern_name:
        """
        match = self.match_form(words, poses, nps)

        fword = words[0]
        if fword in self.fword_patterns:
            patterns = self.fword_patterns[fword]
        else:
            patterns = []
            # subordinate
            for subwords in list_split(lambda e: e == ",", words):
                if subwords[0] in self.fword_patterns:
                    patterns = self.fword_patterns[subwords[0]]
                    break

            #if len(patterns) == 0:
            #    for word in words:
            #        if word in self.fword_patterns:
            #            patterns = self.fword_patterns[word]
            #            break

        convtext = None
        pattern_name = None
        for name, pat, rep in patterns:
            if fword in self.fword_patterns:
                m = re.search("^{}".format(pat), match) # Beginning of sent
            else:
                m = re.search("(?<= ){}".format(pat), match) 
            if not m: continue

            convtext = []
            for r in rep:
                if type(r) is str: convtext.append(r)
                elif type(r) is int: convtext.append(
                                            self.stem(m.group(r), True))
                else: convtext.append(self.stem(m.group(r[0]), r[1]))
            prefix = self.stem(match[:m.start()])
            convtext = prefix +  "".join(convtext).strip()
            pattern_name = name
            break
        
        if not convtext:
            convtext = " ".join(words)

        return convtext, pattern_name


    def stem(self, text, space=True):
        if space:
            return re.sub("//\\S+", "", text)
        else:
            return re.sub("//\\S+", "", text).strip()

    def load_patterns(self):
        fwords_patterns = [
            (["why"], [
            ("why", "(why//\\S+ )(would//MD )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$",
                [3, 2, "n't ", 5]), # (1, .2824)
            ("why", "(why//\\S+ )((?!would)\\S+//MD )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$",
                [3, 2, "not ", 5]), # (3, .3765) "trust" is tagged as noun
            ("why", "(why//\\S+ )(\\S+//MD )(n't//\\S+ )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$",
                [4, 2, 6]), # (2,
            ("why", "(why//\\S+ )(ca//\\S+ n't//\\S+ )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$",
                [3, "can ", 5]), # (1, .1825)
            ("why", "(why//\\S+ )((?!ca)\\S+//MD )(\\S+//NOT )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$",
                [4, 2, 6]), # (1, .2917)
            ("why", "(why//\\S+ )((do)//\\S+ )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$",
                [4, 6]), # (2, .1573)
            ("why", "(why//\\S+ )((does|did)//\\S+ )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$",
                [4, 2, 6]), # (1, .0063)
            ("why", "(why//\\S+ )(is//\\S+ )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$",
                #[3, 2, 5]) # (6, .1502) better than below but not as informative
                [3, 2, 5, "because xxx"]), # (6, .1234)
            ("why", "(why//\\S+ )((are|were|was)//\\S+ )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$",
                [4, 2, 6]), # (6, .1234)
            ("why", "why//\\S+ ((is|are|am)//\\S+ )\\S+//NOT ((\\S+##NP )*\\S+##NPE )(\\S+//JJ .*?)([?]//\\S+ )*$",
                [3, 1, 5]), # (
            ("why", "why//\\S+ ((is|are|am)//\\S+ )\\S+//NOT ((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* .*?)([?]//\\S+ )*$",
                [3, "should be ", 5]), # (
            ("why", "(why//\\S+ not//\\S+ )([?]//\\S+ )*$",
                [1]), # (4, -.0017)
            ("why", "why//\\S+ not//\\S+ (\\S+//VB\\S* )(.*?)([?]//\\S+ )*$",
                ["should ", 1, 2]), # (2, .0598)
            ]),
            (["where", "when"], [
            ("where1", "(where//\\S+ )((do|did|does)//\\S+ |\\S+//MD )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$",
                [4, 2, 6, "at xxx"]), # (2, .0396)
            ("when1", "(when//\\S+ )((did|has)//\\S+ )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$",
                [4, 2, "not ", 6]), # (2, .0556)
            ]),
            (["how"], [
            ("how1", "(how//\\S+ )(can//MD )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$",
                [3, 2, "not ", 5, 6]), # (6, .0412)
            ("how2", "(how//\\S+ )((?!can)\\S+//MD )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$",
                [3, 2, 5, 6, "by xxx"]), # (5, .0828)
            ("how3", "(how//\\S+ )(do//VB\\S* )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$",
                [3, 5, 6, "by xxx"]), # (11, .3322)
            ("how4", "(how//\\S+ )(does//VB\\S* )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$",
                #[3, (5, False), "s ", 6, "by xxx"]), # (2, .0145)
                [3, 5, 6, "by xxx"]), ############
            ("how5", "(how//\\S+ )(did//VB\\S* )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$",
                [1, 2, 3, 5]), # (1, 0)
            ("how6", "(how//\\S+ )(\\S+//MD |(do|does|did)//\\S+ )((\\S+##NP )*\\S+##NPE )(\\S+/NOT )(.*?)([?]//\\S+ )*$",
                [4, "should ", 7]), # (1, .0914)
            ("how7", "(how//\\S+ )(are//VB\\S* )((\\S+##NP )*\\S+##NPE )(going//VB\\S* )(to//\\S+ )(.*?)([?]//\\S+ )*$",
                [3, "need to ", 7]), # (6, .2477)
            ("how8", "(how//\\S+ )(are//VB\\S* )((\\S+##NP )*\\S+##NPE )(supposed//VB\\S* )(to//\\S+ )(.*?)([?]//\\S+ )*$",
                [3, "ca n't ", 7]), # (2, .1498)
            ("how9", "(how//\\S+ )((?!do|does|did|are)\\S+//VB\\S* )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$",
                [1, 2, 3, 5]), # (3, 0)
            ("how10", "(how//\\S+ )((am|are|is)//\\S+ )((\\S+##NP )*\\S+##NPE )(\\S+/NOT )(.*?)([?]//\\S+ )*$",
                [4, "should be ", 7]), # (1, .7373)
            ("how11", "(how//\\S+ )(much//\\S* )(.*?)([?]//\\S+ )*$",
                ["xxx ", 3]), # (4, .2751)
            ("how12", "(how//\\S+ )(about//\\S* )(.*?)([?]//\\S+ )*$",
                [1, 2, 3]), # (3, .0821)
            ("how13", "(how//\\S+ )(on//\\S+ earth//\\S+ )(\\S+//(?!MD|VB)\\S+ )*(\\S+//(MD|VB\\S*) )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$",
                [6, 4, "n't ", 8]), # (1, .5920)
            ("how14", "(how//\\S+ )(else//\\S+ )(\\S+//(?!MD|VB)\\S+ )*(\\S+//(MD|VB\\S*) )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$",
                [6, 4, "n't ", 8]), # (1, .2566)
            ("how15", "(how//\\S+ )(?!on//\\S+ earth//\\S+ |else//\\S+ )(\\S+//(?!MD|VB)\\S+ )+(\\S+//(MD|VB\\S*) )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$",
                [5, 3, 7, 8]), # (4, .2719)
            ("how16", "(how//\\S+ )(?!on//\\S+ earth//\\S+ |else//\\S+ )(\\S+//(?!MD|VB)\\S+ )+(\\S+//(MD|VB\\S*) )((\\S+##NP )*\\S+##NPE )(\\S+//(?!VB)\\S+ )*([?]//\\S+ )*$",
                [5, 3, "xxx"]), # (1, .7636)
            ]),
            (["what"], [
            ("what1", "(what//\\S+ )(\\S+//MD )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$",
                [3, 2, 5, "xxx ", 6]), # (5, .0517)
            ("what2", "(what//\\S+ )(\\S+//MD )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$", 
                [1, 2, 3, 4]),  # (2, 0), #["xxx ", 2, 3, 4]),  # (2, -.0615)
            ("what3", "(what//\\S+ )(did//\\S+ )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$",
                [3, "did ", 5, "xxx ", 6]), # (4, .1092)
            ("what4", "(what//\\S+ )(does//\\S+ )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$",
                #[3, (5, False), "s xxx ", 6]),  # (5, .0332)
                [3, 5, "xxx ", 6]),  ##########
            ("what5", "(what//\\S+ )(do//\\S+ )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$", # combine do+does?
                #[3, (5, False), "s xxx ", 6]),  # (7, .1223),
                [3, 5, "xxx ", 6]),  ##########
            ("what6", "(what//\\S+ )(am//\\S+ )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$", 
                [3, 2, 5, "xxx ", 6]),  # (2, .3949)
            ("what7", "(what//\\S+ )(was//\\S+ )(.*?)([?]//\\S+ )*$", 
                [3, 2, "xxx"]),  # (1, .6464)
            ("what8", "(what//\\S+ )(is//\\S+ )(.*?)([?]//\\S+ )*$", 
                [3, 2, "xxx"]),  # (4, .1274)
            ("what9", "(what//\\S+ )('s//\\S+ )(.*?)([?]//\\S+ )*$", 
                [3, "is xxx"]),  # (6, .0711)
            ("what10", "(what//\\S+ )(are//\\S+ )(.*?)([?]//\\S+ )*$", 
                [3, 2, "xxx"]),  # (3, .0459)
            ("what11", "(what//\\S+ )((?!did|does|do|am|was|is|'s|are)\\S+//VB\\S* )(.*?)([?]//\\S+ )*$", 
                ["xxx ", 2, 3]),  # (3, 0.0521)
            ("what12", "(what//\\S+ )((\\S+//(?!VB|NOT|MD)\\S+ )+)(\\S+//MD\\S* )(.*?)([?]//\\S+ )*$", 
                [1, 2, 4, 5]),  # (2, 0)
            ("what13", "(what//\\S+ )((\\S+//(?!VB|NOT|MD)\\S+ )+)(\\S+//VB\\S* )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$", 
                [1, 2, 4, 5, 7]),  # (4, 0)
            ]),

            (["which"], [
            ("which1", "(which//\\S+ )(\\S+//(?!VB)\\S+ )+([?]//\\S+ )*$",
                [2, "xxx "]), # (1, .0991)
            ("which2", "(which//\\S+ )(\\S+//(?!VB)\\S+ )+(\\S+//VB\\S* )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$",
                [4, 3, 6, 2, "xxx"]), # (2, -0.1656)
            ]),
            (["who"], [
            ("who1", "(who//\\S+ )([?]//\\S+ )*$",
                [1]), # (2, 0)
            ("who2", "(who//\\S+ )(\\S+//VB\\S* )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$",
                [3, 2, 5, 6, "xxx"]), # (2, .0046)
            ("who3", "(who//\\S+ )('s//VB\\S* )((?!\\S+//VB).*?)([?]//\\S+ )*$",
                [3, "is xxx"]), # (3, .0688)
            ("who4", "(who//\\S+ )('s//VB\\S* )(\\S+//VB\\S* .*?)([?]//\\S+ )*$",
                ["xxx is ", 3]), # (1, .0358)
            ("who5", "(who//\\S+ )((?!'s)\\S+ )(.*?)([?]//\\S+ )*$",
                ["xxx ", 2, 3]), # (6, .0198)
            ]),

            (["have", "has"], [
            ("have1", "((have)//\\S+ )(you//\\S+ )(\\S+//NOT )*(.*?)([?]//\\S+ )*$", 
                [3, 1, "not ", 5]),  # (5, .0522)
            ("have2", "((have|has)//\\S+ )((?!you)(\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$", 
                [3, 1, 5]),  # (13, .0919)
            ]),
            (["is"], [
            ("is1", "((is)//\\S+ )((\\S+##NP )*\\S+##NPE |that//IN )(\\S+##NP.*?)([?]//\\S+ )*$", 
                [3, 1, 5]), # (20, .0466)
            ("is1", "((is)//\\S+ )((\\S+##NP )*\\S+##NPE |that//IN )(?!\\S+##NP)(.*?)([?]//\\S+ )*$", 
                [3, 1, "/ ", 1, "not ", 5]),  # (11, .0443)
            ("is1", "((is)//\\S+ )(\\S+//NOT )(that//IN |it//\\S+ |(\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$", 
                [4, 1, 6]), # (4, .0051)
            ]),
            (["are"], [
            ("are1", "((are)//\\S+ )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$", 
                [3, 1, "not ", 5]), # (14, .0405)
            ("are2", "((are)//\\S+ )(\\S+//NOT )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$", 
                [4, 1, 6]), # (3, .0097)
            ]),
            (["was", "were"], [
            ("be1", "((was|were)//\\S+ )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$", 
                [3, 1, 5]), # (8, .0790)
            ("be2", "((was|were)//\\S+ )(\\S+//NOT )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$", 
                [4, 1, 6]), # (1, .0253)
            ]),
            (["can", "will", "should", "would", "could"], [
            ("md1", "(can//MD )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$",
                [2, 1, 4, 5]), # (13, -0.0588)
            ("md2", "((?!can)\\S+//MD )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$",
                [2, 1, "/ ", 1, "not ", 4, 5]), # (20, 0.1942)
            ("md3", "(\\S+//MD )(n't//\\S+ )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$",
                [3, 1, 5, 6]), # (1, 0.4757)
            ("md4", "(\\S+//MD )(\\S+//(?!NOT|VB)\\S+ )(\\S+//(?!NOT|VB)\\S+ )*([?]//\\S+ )*$",
                [2, 1, "/ ", 1, "not ", 3]), # (2, .0191)
            ]),
            (["does"], [
            ("does1", "(does//\\S+ )(n't//\\S+ )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$", 
                #[3, (5, False), "s ", 6]),
                [3, 5, 6]), ##########
            ("does2", "(does//\\S+ )((\\S+##NP )*\\S+##NPE )(not//\\S+ )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$", 
                [2, 5, 6]),
            ("does5", "(does//\\S+ )(\\S+//\\S+ )(not//\\S+ )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$", 
                [2, "does not ", 4, 5]),  # specific
            ("does3", "(does//\\S+ )(n't//\\S+ )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$", 
                 [1, 2, 3, 4]),
            ("does4", "(does//\\S+ )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$", 
                 [2, "does not ", 4, 5]),
            ]),
            (["do"], [
            ("do1", "(do//\\S+ )(n't//\\S+ )((\\S+##NP )*\\S+##NPE )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$", 
                [3, 5, 6]),
            ("do2", "(do//\\S+ )((\\S+##NP )*\\S+##NPE )(\\S+//NOT )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$", 
                [2, 5, 6]),
            ("do3", "(do//\\S+ )(n't//\\S+ )(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$", 
                 [1, 2, 3, 4]),
            ("do4", "(do//\\S+ )((\\S+//(?!VB|NOT)\\S+ )+)(\\S+//VB\\S* )(.*?)([?]//\\S+ )*$", 
                 [2, "does / does not ", 4, 5]),
            ("do5", "(do//\\S+ )((\\S+//(?!VB|NOT)\\S+ )*)(\\S+//(?!VB)\\S+ )([?]//\\S+ )*$", 
                 [2, "do / do not ", 4]),  # specific
            ]),
            (["did"], [
            ("did1", "(did//\\S+ )((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$", 
                [2, 1, "not ", 4]),  # (13, .0442)
            ("did2", "(did//\\S+ )n't//\\S+ ((\\S+##NP )*\\S+##NPE )(.*?)([?]//\\S+ )*$", 
                [2, 1, "n't ", 4]),  # (3, .0058)
            ]),
        ]

        fword_patterns = {}
        for fwords, patterns in fwords_patterns:
            for fword in fwords:
                fword_patterns[fword] = patterns
        return fword_patterns

    def is_well_formed(self, fword):
        return fword in self.well_formed_fwords




