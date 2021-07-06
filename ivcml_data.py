import lmdb
import json
import io
import numpy as np
import torch
import msgpack
import msgpack_numpy as m
from transformers import BertTokenizer
from ivcml_util import padding
m.patch()
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_CAP_LEN = 32


def _fp16_to_fp32(feat_dict):
    out = {k: arr.astype(np.float32) if arr.dtype == np.float16 else arr for k, arr in feat_dict.items()}
    return out


class HEROUnitFeaLMDB(object):
    def __init__(self, data_dir, fea_type, split, tag, encode_method="ndarray", readonly=True):
        f_path = "/%s/hero_unit_fea_lmdb/%s_%s_%s_lmdb" % (data_dir.strip("/"), split, fea_type, tag)
        create = not readonly
        self.env = lmdb.open(f_path, readonly=readonly, create=create, readahead=False, map_size=int(5e10))
        self.txn = self.env.begin()
        self.encode_method = encode_method
        print("LMDB file created: %s" % f_path)

    def put(self, keys, values):
        with self.env.begin(write=True) as txn:
            for k, v in zip(keys, values):
                k_enc, v_enc = self._encode_(k, v)
                txn.put(key=k_enc, value=v_enc, overwrite=True)
            # commit automatically

    def get(self, v_id):
        v_enc = self.txn.get(v_id.encode('utf-8'))
        if v_enc is None:
            return None
        v = self._decode_(v_enc)
        return v

    def _encode_(self, k, v):
        k_enc = k.encode()
        if self.encode_method == "ndarray":
            assert isinstance(v, np.ndarray), type(v)
            v_enc = msgpack.packb(v, default=m.encode)

        elif self.encode_method == "json":
            assert isinstance(v, list) or isinstance(v, dict), type(v)
            v_enc = json.dumps(v).encode()

        else:
            assert False, type(v)

        return k_enc, v_enc

    def _decode_(self, v_enc):
        if self.encode_method == "ndarray":
            v = msgpack.unpackb(v_enc, object_hook=m.decode)
            v = torch.from_numpy(v).float()
        elif self.encode_method == "json":
            v = json.loads(v_enc)
        else:
            assert False, v_enc

        return v

    def keys(self):
        with self.env.begin() as txn:
            keys = [key.decode() for key, _ in txn.cursor()]
        return keys

    def __getitem__(self, video_id):
        if isinstance(video_id, list):
            return torch.cat([self.get(v_id) for v_id in video_id], dim=0)
        return self.get(video_id)

    def __del__(self):
        self.env.close()


def text_to_id_bert(text):
    tokens = ["[CLS]"] + bert_tokenizer.tokenize(text) + ["[SEP]"]
    tok_len = min(len(tokens), MAX_CAP_LEN)
    tokens = padding(tokens, MAX_CAP_LEN, "[PAD]")
    text_tok_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    text_tok_ids = torch.tensor(text_tok_ids)
    return text_tok_ids, tok_len
