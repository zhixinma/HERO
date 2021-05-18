import json
from tqdm import tqdm
from utils.misc import Struct
from utils.basic_utils import load_json
from os.path import exists
from model.vcmr import HeroForVcmr
import torch
from utils.const import VFEAT_DIM
from apex import amp

from load_data import get_video_ids, load_video_sub_dataset
from data import VcmrFullEvalDataset, vcmr_full_eval_collate, PrefetchLoader, QueryTokLmdb
from torch.utils.data import DataLoader

import networkx as nx
import numpy as np

from ivcml_graph import shortest_path_edges
from ivcml_util import build_sparse_adjacent_matrix, indicator_vec
from ivcml_util import flatten
from ivcml_util import show_type_tree


def load_tvr_subtitle():
    data_folder = "/raw/"
    subtitle = "tvqa_preprocessed_subtitles.jsonl"
    vid_subs = {}
    with open(data_folder+subtitle, "r") as f:
        for line in f:  # 21793 clips
            clip = json.loads(line)
            vid_subs[clip["vid_name"]] = clip["sub"]  # {"start":, "end":, "text":}
    return vid_subs


def load_v_m_map():
    folder_p = "/raw/"
    clip_to_moments = {}
    for part in ["train", "val", "test_public"]:
        des_val = "tvr_%s_release.jsonl" % part
        with open(folder_p + des_val, "r") as f:
            for line in f:
                moment = json.loads(line)
                if "vid_name" not in moment:
                    continue
                vid_name = moment["vid_name"]
                if vid_name not in clip_to_moments:
                    clip_to_moments[vid_name] = []
                clip_to_moments[vid_name].append([moment["desc_id"], moment["ts"]])
    return clip_to_moments


def load_tvr_moment(part="train"):
    data_folder = "/raw/"
    des_val = "tvr_%s_release.jsonl" % part
    moments_len = []
    moments_ratio = []
    c, len_c, r_c = 0, 0, 0
    with open(data_folder+des_val, "r") as f:
        for line in f:
            moment = json.loads(line)
            duration_clip = moment["duration"]
            d_moment = moment["ts"][1] - moment["ts"][0]
            ratio = d_moment / duration_clip
            ratio = int(ratio * 100 + 0.5) * 1.0 / 100  # round

            assert ratio <= 1, (round(d_moment, 2), round(duration_clip, 2), ratio)
            c += 1
            if d_moment < 100:
                len_c += 1
                moments_len.append(d_moment)
            if ratio <= 1:
                r_c += 1
                moments_ratio.append(ratio)

    print(c, len_c, r_c)


def load_hero_pred(opts, task):
    # TODO: 1. top-k setting may be a problem
    assert task in {"vr", "vcmr_base_on_vr", "vcmr"}
    result_dir = f'{opts.output_dir}/results_{opts.split}'
    pred_dir = f'{result_dir}/results_{opts.checkpoint}_{opts.split}_{task}.json'
    with open(pred_dir, "r") as f:
        pred = json.load(f)
    return pred


def load_subtitle_range(val_loader, opts):
    # subtitle range
    video_db = val_loader.dataset.video_db
    vid2idx = val_loader.dataset.vid2idx
    vid_sub_range = {}
    for vid in tqdm(vid2idx[opts.split], desc="Loading Subtitle Feature", total=len(vid2idx[opts.split])):
        item = video_db[vid]
        frame_level_input_ids, frame_level_v_feats, frame_level_attn_masks, clip_level_v_feats, clip_level_attn_masks, num_subs, sub_idx2frame_idx = item
        vid_sub_range[vid] = {}
        for sub_idx, frame_range in sub_idx2frame_idx:
            vid_sub_range[vid][sub_idx] = frame_range
    return vid_sub_range


def load_video2duration(split="train"):
    """ The map from video id to the video duration
    """
    folder_p = "/raw/"
    f_name = "tvr_video2dur_idx.json"
    video_duration = {}
    with open(folder_p+f_name, "r") as f:
        for line in f:
            clips = json.loads(line)
            if split == "all":
                for tag in clips:
                    partial_clips = clips[tag]
                    for clip_id in partial_clips:
                        video_duration[clip_id] = partial_clips[clip_id][0]
            else:
                for clip_id in clips[split]:
                    video_duration[clip_id] = clips[split][clip_id][0]
    return video_duration


def build_vid_to_frame_num(video_db):
    """ The map from video id to frame number of the video
    """
    v_id_to_frame_number = {}
    for _id in video_db.img_db.name2nframe:
        print(_id)
        v_id_to_frame_number[_id] = video_db[_id][3].shape[0]
        n = min(video_db.img_db.max_clip_len, video_db[_id][3].shape[0])
        n_1 = min(video_db.img_db.max_clip_len, video_db.img_db.name2nframe[_id])
        assert n_1 == n, f"Inconsistent {_id}: {n_1} and {n}; min({video_db.img_db.name2nframe[_id]}, {video_db.img_db.max_clip_len}) and min({video_db[_id][3].shape[0]}, {video_db.img_db.max_clip_len})"
    return v_id_to_frame_number


def load_model(opts, device):
    hps_file = f'{opts.output_dir}/log/hps.json'
    model_opts = Struct(load_json(hps_file))
    model_config = f'{opts.output_dir}/log/model_config.json'
    # Prepare model
    if exists(opts.checkpoint):
        ckpt_file = opts.checkpoint
    else:
        ckpt_file = f'{opts.output_dir}/ckpt/model_step_{opts.checkpoint}.pt'
    checkpoint = torch.load(ckpt_file)
    img_pos_embed_weight_key = ("v_encoder.f_encoder.img_embeddings.position_embeddings.weight")
    assert img_pos_embed_weight_key in checkpoint
    max_frm_seq_len = len(checkpoint[img_pos_embed_weight_key])
    model = HeroForVcmr.from_pretrained(
        model_config,
        state_dict=checkpoint,
        vfeat_dim=VFEAT_DIM,
        max_frm_seq_len=max_frm_seq_len,
        lw_neg_ctx=model_opts.lw_neg_ctx,
        lw_neg_q=model_opts.lw_neg_q, lw_st_ed=0,
        ranking_loss_type=model_opts.ranking_loss_type,
        use_hard_negative=False,
        hard_pool_size=model_opts.hard_pool_size,
        margin=model_opts.margin,
        use_all_neg=model_opts.use_all_neg,
        drop_svmr_prob=model_opts.drop_svmr_prob)
    model.to(device)
    if opts.fp16:
        model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')
    return model, model_opts


def build_dataloader(opts):
    # Load ground truth, query db and video db
    hps_file = f'{opts.output_dir}/log/hps.json'
    model_opts = Struct(load_json(hps_file))
    video_ids = get_video_ids(opts.query_txt_db)
    video_db = load_video_sub_dataset(opts.vfeat_db, opts.sub_txt_db, model_opts.vfeat_interval, model_opts)
    assert opts.split in opts.query_txt_db, (opts.split, opts.query_txt_db)
    q_txt_db = QueryTokLmdb(opts.query_txt_db, -1)

    eval_dataset = VcmrFullEvalDataset(video_ids, video_db, q_txt_db, distributed=model_opts.distributed_eval)
    eval_dataloader = DataLoader(eval_dataset, batch_size=opts.batch_size, num_workers=opts.n_workers, pin_memory=opts.pin_mem, collate_fn=vcmr_full_eval_collate)
    eval_dataloader = PrefetchLoader(eval_dataloader)
    return eval_dataloader


def ivcml_preprocessing(g: nx.Graph, node_src, node_tar, feature_embedding):
    def aggregate(_adj, _state_vec, _fea_emb, alpha, step_num):
        """
        Aggregation function for single state
        :param _adj: Adjacency matrix [N, N]
        :param _state_vec: State vector [N, ]
        :param _fea_emb: Feature embedding [N, d]
        :param alpha: decay rate
        :param step_num: number of steps to aggregate
        :return: Aggregated feature of _s given _a
        """
        _fea = torch.einsum('nd,n->d', _fea_emb, _state_vec) * alpha
        cover = None
        for j in range(step_num):
            neis_leq_j_step = torch.einsum('io,i->o', _adj, _state_vec)
            cover = ((_state_vec > 0) | (neis_leq_j_step > 0)).to(dtype=dtype)
            new_explore = cover - _state_vec
            _state_vec = cover
            num_new_nodes = new_explore.sum().item()
            if num_new_nodes:
                weight_sum = torch.einsum('nd,n->d', _fea_emb, new_explore) * (alpha ** (j + 2))
                _fea += weight_sum
        _fea /= cover.sum().item()
        return _fea

    def get_adj_nei(_g: nx.Graph, _center, _nei):
        edges = [_e for _e in g.edges if not (_center in _e and _nei not in _e)]
        return build_sparse_adjacent_matrix(edges, node_num, undirectional=True)

    states, targets = [], []
    nei_vectors, nei_adj_indices = [], []
    sp = shortest_path_edges(g, node_src, node_tar)
    node_num = g.number_of_nodes()
    device = feature_embedding.device
    dtype = feature_embedding.dtype

    adj_mat = build_sparse_adjacent_matrix(list(g.edges), node_num, device=device, dtype=dtype, undirectional=True)  # directed adjacency matrix
    for _st, _ed in sp:
        # a training sample
        state, ground_truth = _st, _ed
        # vec_state = indicator_vec(state, node_num, device=device, dtype=dtype)
        # fea_state = aggregate(adj_mat, vec_state, feature_embedding, alpha=0.5, step_num=10)

        # build input sample
        states.append(state)
        nei_tmp, nei_adj_coo_tmp, nei_adj_val_tmp = [], [], []
        for i, nei in enumerate(g.adj[state]):
            if nei == ground_truth:
                # build input sample
                targets.append(i)
            adj_mat_nei = get_adj_nei(g, state, nei)
            # vec_nei = indicator_vec(nei, node_num, device=device, dtype=dtype)
            # fea_nei = aggregate(adj_mat_nei, vec_nei, feature_embedding, alpha=0.5, step_num=10)

            # build input sample
            adj_mat_nei = adj_mat_nei.tril(-1)  # remove the duplicated edges
            coo = adj_mat_nei.nonzero(as_tuple=True)
            coo = torch.stack(coo, dim=0)
            nei_tmp.append(nei)
            nei_adj_coo_tmp.append(coo)
        nei_vectors.append(nei_tmp)
        nei_adj_indices.append(nei_adj_coo_tmp)

    return states, targets, nei_vectors, nei_adj_indices


if __name__ == "__main__":
    num_class = 5
    st, ed = 1, 4
    decay_rate = 0.5
    n_step = 6
    emb = torch.eye(num_class)

    v = [_ for _ in range(num_class)]
    e = [(0, 1), (1, 2), (2, 3), (3, 4)]
    g = nx.Graph()
    g.add_nodes_from(v)
    g.add_edges_from(e)

    x = ivcml_preprocessing(g, 1, 4, emb)
    show_type_tree(x)
