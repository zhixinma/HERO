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
from ivcml_util import undirectionalize, remove_undirectional_edge
from ivcml_util import l2_norm
from ivcml_util import stack_tenor_list

from ivcml_data import HEROUnitFeaLMDB
from ivcml_data import text_to_id_bert

from ivcml_graph import graph_shortest_path

c = 0


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


def compress_adj_mat(graph: nx.Graph):
    node_num = graph.number_of_nodes()
    adj_mat = build_sparse_adjacent_matrix(list(graph.edges), node_num, undirectional=True)  # directed adjacency matrix
    adj_mat = adj_mat.tril(-1)  # remove the duplicated edges
    coo_adj_zero = (1 - adj_mat.diag(-1)).nonzero().reshape(-1).tolist()  # edges adjacent nodes (aka temporal edges)
    coo_mom = adj_mat.tril(-2).nonzero().transpose(0, 1).tolist()  # remove the edges between adjacent nodes (aka temporal edges)
    return coo_adj_zero, coo_mom


def recover_adj_mat(num_nodes, coo_adj_zero, coo_mom):
    mat = torch.eye(num_nodes)
    mat[[coo_adj_zero, coo_adj_zero]] = 0
    mat[coo_mom] = 1
    mat = undirectionalize(mat)
    return mat


def aggregate(_adj, _src_vec, _alpha, step_num, norm=True):
    """
    Aggregation function for single state
    :param _adj: Adjacency matrix [N, N]
    :param _src_vec: State vector [N, ]
    :param _alpha: decay rate
    :param norm: do L2 norm if True
    :param step_num: number of steps to aggregate
    :return: Aggregated node weight of _s given _a
    """
    _w = _src_vec * _alpha
    dtype = _src_vec.dtype
    for j in range(step_num):
        neis_leq_j_step = torch.einsum('io,i->o', _adj, _src_vec)
        cover = ((_src_vec > 0) | (neis_leq_j_step > 0)).to(dtype=dtype)
        new_explore = cover - _src_vec
        _src_vec = cover
        num_new_nodes = new_explore.sum().item()
        if num_new_nodes:
            _w += new_explore * (_alpha ** (j + 2))
    if norm:
        _w = l2_norm(_w)
    return _w


def aggregate_batch(_a, _s, alpha, steps, norm=True):
    """
    Aggregation function in batch
    :param _a: Adjacency matrix [B, N, N] or [B, A, N, N] where A is dim of neighbors
    :param _s: State vector [B, N, ] or [B, A, N]
    :param steps
    :param alpha
    :param norm:
    :return: Aggregated feature of _s given _a
    """
    batch_size, nei_size = _a.shape[0], _a.shape[1]
    device = _a.device
    # _fea = torch.einsum('band,ban->bad', _fea_emb, _s) * alpha
    _w = _s * alpha
    # cover = torch.ones(batch_size, nei_size, 1, device=device)  # initialize TODO check
    for i in range(steps):
        neis_leq_i_step = torch.einsum('bio,bi->bo', _a, _s)
        cover = ((_s > 0) | (neis_leq_i_step > 0)).to(torch.float)
        new_explore = cover - _s
        _s = cover
        num_new_nodes = new_explore.sum().item()
        if num_new_nodes:
            # weight_sum = torch.einsum('band,ban->bad', _fea_emb, new_explore) * (lpha ** (i + 2))
            # _fea += weight_sum
            _w += new_explore * (alpha ** (i + 2))

    if norm:
        _w = l2_norm(_w)

    # _fea /= cover.sum(dim=-1, keepdim=True)
    return _w


def get_adj_nei(_g: nx.Graph, _center, _nei):
    edges = [_e for _e in _g.edges if not (_center in _e and _nei not in _e)]
    return build_sparse_adjacent_matrix(edges, _g.number_of_nodes(), undirectional=True)


def ivcml_preprocessing(graph: nx.Graph, node_src, node_tar):
    states_idx, targets_bias = [], []
    neis_unit_idx = []
    neis_rewards = []
    sp = shortest_path_edges(graph, node_src, node_tar)

    for _st, _ed in sp:
        state, ground_truth = _st, _ed
        states_idx.append(state)
        nei_tmp = []
        rewards_tmp = []
        actions = [state] + list(graph.adj[state].keys())
        for i, nei in enumerate(actions):
            reward = 0
            if nei == ground_truth:
                targets_bias.append(i)
                reward = 1

            # edge to remove in adjacent matrix of neighbors
            nei_tmp.append(nei)
            rewards_tmp.append(reward)

        neis_unit_idx.append(nei_tmp)
        neis_rewards.append(rewards_tmp)

    return states_idx, targets_bias, neis_unit_idx, neis_rewards


def ivcml_node_observe_feature(query_id: str, graph: nx.Graph, node_src, node_tar, feature_embedding):
    sp = graph_shortest_path(graph, node_src, node_tar)
    edges = []
    for _n in sp:
        actions = [_n] + list(graph.adj[_n].keys())
        edges += [(_n, _nei) for _nei in actions]

    node_num = graph.number_of_nodes()
    dtype = feature_embedding.dtype

    _a_batch = []
    _vec_batch = []
    _keys_batch = []
    for center, neighbor in edges:
        adj_mat_nei = get_adj_nei(graph, center, neighbor)
        vec_nei = indicator_vec(center, node_num, dtype=dtype)
        key = "%s_%d_%d" % (query_id, center, neighbor)
        _a_batch.append(adj_mat_nei)
        _vec_batch.append(vec_nei)
        _keys_batch.append(key)

    return _a_batch, _vec_batch, _keys_batch


def get_desc_input_batch(query_id, input_reader, fea_reader):
    query_data = input_reader[query_id]

    video_pool = query_data["video_pool"]
    node_num = query_data["num_node"]
    coo_adj_zero = query_data["coo_adj_zero"]
    coo_mom = query_data["coo_mom"]
    nei_edge_to_remove = query_data["nei_edge_to_remove"]
    state_node_idx = query_data["state_node_idx"]
    nei_node_idx = query_data["nei_node_idx"]
    tar_bias = query_data["tar_bias"]
    query_text = query_data["desc"]
    neis_rewards = query_data["reward"]
    max_nei_num = 8  # including the current state

    adj_mat = recover_adj_mat(node_num, coo_adj_zero, coo_mom)
    unit_fea = fea_reader[video_pool]
    text_id_bert, mask = text_to_id_bert(query_text)
    fea_dim = unit_fea.shape[1]

    nei_vec_batch = []
    adj_mat_nei_batch = []
    tar_bias_batch = []
    unit_fea_batch = []
    tar_mask_batch = []
    text_ids_batch = []
    nei_reward_batch = []
    for state, neis, target_bias, neis_edges_to_remove, reward in \
            zip(state_node_idx, nei_node_idx, tar_bias, nei_edge_to_remove, neis_rewards):
        adj_mat_nei_case = torch.zeros(max_nei_num, node_num, node_num)
        nei_vec_case = torch.zeros(max_nei_num, node_num)
        unit_fea_case = torch.zeros(max_nei_num, node_num, fea_dim)
        tar_mask_case = torch.zeros(max_nei_num, 1)
        nei_reward = torch.zeros(max_nei_num)

        nei_num = min(max_nei_num, len(reward))
        nei_reward[:nei_num] = torch.tensor(reward)[:nei_num]

        for i, (nei, nei_edges) in enumerate(zip(neis, neis_edges_to_remove)):
            if i > max_nei_num-1:  # 1 state and max_nei_num neighbors
                break
            if i == max_nei_num-1 and target_bias > max_nei_num-1:
                nei = neis[target_bias]
                nei_edges = neis_edges_to_remove[target_bias]
                target_bias = i

            nei_vec = indicator_vec(nei, node_num)
            adj_mat_nei = remove_undirectional_edge(adj_mat, nei_edges)

            tar_mask_case[i] = 1
            adj_mat_nei_case[i] = adj_mat_nei
            nei_vec_case[i] = nei_vec
            unit_fea_case[i] = unit_fea

        text_ids_batch.append(text_id_bert)
        tar_bias_batch.append(target_bias)
        nei_vec_batch.append(nei_vec_case)
        adj_mat_nei_batch.append(adj_mat_nei_case)
        unit_fea_batch.append(unit_fea_case)
        tar_mask_batch.append(tar_mask_case)
        nei_reward_batch.append(nei_reward)

    adj_mat_nei_batch = torch.stack(adj_mat_nei_batch, dim=0)
    nei_vec_batch = torch.stack(nei_vec_batch, dim=0)
    unit_fea_batch = torch.stack(unit_fea_batch, dim=0)
    tar_mask_batch = torch.stack(tar_mask_batch, dim=0)
    text_ids_batch = torch.stack(text_ids_batch, dim=0)
    tar_bias_batch = torch.tensor(tar_bias_batch)
    nei_reward_batch = torch.stack(nei_reward_batch, dim=0)

    tmp = [text_ids_batch, adj_mat_nei_batch, nei_vec_batch, unit_fea_batch, tar_mask_batch, tar_bias_batch, nei_reward_batch]
    show_type_tree(tmp)
    exit()

    return text_ids_batch, adj_mat_nei_batch, nei_vec_batch, unit_fea_batch, tar_mask_batch, tar_bias_batch, nei_reward_batch


def get_desc_input_batch_wo_aggregate(query_id, input_reader, fea_reader, node_observe_weight_reader):
    query_data = input_reader[query_id]
    video_pool = query_data["video_pool"]
    node_num = query_data["num_node"]
    state_node_idx = query_data["state_node_idx"]
    nei_node_idx = query_data["nei_node_idx"]
    tar_bias = query_data["tar_bias"]
    query_text = query_data["desc"]
    neis_rewards = query_data["reward"]

    unit_fea = fea_reader[video_pool]
    text_id_bert, mask = text_to_id_bert(query_text)

    nei_observe_weight_batch = []
    tar_mask_batch = []
    nei_reward_batch = []
    for state, neis, reward in zip(state_node_idx, nei_node_idx, neis_rewards):
        nei_num = len(neis)
        tar_mask_step = torch.ones(nei_num)
        nei_reward = torch.tensor(reward, dtype=torch.float)
        weight_node_obs_step = torch.zeros(nei_num, node_num)

        for i, nei in enumerate(neis):
            key = "%s_%d_%d" % (query_id, state, nei)
            weight_node_obs = node_observe_weight_reader[key]
            if weight_node_obs is not None:
                weight_node_obs_step[i] = weight_node_obs
            # else:
            #     print("Error:", key, "is missing.")

        nei_observe_weight_batch.append(weight_node_obs_step)
        tar_mask_batch.append(tar_mask_step)
        nei_reward_batch.append(nei_reward)

    unit_fea_batch = unit_fea
    text_ids_batch = text_id_bert
    tar_bias_batch = torch.tensor(tar_bias)

    nei_observe_weight_batch = stack_tenor_list(nei_observe_weight_batch, dim=0)
    tar_mask_batch = stack_tenor_list(tar_mask_batch, dim=0)
    nei_reward_batch = stack_tenor_list(nei_reward_batch, dim=0)

    return text_ids_batch, nei_observe_weight_batch, unit_fea_batch, tar_mask_batch, tar_bias_batch, nei_reward_batch


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

    x = ivcml_preprocessing(g, 1, 4)
    show_type_tree(x)
