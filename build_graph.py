import argparse
import os
import json
from os.path import exists
from time import time
import pprint

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import pprint
from apex import amp
from horovod import torch as hvd

from data import (VcmrFullEvalDataset, vcmr_full_eval_collate,
                  VcmrVideoOnlyFullEvalDataset,
                  PrefetchLoader, QueryTokLmdb,
                  video_collate)

from load_data import (get_video_ids,
                       load_video_sub_dataset,
                       load_video_only_dataset)

from data.loader import move_to_cuda
from model.vcmr import HeroForVcmr

from utils.logger import LOGGER
from utils.const import VFEAT_DIM, VCMR_IOU_THDS
from utils.tvr_standalone_eval import eval_retrieval
from utils.distributed import all_gather_list
from utils.misc import Struct
from utils.basic_utils import (
    load_json, save_json)
from utils.tvr_eval_utils import (
    find_max_triples_from_upper_triangle_product,
    generate_min_max_length_mask,
    get_submission_top_n, post_processing_vcmr_nms,
    post_processing_svmr_nms)

from matplotlib import pyplot as plt
import random
from collections import Counter
import networkx as nx
from collections.abc import Iterable   # import directly from collections for Python < 3.3
import textwrap

COLOR = {"target": "orange",
         "default": "green",
         "ini": "cyan",
         "ini_and_tar": "yellow"}

# SHAPE = {"target": "*",
#          "default": "o",
#          "subtitle": "^",
#          "ini": "o",
#          "ini_and_tar": "o"}


def deduplicate(arr: list):
    """
    :param arr: the original list
    :return: deduplicated list
    """
    return list(set(arr))


def uniform_normalize(t: torch.Tensor):
    """
    :param t:
    :return: normalized tensor
    >>> a = torch.rand(5)
    tensor([0.3357, 0.9217, 0.0937, 0.1567, 0.9447])
    >>> uniform_normalize(a)
    tensor([0.2843, 0.9730, 0.0000, 0.0740, 1.0000])
    """
    t -= t.min(-1, keepdim=True)[0]
    t /= t.max(-1, keepdim=True)[0]
    return t


def sample_dict(d: dict, n: int, seed=None):
    """
    :param d: original dict
    :param n: number of keys to sample
    :param seed: random seed of sampling
    :return: sampled dictionary
    """
    if seed is not None:
        random.seed(seed)
    keys = random.sample(d.keys(), n)
    sample_d = {k: d[k] for k in keys}
    return sample_d


def cosine_sim(fea: torch.Tensor):
    """
    :param fea: feature vector [N, D]
    :return: score: cosine similarity score [N, N]
    """
    fea /= torch.norm(fea, dim=-1, keepdim=True)
    score = fea.mm(fea.T)
    return score


def percentile(data: list, p=0.5):
    """
    :param data: origin list
    :param p: frequency percentile
    :return: the element at frequency percentile p
    """
    assert 0 < p < 1
    boundary = len(data) * p
    counter = sorted(Counter(data).items(), key=lambda x: x[0])
    keys, counts = zip(*counter)
    accumulation = 0
    for i, c in enumerate(counts):
        accumulation += c
        if accumulation > boundary:
            return keys[i]
    return None


def save_plt(fig_name: str, mute):
    """
    :param fig_name: path of target file
    :param mute: mute the output if True
    :return: None
    """
    fig_dir = fig_name[:fig_name.rfind("/")]
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(fig_name)
    plt.clf()
    if not mute:
        print("'%s' saved." % fig_name)


def list_histogram(data: list, color="b", title="Histogram of element frequency.", x_label="", y_label="Frequency", fig_name="hist.png", mute=False):
    """
    :param data: the origin list
    :param color: color of the histogram bars
    :param title: bottom title of the histogram
    :param x_label: label of x axis
    :param y_label: label of y axis
    :param fig_name: path of target file
    :param mute: mute the output if True
    :return: None
    """
    def adaptive_bins(_data):
        """
        :param _data: the original list to visualize
        :return: the adaptive number of bins
        """
        n = len(deduplicate(_data))
        return n

    bins = adaptive_bins(data)
    plt.hist(data, color=color, bins=bins)
    plt.gca().set(xlabel=x_label, ylabel=y_label)
    plt.title("Fig. "+title, fontdict={'family': 'serif', "verticalalignment": "bottom"})
    save_plt(fig_name, mute)


def get_subtitle_subject(subtitle):
    if ":" in subtitle:
        return subtitle.split(":")[0].strip().lower()
    return None


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


def clip_info(part="train"):
    folder_p = "/raw/"
    f_name = "tvr_video2dur_idx.json"
    clip_duration = {}
    with open(folder_p+f_name, "r") as f:
        for line in f:
            clips = json.loads(line)
            if part == "all":
                for tag in clips:
                    partial_clips = clips[tag]
                    for clip_id in partial_clips:
                        clip_duration[clip_id] = partial_clips[clip_id][0]
            else:
                for clip_id in clips[part]:
                    clip_duration[clip_id] = clips[part][clip_id][0]
    return clip_duration


def plot_graph(v, e, markers=None, colors=None, confidence=None, title="Example graph.", fig_name="graph.png", mute=False, full_edge=False):
    """
    :param v: list of vertices [(x1, y1), ...]
    :param e: list of edges [(i, j), ...]
    :param markers: shapes of vertices
    :param colors: colors of vertices
    :param confidence: confidence of edges
    :param title: title of image
    :param fig_name: file name of saved image
    :param mute: mute the output if True
    :param full_edge: draw all edges including intrinsic temporal relation
    :return: None
    >>>     v = [(1, 2), (4, 6), (6, 4), (7, 9), (9, 5)]
    >>>     e = [(0, 1), (2, 3)]
    >>>     plot_graph(v, e, fig_name="test_graph.png")
    """
    # Initialize
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index

    # plotting points as a scatter plot
    x, y = zip(*v)

    for i, (st, ed) in enumerate(e):
        p = confidence[i] if confidence is not None else None
        if y[st] != y[ed] or (st == ed-1):  # lines with overlap
            if st == ed-1:
                if not full_edge:
                    continue
                color = "blue" if st % 2 else "green"
            else:
                color = "red"
            ax.plot([x[st], x[ed]], [y[st], y[ed]], color=color, marker='', alpha=p)
        else:
            tmp_x, tmp_y = (x[st] + x[ed]) / 2, y[st] + random.uniform(1, 2)
            ax.plot([x[st], tmp_x], [y[st], tmp_y], color="red", marker='', alpha=p)
            ax.plot([tmp_x, x[ed]], [tmp_y, y[ed]], color="red", marker='', alpha=p)

    for i, v in enumerate(v):
        color = "green" if colors is None else colors[i]
        marker = "o" if markers is None else markers[i]
        ax.scatter(v[0], v[1], color=color, marker=marker, s=30)

    title = ax.set_title("Fig. " + title, fontdict={'family': 'serif', "verticalalignment": "bottom"}, loc='center', wrap=True)
    fig.tight_layout()
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    save_plt(fig_name, mute)


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


def load_hero_pred(opts, task):
    # TODO: 1. top-k setting may be a problem
    assert task in {"vr", "vcmr"}
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


def node_color(_vid, _range, _vid_tar, _range_gt, _ini_vid, _frame_ini):
    if not _range:
        return "grey"

    _st, _ed = _range[0], _range[-1]
    st_gt, ed_gt = _range_gt

    is_tar_video = (_vid == _vid_tar)
    case_1 = _st <= st_gt <= _ed
    case_2 = _st <= ed_gt <= _ed
    case_3 = st_gt <= _st <= _ed <= ed_gt
    is_part_of_target = (case_1 or case_2 or case_3) and is_tar_video

    is_ini_video = (_vid == _ini_vid)
    is_ini = (_st <= _frame_ini <= _ed) and is_ini_video

    if is_part_of_target and is_ini:
        color = COLOR["ini_and_tar"]
    elif is_part_of_target:
        color = COLOR["target"]
    elif is_ini:
        color = COLOR["ini"]
    else:
        color = COLOR["default"]

    return color


def node_shape(_idx):
    if _idx is None:
        return "^"  # triangle for visual only nodes
    return "o"  # circle for visual + text


def complement_sub_unit(sub_idx2frame_idx):
    subs_level_units = [(sub_idx, frame_range[0], frame_range[-1]) if frame_range else (sub_idx, None, None) for sub_idx, frame_range in sub_idx2frame_idx]
    extended_sub_level_units = []
    old_ed = -1
    for sub_idx, st, ed in subs_level_units:
        if st is None:
            extended_sub_level_units.append((sub_idx, []))
            # TODO process the old_ed
            continue
        if st > old_ed + 1:  # missed frame
            interval = (None, list(range(old_ed+1, st-1+1)))
            extended_sub_level_units.append(interval)
        extended_sub_level_units.append((sub_idx, list(range(st, ed+1))))
        old_ed = ed
    # for sub_idx, frame_range in extended_sub_level_units:
    #     if frame_range:
    #         print(sub_idx, frame_range[0], frame_range[-1])
    #     else:
    #         print(sub_idx, None, None)
    # print()

    return extended_sub_level_units


def get_subtitle_level_vid_feature(video_db, vid_pool):
    # TODO: combine both visual and subtitle feature
    sub_fea = []
    for i, vid in enumerate(vid_pool):
        _, _, _, clip_level_v_feats, _, _, sub_idx2frame_idx = video_db[vid]
        sub_idx2frame_idx = complement_sub_unit(sub_idx2frame_idx)
        for j, (sub_idx, frame_range) in enumerate(sub_idx2frame_idx):
            if frame_range:
                sub_level_fea = clip_level_v_feats[frame_range].mean(dim=0)
            else:
                sub_level_fea = torch.zeros(clip_level_v_feats.shape[-1])
            sub_fea.append(sub_level_fea)
    sub_fea = torch.stack(sub_fea, dim=0)
    return sub_fea


def get_vertices(video_db, vid_pool, tar_vid, range_gt, ini_vid, frame_idx_ini):
    vertices, v_color, v_shape = [], [], []
    boundary_indicator = []
    for i, vid in enumerate(vid_pool):
        _, _, _, _, _, _, sub_idx2frame_idx = video_db[vid]
        sub_idx2frame_idx = complement_sub_unit(sub_idx2frame_idx)
        for j, (sub_idx, frame_range) in enumerate(sub_idx2frame_idx):
            vertices.append([j * 3, i * 5])
            color = node_color(vid, frame_range, tar_vid, range_gt, ini_vid, frame_idx_ini)
            shape = node_shape(sub_idx)
            # if frame_range:
            #     # print(f"{_:4}", f"{frame_range[0]:3}", f"{frame_range[-1]:3}", f"{range_gt[0]:3}", f"{range_gt[1]:3}", vid == tar_vid)
            #     if vid == ini_vid:
            #         print(f"{_:4}", f"{frame_range[0]:3}", f"{frame_range[-1]:3}", f"{frame_idx_ini:3}", vid == ini_vid, color)
            v_color.append(color)
            v_shape.append(shape)
            boundary_indicator.append(1)
        boundary_indicator[-1] = 0
    natural_edges = [(node_i, node_i+1) for node_i, is_not_boundary in enumerate(boundary_indicator) if is_not_boundary]
    return vertices, v_color, v_shape, natural_edges


def build_network(vertices, edges):
    g = nx.Graph()
    node_ids = list(range(len(vertices)))
    g.add_nodes_from(node_ids)
    g.add_edges_from(edges)
    return g


def show_type_tree(data, indentation=4, depth=0):
    def _indent(content: str):
        if depth == 0:
            print()
        print(" " * (depth * indentation) + content)

    if not isinstance(data, Iterable):
        if isinstance(data, int):
            _indent("int: %d" % data)
        elif isinstance(data, float):
            _indent("float: %.2f" % data)
        else:
            _indent(str(type(data)))
        return

    if isinstance(data, list):
        _indent("list with size %d" % len(data))
        for item in data:
            show_type_tree(item, depth=depth + 1)

    elif isinstance(data, tuple):
        _indent("tuple with size %d" % len(data))
        for item in data:
            show_type_tree(item, depth=depth + 1)

    elif isinstance(data, dict):
        _indent("dict with size %d" % len(data))
        for key in data:
            _indent(str(key))
            show_type_tree(data[key], depth=depth + 1)

    elif isinstance(data, str):
        _indent("str")

    elif isinstance(data, torch.Tensor):
        _indent("Tensor with shape" + str(list(data.shape)))

    else:
        _indent(str(type(data)))
        for item in data:
            show_type_tree(item, depth=depth + 1)


def graph_diameter(g):
    """
    :param g: instance of NetworkX graph
    :return: diameter of g
    """
    try:
        dia = nx.algorithms.distance_measures.diameter(g)
    except nx.exception.NetworkXError:
        dia = -1
    return dia


def graph_shortest_distance(g, src, tar):
    """
    :param g: instance of NetworkX graph
    :param src: source node
    :param tar: target node
    :return: length of shortest path from src to tar
    """
    try:
        dis = nx.algorithms.shortest_paths.generic.shortest_path_length(g, source=src, target=tar)
    except nx.exception.NetworkXNoPath:
        dis = -1
    return dis


def process_query(val_loader, vr_pred, vcmr_pred, opts):
    modes = ["vis graph", "vis sta", "toy sta"]
    mode = modes[1]

    if mode == "vis graph":
        plot_graph_mode = True  # sample and draw graph if True else analyze the data
        plot_sta_mode = False
        sample_mode = True

    elif mode == "toy sta":
        plot_graph_mode = False
        plot_sta_mode = True
        sample_mode = True

    elif mode == "vis sta":
        plot_graph_mode = False
        plot_sta_mode = True
        sample_mode = False

    else:
        assert False

    if sample_mode:
        sample_num = 2
        vr_pred = sample_dict(vr_pred, sample_num, seed=1)

    query_data = val_loader.dataset.query_data
    video_db = val_loader.dataset.video_db

    thds = {t / 100: [] for t in range(80, 90, 1)}

    dia_count = {thd: [] for thd in thds}
    dis_rand_count = {thd: [] for thd in thds}
    dis_hero_count = {thd: [] for thd in thds}

    for desc_idx, desc_id in tqdm(enumerate(vr_pred), desc="Computing moment similarity", total=len(vr_pred)):
        # desc_id = "90143"
        print("\nDESC_%s" % desc_id)
        pred_vids = vr_pred[desc_id]
        query_item = query_data[desc_id]
        query_text = query_item["desc"]
        tar_vid = query_item["vid_name"]
        tar_vid_item = video_db[tar_vid]

        # Target Video Information
        _, _, _, frame_fea_tar, _, _, _ = tar_vid_item
        frame_num_tar = frame_fea_tar.shape[0]
        st_tar, ed_tar = query_item["ts"]
        duration = query_item["duration"]
        frame_st_gt, frame_ed_gt = int(st_tar/duration*frame_num_tar), int(ed_tar/duration*frame_num_tar)

        # Initialization Video Information
        rank, rank_in_vr, st_ini, ed_ini = vcmr_pred[desc_id]
        ini_vid = vr_pred[desc_id][rank_in_vr]
        _, _, _, frame_fea_ini, _, _, _ = video_db[ini_vid]
        frame_num_ini = frame_fea_ini.shape[0]
        frame_ini = int((st_ini + ed_ini) / 2 / duration * frame_num_ini)

        if tar_vid in pred_vids:
            vid_pool = pred_vids
        else:
            vid_pool = [tar_vid] + pred_vids[:-1]

        sub_fea = get_subtitle_level_vid_feature(video_db, vid_pool)
        s = torch.tril(cosine_sim(sub_fea), diagonal=-2)

        tar_range = (frame_st_gt, frame_ed_gt)
        vertices, v_color, v_shape, natural_edges = get_vertices(video_db, vid_pool, tar_vid, tar_range, ini_vid, frame_ini)
        assert "orange" in v_color

        for thd in thds:
            ind = (s > thd)
            edges = ind.nonzero().tolist()

            if plot_graph_mode:  # sample and plot graph
                conf = uniform_normalize(s[ind.nonzero(as_tuple=True)] - thd * 0.9).tolist()
                plot_graph(vertices, edges,
                           markers=v_shape,
                           colors=v_color,
                           confidence=conf,
                           title=f"Graph of description {desc_id} with threshold {thd:.2f}. '{query_text}'",
                           fig_name=f"desc_graph/desc_{desc_id}/desc_{desc_id}_t_{thd:.2f}_graph.png",
                           mute=False)

            else:  # build network and analyze statistic
                g_network = build_network(vertices, edges + natural_edges)
                dia = graph_diameter(g_network)

                node_tar = v_color.index("orange")
                if "cyan" in v_color or "yellow" in v_color:
                    if "cyan" in v_color:
                        node_src_hero = v_color.index("cyan")
                    else:
                        node_src_hero = v_color.index("yellow")
                    dis_hero = graph_shortest_distance(g_network, node_src_hero, node_tar)
                else:
                    dis_hero = -2

                node_src_rand = int(random.uniform(0, len(vertices)))
                dis_rand = graph_shortest_distance(g_network, node_src_rand, node_tar)

                print("DESC_%s, THRESHOLD:%.2f RANDOM: %2d/%2d; HERO: %2d/%2d" % (desc_id, thd, dis_rand, dia, dis_hero, dia))
                dia_count[thd].append(dia)
                dis_hero_count[thd].append(dis_hero)
                dis_rand_count[thd].append(dis_rand)

    if plot_sta_mode:  # global statistic
        toy_marker = "_toy" if (mode == "toy sta") else ""
        for thd in thds:
            p = 0.9
            b_dia, b_dis = percentile(dia_count[thd], p), percentile(dis_rand_count[thd], p)
            list_histogram(dia_count[thd], x_label="Diameter", fig_name=f"desc_graph/{opts.split}_diameter{toy_marker}/graph_diameter_hist_t_{thd:.2f}_p{p:.2f}_{b_dia}.png")
            list_histogram(dis_rand_count[thd], x_label="Shortest Distance", fig_name=f"desc_graph/{opts.split}_distance_random{toy_marker}/graph_distance_hist_t_{thd:.2f}_p{p:.2f}_{b_dis}.png")
            list_histogram(dis_hero_count[thd], x_label="Shortest Distance", fig_name=f"desc_graph/{opts.split}_distance_hero_vcmr{toy_marker}/graph_distance_hist_t_{thd:.2f}_p{p:.2f}_{b_dis}.png")


def main(opts):
    hvd.init()
    vr_pred = load_hero_pred(opts, task="vr")
    vcmr_pred = load_hero_pred(opts, task="vcmr")
    val_loader = build_dataloader(opts)
    # vid_sub_range = load_subtitle_range(val_loader, opts)
    # vid_to_subs = load_tvr_subtitle()

    if hvd.rank() != 0:  # save for only one time
        exit()
    process_query(val_loader, vr_pred, vcmr_pred, opts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--sub_txt_db", default="/txt/tv_subtitles.db", type=str, help="The input video subtitle corpus. (LMDB)")
    parser.add_argument("--vfeat_db", default="/video/tv", type=str, help="The input video frame features.")
    parser.add_argument("--query_txt_db", default="/txt/tvr_val.db", type=str, help="The input test query corpus. (LMDB)")
    parser.add_argument("--split", choices=["val", "test_public", "test"], default="val", type=str, help="The input query split")
    parser.add_argument("--task", choices=["tvr", "how2r", "didemo_video_sub", "didemo_video_only"], default="tvr", type=str, help="The evaluation vcmr task")
    parser.add_argument("--checkpoint", default=None, type=str, help="pretrained model checkpoint steps")
    parser.add_argument("--batch_size", default=80, type=int, help="number of queries in a batch")
    parser.add_argument("--vcmr_eval_video_batch_size", default=50, type=int, help="number of videos in a batch")
    parser.add_argument( "--full_eval_tasks", type=str, nargs="+", choices=["VCMR", "SVMR", "VR"], default=["VCMR", "SVMR", "VR"], help="Which tasks to run. VCMR: Video Corpus Moment Retrieval; SVMR: "
                                                                                                                                        "Single Video Moment Retrieval;" "VR: regular Video Retrieval. "
                                                                                                                                        "(will be performed automatically with VCMR)")
    parser.add_argument("--output_dir", default=None, type=str, help="The output directory where the model checkpoints will be written.")

    # device parameters
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4, help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true', help="pin memory")

    args = parser.parse_args()

    # options safe guard
    # TODO

    main(args)
