import argparse
import torch
from tqdm import tqdm
from horovod import torch as hvd
from data import video_collate
from data.loader import move_to_cuda
from utils.logger import LOGGER

import random
import numpy as np
import math
import copy
import pprint
from ivcml_util import cosine_sim, uniform_normalize, show_type_tree
from ivcml_util import sample_dict, percentile, list_histogram, pairwise_equation
from ivcml_graph import complement_sub_unit, plot_graph, build_static_graph, get_src_node, get_tar_node
from ivcml_graph import build_network, graph_diameter, graph_shortest_distance, get_cliques, group_clique_nodes
from ivcml_graph import COLOR
from ivcml_data import load_video2duration, load_hero_pred, load_model, load_tvr_subtitle, build_dataloader


def get_subtitle_level_unit_feature(video_db, vid_pool, mode="vt", model=None):
    if mode == "v":
        sub_fea = []
        for i, vid in enumerate(vid_pool):
            _, _, _, frame_fea, _, _, sub_idx2frame_idx = video_db[vid]
            frame_num = frame_fea.shape[0]

            sub_idx2frame_idx = complement_sub_unit(sub_idx2frame_idx, frame_num)
            for j, (sub_idx, frame_range) in enumerate(sub_idx2frame_idx):
                if frame_range:
                    sub_level_fea = frame_fea[frame_range].mean(dim=0)
                else:
                    sub_level_fea = torch.zeros(frame_fea.shape[-1])
                sub_fea.append(sub_level_fea)
        sub_fea = torch.stack(sub_fea, dim=0)

    elif mode == "vt":
        video_batch = [video_db[vid] for vid in vid_pool]
        batch_frame_emb, batch_c_attn_masks = calc_single_video_repr(model, video_batch)
        sub_fea = []
        for i, vid in enumerate(vid_pool):
            _, _, _, _, mask, _, sub_idx2frame_idx = video_db[vid]
            frame_num = mask.shape[0]
            sub_idx2frame_idx = complement_sub_unit(sub_idx2frame_idx, frame_num)
            for j, (sub_idx, frame_range) in enumerate(sub_idx2frame_idx):
                if frame_range:
                    sub_level_fea = batch_frame_emb[i][frame_range].mean(dim=0)
                else:
                    sub_level_fea = torch.zeros(batch_frame_emb.shape[-1],
                                                dtype=batch_frame_emb.dtype,
                                                device=batch_frame_emb.device)
                sub_fea.append(sub_level_fea)
        sub_fea = torch.stack(sub_fea, dim=0)

    else:
        assert False, mode

    return sub_fea


def build_video_pool(pred_vids, vid_tar, vid_ini):
    if vid_tar in pred_vids:  # initialization video must be in predicted video pool
        return pred_vids

    base = [vid_tar, vid_ini]
    random.seed(1)
    vid_pool = base + random.sample([vid for vid in pred_vids if vid not in base], 2)
    return vid_pool


def calculate_similarity(unit_visual_fea):
    score = torch.tril(cosine_sim(unit_visual_fea), diagonal=-2)
    return score


def calc_single_video_repr(model, video_batch):
    video_batch = move_to_cuda(video_collate(video_batch))
    # Safeguard fp16
    for k, item in video_batch.items():
        if isinstance(item, torch.Tensor) and item.dtype == torch.float32:
            video_batch[k] = video_batch[k].to(dtype=next(model.parameters()).dtype)
    # show_type_tree(video_batch)
    curr_frame_embeddings = model.v_encoder(video_batch, 'repr')
    curr_c_attn_masks = video_batch['c_attn_masks']
    return curr_frame_embeddings, curr_c_attn_masks


@torch.no_grad()
def calc_full_frame_repr(model, val_loader, split, opts, model_opts):
    LOGGER.info("start running full VCMR evaluation on {opts.task} {split} split...")
    model.eval()
    val_vid2idx = val_loader.dataset.vid2idx
    video2idx_global = val_vid2idx[split] if split in val_vid2idx else val_vid2idx

    video_ids = sorted(list(video2idx_global.keys()))
    video2idx_local = {e: i for i, e in enumerate(video_ids)}
    total_frame_embeddings, total_c_attn_masks = None, None
    video_batch = []
    video_idx: [] = []
    max_clip_len = 0
    for video_i, (vid, vidx) in tqdm(enumerate(video2idx_local.items()), desc="Computing Video Embeddings", total=len(video2idx_local)):
        video_item = val_loader.dataset.video_db[vid]
        video_batch.append(video_item)
        video_idx.append(vidx)

        full_batch = len(video_batch) == opts.vcmr_eval_video_batch_size
        data_finished = video_i == len(video2idx_local) - 1
        if not full_batch and not data_finished:
            continue  # continue to accumulate

        # process a batch
        curr_frame_embeddings, curr_c_attn_masks = calc_single_video_repr(model, video_batch)
        curr_clip_len = curr_frame_embeddings.size(-2)
        assert curr_clip_len <= model_opts.max_clip_len

        if total_frame_embeddings is None:
            feat_dim = curr_frame_embeddings.size(-1)
            total_frame_embeddings = torch.zeros((len(video2idx_local), model_opts.max_clip_len, feat_dim), dtype=curr_frame_embeddings.dtype, device=curr_frame_embeddings.device)
            total_c_attn_masks = torch.zeros((len(video2idx_local), model_opts.max_clip_len), dtype=curr_c_attn_masks.dtype, device=curr_frame_embeddings.device)

        indices = torch.tensor(video_idx)
        total_frame_embeddings[indices, :curr_clip_len] = curr_frame_embeddings
        total_c_attn_masks[indices, :curr_clip_len] = curr_c_attn_masks
        max_clip_len = max(max_clip_len, curr_clip_len)
        video_batch, video_idx = [], []

    total_frame_embeddings = total_frame_embeddings[:, :max_clip_len, :]
    total_c_attn_masks = total_c_attn_masks[:, :max_clip_len]
    return total_frame_embeddings, total_c_attn_masks


def set_params(mode):
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

    elif mode == "no io":
        plot_graph_mode = False
        plot_sta_mode = False
        sample_mode = False

    else:
        assert False

    return plot_graph_mode, plot_sta_mode, sample_mode


def process_query(model, val_loader, vr_pred, vcmr_pred, vid_to_subs, opts, model_opts):
    model.eval()

    # super parameters
    feature_types = ["v", "vt"]
    feature_type = feature_types[1]
    sim_funcs = ["cosine"]
    sim_func = sim_funcs[0]
    modes = ["vis graph", "vis sta", "toy sta", "no io"]
    mode = modes[-1]
    plot_graph_mode, plot_sta_mode, sample_mode = set_params(mode)
    is_group_clique = True
    group_folder_tag = "grouped/" if is_group_clique else "ungrouped/"
    curve_functions = {
        "double_sqrt": lambda x: math.sqrt(math.sqrt(x)),
        "sqrt": lambda x: math.sqrt(x),
        "linear": lambda x: x
    }
    curve_tag = "linear"
    f_curve = curve_functions[curve_tag]

    pprint.pprint({
        "feature_types": feature_types,
        "sim_func": sim_func,
        "mode": mode,
        "is_group_clique": is_group_clique,
        "curve_tag": curve_tag,
    })

    # start process
    try:
        is_to_write = hvd.rank() == 0
    except ValueError:
        is_to_write = True

    if sample_mode:
        vr_pred = sample_dict(vr_pred, 2, seed=1)

    query_data = val_loader.dataset.query_data
    video_db = val_loader.dataset.video_db
    vid2duration = load_video2duration(opts.split)

    if feature_type == "v":
        thds = {t / 100: [] for t in range(95, 84, -1)}
    elif feature_type == "vt":
        thds = {t / 100: [] for t in range(85, 34, -2)}
    else:
        assert False, feature_type

    dia_count = {thd: [] for thd in thds}
    dis_rand_count = {thd: [] for thd in thds}
    dis_hero_count = {thd: [] for thd in thds}

    for desc_idx, desc_id in tqdm(enumerate(vr_pred), desc="Processing Moment", total=len(vr_pred)):
        print("\nDESC_%s" % desc_id)
        pred_vids = vr_pred[desc_id]
        query_item = query_data[desc_id]
        query_text = query_item["desc"]
        vid_tar = query_item["vid_name"]

        # Target Video Information
        tar_vid_item = video_db[vid_tar]
        _, _, _, frame_fea_tar, _, _, _ = tar_vid_item
        frame_num_tar = frame_fea_tar.shape[0]

        # Target moment information
        st_tar, ed_tar = query_item["ts"]
        duration_tar = vid2duration[vid_tar]
        frame_st_gt, frame_ed_gt = int(st_tar/duration_tar*frame_num_tar), int(ed_tar/duration_tar*frame_num_tar)
        assert ed_tar <= duration_tar, f"TARGET: {ed_tar}/{duration_tar}"

        # Initialization Video Information
        rank, rank_in_vr, vid_ini_hero, st_ini, ed_ini = vcmr_pred[desc_id]
        vid_ini = vr_pred[desc_id][rank_in_vr]
        _, _, _, frame_fea_ini, _, _, _ = video_db[vid_ini]
        frame_num_ini = frame_fea_ini.shape[0]
        duration_ini = vid2duration[vid_ini]
        frame_ini = int((st_ini + ed_ini) / 2 / duration_ini * frame_num_ini)
        # assert ed_ini <= duration_ini, f"INITIALIZATION: {ed_ini}/{duration_ini}"
        assert frame_ini <= frame_num_ini, f"INITIALIZATION: {frame_ini}/{frame_num_ini}"

        vid_pool = build_video_pool(pred_vids, vid_tar, vid_ini)
        unit_fea = get_subtitle_level_unit_feature(video_db, vid_pool, feature_type, model)
        sim = calculate_similarity(unit_fea)

        tar_range = (frame_st_gt, frame_ed_gt)
        vertices, v_color, v_shape, static_edges, static_edges_color, static_edges_conf, sub_2_vid = \
            build_static_graph(video_db, vid_to_subs, vid_pool, vid_tar, tar_range, vid_ini, frame_ini)
        vid_level_mask = pairwise_equation(np.array(sub_2_vid))
        vid_level_mask = torch.from_numpy(vid_level_mask).to(sim.device)
        nx_graph = build_network(vertices, static_edges)  # initialize the networkx graph

        edges_old = []  # old edge from last step
        for thd_cross in thds:  # threshold from high to low
            ind_cross = (sim > thd_cross) * torch.logical_not(vid_level_mask)
            edges_cross = ind_cross.nonzero().tolist()  # edges using unit similarity
            thd_inner = f_curve(thd_cross)
            ind_inner = (sim > thd_inner) * vid_level_mask
            edges_inner = ind_inner.nonzero().tolist()  # edges using unit similarity
            edges = edges_cross + edges_inner
            edges_increment = [e for e in edges if e not in edges_old]  # edges to be incrementally added
            edges_old = edges  # update the old edges

            nx_graph.add_edges_from(edges_increment)
            e_color = [COLOR["dark red"]] * len(edges)  # dark red

            e_conf = []
            e_conf += uniform_normalize(sim[ind_cross.nonzero(as_tuple=True)] - thd_cross * 0.9).tolist() if edges_cross else []
            e_conf += uniform_normalize(sim[ind_inner.nonzero(as_tuple=True)] - thd_inner * 0.9).tolist() if edges_inner else []

            if plot_graph_mode and is_to_write:  # sample and plot graph
                # print("GRAPH-PLOT DESC_%s, THRESHOLD:%.2f |E|:%3d |V|:%3d |En|: %3d" % (desc_id, thd_cross, len(edges), len(vertices), len(static_edges)))

                if is_group_clique:
                    cliques = get_cliques(nx_graph)
                    vertices_grouped, v_color_grouped, v_shape_grouped, edges_grouped, e_color_grouped, e_conf_grouped = \
                        group_clique_nodes(
                            copy.deepcopy(cliques),
                            copy.deepcopy(vertices),
                            copy.deepcopy(v_color),
                            copy.deepcopy(v_shape),
                            edges + static_edges,
                            e_color + static_edges_color,
                            e_conf + static_edges_conf,
                            sub_2_vid)
                    v_plot, v_color_plot, v_shape_plot = vertices_grouped, v_color_grouped, v_shape_grouped
                    e_plot, e_color_plot, e_conf_plot = edges_grouped, e_color_grouped, e_conf_grouped

                else:
                    v_plot, v_color_plot, v_shape_plot = vertices, v_color, v_shape
                    e_plot, e_color_plot, e_conf_plot = edges + static_edges, e_color + static_edges_color, e_conf + static_edges_conf

                plot_graph(v_plot,
                           e_plot,
                           markers=v_shape_plot,
                           v_colors=v_color_plot,
                           e_colors=e_color_plot,
                           confidence=e_conf_plot,
                           title=f"Graph of description {desc_id} with cross-θ {thd_cross:.2f} and inner-θ {thd_inner:.2f} . '{query_text}'",
                           fig_name=f"desc_graph/{sim_func}_{feature_type}/{group_folder_tag}desc_{desc_id}/desc_{desc_id}_tc_{thd_cross:.2f}_ti_{thd_inner:.2f}_graph.png",
                           mute=False)

            if plot_sta_mode:  # build network and analyze statistic
                # nx_graph.add_edges_from(edges_increment)
                dia = graph_diameter(nx_graph)

                node_tar = get_tar_node(v_color)
                node_src_hero = get_src_node(v_color)
                node_src_rand = int(random.uniform(0, len(v_color)))
                dis_hero = graph_shortest_distance(nx_graph, node_src_hero, node_tar)
                dis_rand = graph_shortest_distance(nx_graph, node_src_rand, node_tar)

                print("DESC_%s, THRESHOLD:%.2f RANDOM: %2d/%2d; HERO: %2d/%2d |E|:%3d |V|:%3d |En|: %3d" % (desc_id, thd_cross, dis_rand, dia, dis_hero, dia, len(edges), len(vertices), len(static_edges)))
                dia_count[thd_cross].append(dia)
                dis_hero_count[thd_cross].append(dis_hero)
                dis_rand_count[thd_cross].append(dis_rand)

    if plot_sta_mode and is_to_write:  # global statistic
        toy_tag = "_toy" if (mode == "toy sta") else ""
        for thd_cross in thds:
            p = 0.9
            b_dia, b_dis = percentile(dia_count[thd_cross], p), percentile(dis_rand_count[thd_cross], p)
            thd_inner = f_curve(thd_cross)

            list_histogram(dia_count[thd_cross],
                           x_label="Diameter",
                           fig_name=f"desc_graph/{sim_func}_{feature_type}/{group_folder_tag}{opts.split}_diameter{toy_tag}/graph_diameter_hist_tc_{thd_cross:.2f}_ti_{thd_inner:.2f}_p{p:.2f}_{b_dia}.png")

            list_histogram(dis_rand_count[thd_cross],
                           x_label="Shortest Distance (Random)",
                           fig_name=f"desc_graph/{sim_func}_{feature_type}/{group_folder_tag}{opts.split}_distance_random{toy_tag}/graph_distance_hist_tc_{thd_cross:.2f}_ti_{thd_inner:.2f}_p{p:.2f}_{b_dis}.png")

            list_histogram(dis_hero_count[thd_cross],
                           x_label="Shortest Distance (HERO)",
                           fig_name=f"desc_graph/{sim_func}_{feature_type}/{group_folder_tag}{opts.split}_distance_hero_vcmr{toy_tag}/graph_distance_hist_tc_{thd_cross:.2f}_ti_{thd_inner:.2f}_{b_dis}.png")


def main(opts):
    # hvd.init()
    # device = torch.device("cuda", hvd.local_rank())
    device = torch.device("cuda")
    # torch.cuda.set_device(hvd.local_rank())

    vr_pred = load_hero_pred(opts, task="vr")
    vcmr_pred = load_hero_pred(opts, task="vcmr")
    val_loader = build_dataloader(opts)
    model, model_opts = load_model(opts, device)
    # vid_sub_range = load_subtitle_range(val_loader, opts)
    vid_to_subs = load_tvr_subtitle()

    process_query(model, val_loader, vr_pred, vcmr_pred, vid_to_subs, opts, model_opts)


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
