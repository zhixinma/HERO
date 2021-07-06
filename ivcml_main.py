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

from ivcml_graph import EDGE_COLOR
from ivcml_graph import complement_sub_unit, plot_graph, build_static_graph
from ivcml_graph import build_network, get_cliques, group_clique_nodes
from ivcml_graph import get_src_node, get_tar_node
from ivcml_graph import graph_diameter, graph_shortest_distance
from ivcml_graph import build_vcmr_edges
from ivcml_graph import get_mid_frame, get_frame_range
from ivcml_graph import render_shortest_path, shortest_path_edges

from ivcml_load import load_hero_pred, load_model, build_dataloader
from ivcml_load import load_video2duration, build_vid_to_frame_num
from ivcml_load import load_tvr_subtitle
from ivcml_load import ivcml_preprocessing
from ivcml_load import compress_adj_mat
from ivcml_data import HEROUnitFeaLMDB
from ivcml_load import ivcml_node_observe_feature
from ivcml_load import aggregate_batch


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
        c = 0
        for i, vid in enumerate(vid_pool):
            _, _, _, _, mask, _, sub_idx2frame_idx = video_db[vid]
            frame_num = mask.shape[0]
            sub_idx2frame_idx = complement_sub_unit(sub_idx2frame_idx, frame_num)
            c += len(sub_idx2frame_idx)
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


def build_video_pool(vid_proposal, vid_tar, vid_ini):
    if vid_tar in vid_proposal:  # initialization video must be in predicted video pool
        return vid_proposal

    # base = [vid_tar, vid_ini]
    # random.seed(1)
    # vid_pool = base + random.sample([vid for vid in vid_proposal if vid not in base], len(vid_proposal) - 2)
    # random.shuffle(vid_pool)
    # return vid_pool
    return vid_proposal + [vid_tar]


def complete_video_and_moment_pool(vid_proposal, moment_proposal, vid_tar, duration_tar):
    if vid_tar in vid_proposal:  # initialization video must be in predicted video pool
        return vid_proposal, moment_proposal, False

    vid_proposal += [vid_tar]

    # random a moment proposal
    moment_partition = random.uniform(0.1, 0.2)
    _st = random.uniform(0, 1 - moment_partition)
    _ed = _st + moment_partition
    _st, _ed, _score = _st*duration_tar, _ed*duration_tar, 0.1
    moment_proposal += [[vid_tar, _st, _ed, _score]]

    return vid_proposal, moment_proposal, True


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
        ivcml_mode = False

    elif mode == "toy sta":
        plot_graph_mode = False
        plot_sta_mode = True
        sample_mode = True
        ivcml_mode = False

    elif mode == "vis sta":
        plot_graph_mode = False
        plot_sta_mode = True
        sample_mode = False
        ivcml_mode = False

    elif mode == "preprocess":
        plot_graph_mode = False
        plot_sta_mode = False
        sample_mode = False
        ivcml_mode = True

    elif mode == "no io":
        plot_graph_mode = False
        plot_sta_mode = False
        sample_mode = False
        ivcml_mode = False

    else:
        assert False

    return plot_graph_mode, plot_sta_mode, sample_mode, ivcml_mode


def process_ad_hoc_query_graph(model, val_loader, top_videos, vcmr_pred, vid_to_subs, opts, model_opts):
    model.eval()

    # super parameters
    feature_types = ["v", "vt"]
    feature_type = feature_types[1]
    sim_funcs = ["cosine"]
    sim_func = sim_funcs[0]
    modes = ["vis graph", "vis sta", "toy sta", "no io", "preprocess"]
    mode = modes[0]
    plot_graph_mode, plot_sta_mode, sample_mode, ivcml_mode = set_params(mode)
    plot_shortest_path = True
    shortest_path_tag = "_w_sp" if plot_shortest_path else "_wo_sp"
    is_group_clique = False
    group_folder_tag = "grouped/" if is_group_clique else "ungrouped/"
    curve_functions = {
        "double_sqrt": lambda x: math.sqrt(math.sqrt(x)),
        "sqrt": lambda x: math.sqrt(x),
        "linear": lambda x: x
    }
    curve_tag = "double_sqrt"
    f_curve = curve_functions[curve_tag]

    pprint.pprint({
        "feature_type": feature_type,
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
        top_videos = sample_dict(top_videos, 5, seed=2)

    query_data = val_loader.dataset.query_data
    video_db = val_loader.dataset.video_db
    vid2duration = load_video2duration(opts.split)

    if mode == "preprocess":
        thds = {t / 100: [] for t in range(35, 34, -2)}
    else:
        if feature_type == "v":
            thds = {t / 100: [] for t in range(95, 84, -1)}
        elif feature_type == "vt":
            thds = {t / 100: [] for t in range(85, 44, -2)}
        else:
            assert False, feature_type

    dia_count = {thd: [] for thd in thds}
    dis_rand_count = {thd: [] for thd in thds}
    dis_hero_count = {thd: [] for thd in thds}
    #
    for desc_idx, desc_id in tqdm(enumerate(top_videos), desc="Processing Moment", total=len(top_videos)):
        print("\nDESC_%s" % desc_id)
        pred_vids = top_videos[desc_id]
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
        frame_st_gt, frame_ed_gt = get_frame_range(st_tar, ed_tar, duration_tar, frame_num_tar)
        assert ed_tar <= duration_tar, f"TARGET: {ed_tar}/{duration_tar}"

        # Initialization Video Information
        rank, rank_in_vr, vid_ini_hero, st_ini, ed_ini = vcmr_pred[desc_id]
        vid_ini = top_videos[desc_id][rank_in_vr]
        _, _, _, frame_fea_ini, _, _, _ = video_db[vid_ini]
        frame_num_ini = frame_fea_ini.shape[0]
        duration_ini = vid2duration[vid_ini]
        frame_ini = get_mid_frame(st_ini, ed_ini, duration_ini, frame_num_ini)
        # assert ed_ini <= duration_ini, f"INITIALIZATION: {ed_ini}/{duration_ini}"
        assert frame_ini <= frame_num_ini, f"INITIALIZATION: {frame_ini}/{frame_num_ini}"

        video_pool = build_video_pool(pred_vids, vid_tar, vid_ini)
        unit_fea = get_subtitle_level_unit_feature(video_db, video_pool, feature_type, model)
        sim = calculate_similarity(unit_fea)

        tar_range = (frame_st_gt, frame_ed_gt)
        vertices, v_color, v_shape, static_edges, static_edges_color, static_edges_conf, sub_2_vid = \
            build_static_graph(video_db, video_pool, vid_tar, tar_range, vid_ini, frame_ini, vid_to_subs)
        vid_level_mask = pairwise_equation(np.array(sub_2_vid))
        vid_level_mask = torch.from_numpy(vid_level_mask).to(sim.device)
        nx_graph = build_network(vertices, static_edges)  # initialize the networkx graph

        edges_old = []  # old edge from last step
        for thd_cross in thds:  # threshold from high to low
            # cross video edges
            ind_cross = (sim > thd_cross) * torch.logical_not(vid_level_mask)
            edges_cross = ind_cross.nonzero().tolist()  # edges using unit similarity

            # inner video edges
            thd_inner = f_curve(thd_cross)
            ind_inner = (sim > thd_inner) * vid_level_mask
            edges_inner = ind_inner.nonzero().tolist()  # edges using unit similarity

            # add incremental edges
            edges = edges_cross + edges_inner
            edges_increment = [e for e in edges if e not in edges_old]  # edges to be incrementally added
            nx_graph.add_edges_from(edges_increment)

            # record (old) edges in last iteration
            edges_old = edges

            if plot_graph_mode and is_to_write:  # sample and plot graph
                # print("GRAPH-PLOT DESC_%s, THRESHOLD:%.2f |E|:%3d |V|:%3d |En|: %3d" % (desc_id, thd_cross, len(edges), len(vertices), len(static_edges)))

                # edge colors
                e_color = [EDGE_COLOR["default"]] * len(edges)

                # edge confidence
                e_conf = []
                e_conf += uniform_normalize(sim[ind_cross.nonzero(as_tuple=True)] - thd_cross * 0.9).tolist() if edges_cross else []
                e_conf += uniform_normalize(sim[ind_inner.nonzero(as_tuple=True)] - thd_inner * 0.9).tolist() if edges_inner else []

                if is_group_clique:
                    cliques = get_cliques(nx_graph)
                    vertices_grouped, v_color_grouped, v_shape_grouped, edges_grouped, e_color_grouped, e_conf_grouped = \
                        group_clique_nodes(
                            copy.deepcopy(cliques),
                            copy.deepcopy(vertices),
                            copy.deepcopy(v_color),
                            copy.deepcopy(v_shape),
                            static_edges + edges,
                            static_edges_color + e_color,
                            static_edges_conf + e_conf,
                            sub_2_vid)
                    v_plot, v_color_plot, v_shape_plot = vertices_grouped, v_color_grouped, v_shape_grouped
                    e_plot, e_color_plot, e_conf_plot = edges_grouped, e_color_grouped, e_conf_grouped

                else:
                    v_plot, v_color_plot, v_shape_plot = vertices, v_color, v_shape
                    e_plot, e_color_plot, e_conf_plot = static_edges + edges, static_edges_color + e_color, static_edges_conf + e_conf

                e_plot = list(map(tuple, e_plot))  # unify list and tuple coordinates into tuple

                if plot_shortest_path:
                    node_tar = get_tar_node(v_color)
                    node_src_hero = get_src_node(v_color)
                    sp_edges = shortest_path_edges(nx_graph, node_src_hero, node_tar)
                    e_color_plot, e_conf_plot = render_shortest_path(sp_edges, e_plot, e_color_plot, e_color_plot)

                plot_graph(v_plot,
                           e_plot,
                           markers=v_shape_plot,
                           v_colors=v_color_plot,
                           e_colors=e_color_plot,
                           confidence=e_conf_plot,
                           title=f"Graph of description {desc_id} with cross-θ {thd_cross:.2f} and inner-θ {thd_inner:.2f} . '{query_text}'",
                           fig_name=f"desc_graph/{sim_func}_{feature_type}/{group_folder_tag}desc_{desc_id}/desc_{desc_id}_tc_{thd_cross:.2f}_ti_{thd_inner:.2f}_graph_{shortest_path_tag}.png",
                           mute=False)

            if plot_sta_mode:  # build network and analyze statistic
                # nx_graph.add_edges_from(edges_increment)
                node_tar = get_tar_node(v_color)
                node_src_hero = get_src_node(v_color)
                node_src_rand = int(random.uniform(0, len(v_color)))
                dia = graph_diameter(nx_graph)
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
                           fig_name=f"desc_graph/"
                                    f"{sim_func}_{feature_type}/"
                                    f"{group_folder_tag}{opts.split}_diameter{toy_tag}/"
                                    f"graph_diameter_hist_tc_{thd_cross:.2f}_ti_{thd_inner:.2f}_p{p:.2f}_{b_dia}.png")

            list_histogram(dis_rand_count[thd_cross],
                           x_label="Shortest Distance (Random)",
                           fig_name=f"desc_graph/"
                                    f"{sim_func}_{feature_type}/"
                                    f"{group_folder_tag}{opts.split}_distance_random{toy_tag}/"
                                    f"graph_distance_hist_tc_{thd_cross:.2f}_ti_{thd_inner:.2f}_p{p:.2f}_{b_dis}.png")

            list_histogram(dis_hero_count[thd_cross],
                           x_label="Shortest Distance (HERO)",
                           fig_name=f"desc_graph/"
                                    f"{sim_func}_{feature_type}/"
                                    f"{group_folder_tag}{opts.split}_distance_hero_vcmr{toy_tag}/"
                                    f"graph_distance_hist_tc_{thd_cross:.2f}_ti_{thd_inner:.2f}_{b_dis}.png")


def process_vcmr_based_query_graph(model, data_loader, desc_to_video_pools, desc_to_moment_pools, opts, model_opts):
    model.eval()

    # super parameters
    feature_types = ["v", "vt"]
    feature_type = feature_types[1]
    modes = ["vis graph", "vis sta", "toy sta", "no io", "preprocess"]
    mode = modes[-1]
    plot_graph_mode, plot_sta_mode, sample_mode, ivcml_mode = set_params(mode)
    plot_shortest_path = True
    plot_ev_sta_mode = True
    shortest_path_tag = "_sp" if plot_shortest_path else ""
    toy_tag = "_toy" if (mode == "toy sta") else ""

    pprint.pprint({
        "mode": mode,
    })

    # start process
    try:
        is_to_write = hvd.rank() == 0
    except ValueError:
        is_to_write = True

    if sample_mode:
        desc_to_video_pools = sample_dict(desc_to_video_pools, 10, seed=2)

    hero_fea_reader = HEROUnitFeaLMDB(opts.output_dir, feature_type, opts.split, "unit_fea", readonly=True)
    query_data = data_loader.dataset.query_data
    video_db = data_loader.dataset.video_db
    v_id_to_duration = load_video2duration(opts.split)
    vid_to_frame_num = video_db.img_db.name2nframe
    max_frame_num = video_db.img_db.max_clip_len
    vid_to_frame_num = {k: min(vid_to_frame_num[k], max_frame_num) for k in vid_to_frame_num}
    video_db.img_db.name2nframe = vid_to_frame_num

    moment_proposal_size = max([len(desc_to_moment_pools[k]) for k in desc_to_moment_pools])

    dia_count = []
    dis_rand_count = []
    dis_hero_count = []
    v_count, e_count = [], []
    exceed_id = []

    state_idx, tar_bias = [], []
    neis_idx = []
    nei_adj_temporal_coo, nei_adj_mom_coo = [], []
    edge_to_remove = []
    for desc_idx, desc_id in tqdm(enumerate(desc_to_video_pools), desc="Processing Moment", total=len(desc_to_video_pools)):
        # if desc_idx == 3:
        #     desc_id = "91022"
        print("\nDESC_%s" % desc_id)
        query_item = query_data[desc_id]
        query_text = query_item["desc"]
        vid_tar = query_item["vid_name"]

        # Target moment information
        frame_num_tar = vid_to_frame_num[vid_tar]
        st_tar, ed_tar = query_item["ts"]
        duration_tar = v_id_to_duration[vid_tar]
        frame_st_gt, frame_ed_gt = get_frame_range(st_tar, ed_tar, duration_tar, frame_num_tar)
        assert ed_tar <= duration_tar, f"TARGET: {ed_tar}/{duration_tar}"

        # Initialization Video Information
        vid_ini, st_ini, ed_ini, score_ini = desc_to_moment_pools[desc_id][0]
        frame_num_ini = vid_to_frame_num[vid_ini]
        duration_ini = v_id_to_duration[vid_ini]
        frame_ini = get_mid_frame(st_ini, ed_ini, duration_ini, frame_num_ini)
        # assert ed_ini <= duration_ini, f"INITIALIZATION: {ed_ini}/{duration_ini}"
        assert frame_ini <= frame_num_ini, f"INITIALIZATION: {frame_ini}/{frame_num_ini}"

        video_pool, moment_pool, is_modified = complete_video_and_moment_pool(desc_to_video_pools[desc_id], desc_to_moment_pools[desc_id], vid_tar, duration_tar)
        modified_tag = "*" if is_modified else ""

        tar_range = (frame_st_gt, frame_ed_gt)
        vertices, v_color, v_shape, edges, e_color, e_conf, sub_2_vid, frame_to_unit = \
            build_static_graph(video_db, video_pool, vid_tar, tar_range, vid_ini, frame_ini)

        if len(vertices) > 1000:
            exceed_id.append((desc_id, len(vertices)))
            continue

        edges_vcmr, e_color_vcmr, e_conf_vcmr = \
            build_vcmr_edges(moment_pool, frame_to_unit, v_id_to_duration, vid_to_frame_num, frame_interval=model_opts.vfeat_interval)

        nx_graph = build_network(vertices, edges + edges_vcmr)  # initialize the networkx graph

        edges, e_color, e_conf = edges + edges_vcmr, e_color + e_color_vcmr, e_conf + e_conf_vcmr
        edges = list(map(tuple, edges))  # unify list and tuple coordinates into tuple
        v_count.append(len(vertices))
        e_count.append(len(edges))

        # adj_mat = build_sparse_adjacent_matrix(list(nx_graph.edges), nx_graph.number_of_nodes(), undirectional=True)  # directed adjacency matrix
        coo_adj_zero, coo_mom = compress_adj_mat(nx_graph)

        if ivcml_mode:
            node_tar = get_tar_node(v_color)
            node_src_hero = get_src_node(v_color)
            if hero_fea_reader:
                unit_fea = hero_fea_reader[video_pool]
            else:
                unit_fea = get_subtitle_level_unit_feature(video_db, video_pool, feature_type, model)

            # states_node_idx_batch, tar_idx_batch, nei_node_id_batch, nei_adj_temporal_coo_batch, nei_adj_mom_coo_batch = \
            #     ivcml_preprocessing(nx_graph, node_src_hero, node_tar, unit_fea)
            # nei_adj_temporal_coo += nei_adj_temporal_coo_batch
            # nei_adj_mom_coo += nei_adj_mom_coo_batch

            states_node_idx_batch, tar_idx_batch, nei_node_id_batch, nei_edge_to_remove_batch = \
                ivcml_preprocessing(nx_graph, node_src_hero, node_tar)
            edge_to_remove += nei_edge_to_remove_batch

            state_idx += states_node_idx_batch
            tar_bias += tar_idx_batch
            neis_idx += nei_node_id_batch

            # print("coo_adj_zero:", coo_adj_zero)
            # print("coo_mom:", coo_mom)
            # print("states_node_idx_batch", states_node_idx_batch)
            # print("tar_idx_batch", tar_idx_batch)
            # print("nei_node_id_batch", nei_node_id_batch)
            # print("nei_edge_to_remove_batch", nei_edge_to_remove_batch)
            # exit()

            del unit_fea
            del states_node_idx_batch, tar_idx_batch
            del nei_node_id_batch
            del nei_edge_to_remove_batch
            # del nei_adj_temporal_coo_batch, nei_adj_mom_coo_batch

        assert len(state_idx) == len(tar_bias) == len(neis_idx) == len(edge_to_remove), (len(state_idx), len(tar_bias), len(neis_idx), len(nei_adj_temporal_coo))

        # assert len(state_idx) == len(tar_bias) == len(neis_idx) == len(nei_adj_temporal_coo), (len(state_idx), len(tar_bias), len(neis_idx), len(nei_adj_temporal_coo))
        # for _state_idx, _tar_bias, _neis_idx, _adj_coo_temp, _adj_coo_mom in zip(state_idx, tar_bias, neis_idx, nei_adj_temporal_coo, nei_adj_mom_coo):
        #     print(_state_idx, _tar_bias, _neis_idx)
        #     for i in _adj_coo_temp:
        #         print(" "*4, i)

        if plot_graph_mode and is_to_write:  # sample and plot graph
            if plot_shortest_path:
                node_tar = get_tar_node(v_color)
                node_src_hero = get_src_node(v_color)
                sp_edges_hero = shortest_path_edges(nx_graph, node_src_hero, node_tar)
                if sp_edges_hero is not None:
                    e_color, e_conf = render_shortest_path(sp_edges_hero, edges, e_color, e_conf)

            plot_graph(vertices,
                       edges,
                       markers=v_shape,
                       v_colors=v_color,
                       e_colors=e_color,
                       confidence=e_conf,
                       title=f"VCMR-based graph of query {desc_id}: '{query_text}'",
                       fig_name=f"desc_graph/vcmr_based/desc_{desc_id}_graph{modified_tag}{shortest_path_tag}.png",
                       mute=False)

        if plot_sta_mode:  # build network and analyze statistic
            # nx_graph.add_edges_from(edges_increment)
            node_tar = get_tar_node(v_color)
            node_src_hero = get_src_node(v_color)
            node_src_rand = int(random.uniform(0, len(v_color)))
            dia = graph_diameter(nx_graph)
            dis_hero = graph_shortest_distance(nx_graph, node_src_hero, node_tar)
            dis_rand = graph_shortest_distance(nx_graph, node_src_rand, node_tar)

            print("DESC_%s, RANDOM: %2d/%2d; HERO: %2d/%2d |E|:%3d |V|:%3d |En|: %3d" % (desc_id, dis_rand, dia, dis_hero, dia, len(edges_vcmr), len(vertices), len(edges)))
            dia_count.append(dia)
            dis_hero_count.append(dis_hero)
            dis_rand_count.append(dis_rand)

    if plot_sta_mode and is_to_write:  # global statistic
        p = 0.9
        x = percentile(dia_count, p)
        list_histogram(dia_count,
                       x_label="Diameter",
                       fig_name=f"desc_graph/vcmr_based/statistic_{moment_proposal_size}_proposal{toy_tag}/"
                                f"{opts.split}_graph_diameter_hist_p{p:.2f}_{x}.png")

        x = percentile(dis_rand_count, p)
        list_histogram(dis_rand_count,
                       x_label="Shortest Distance (Random)",
                       fig_name=f"desc_graph/vcmr_based/statistic_{moment_proposal_size}_proposal{toy_tag}/"
                                f"{opts.split}_graph_distance_random_hist_p{p:.2f}_{x}.png")

        x = percentile(dis_hero_count, p)
        list_histogram(dis_hero_count,
                       x_label="Shortest Distance (HERO)",
                       fig_name=f"desc_graph/vcmr_based/statistic_{moment_proposal_size}_proposal{toy_tag}/"
                                f"{opts.split}_graph_distance_hist_hero_p{p:.2f}_{x}.png")

    if plot_ev_sta_mode and is_to_write:
        p = 0.9
        x = percentile(v_count, p)
        list_histogram(v_count,
                       x_label="Number of Vertices (moment based)",
                       fig_name=f"desc_graph/vcmr_based/statistic_{moment_proposal_size}_proposal{toy_tag}/"
                                f"{opts.split}_graph_vertices_num_hist_p{p:.2f}_{x}.png")

        x = percentile(e_count, p)
        list_histogram(e_count,
                       x_label="Number of Edges (moment based)",
                       fig_name=f"desc_graph/vcmr_based/statistic_{moment_proposal_size}_proposal{toy_tag}/"
                                f"{opts.split}_graph_edges_num_hist_p{p:.2f}_{x}.png")

    print("The following (description id, number of vertices) pairs contains too many (> 1000) vertices:\n", exceed_id)


def get_desc_video_pool(vcmr):
    # return {desc_id: list({_[0] for _ in vcmr[desc_id]}) for desc_id in vcmr}  # _vid, _st, _ed, _score
    desc_to_vid_pools = {desc_id: [] for desc_id in vcmr}  # _vid, _st, _ed, _score
    for desc_id in vcmr:
        for moment in vcmr[desc_id]:
            if moment[0] not in desc_to_vid_pools[desc_id]:
                desc_to_vid_pools[desc_id].append(moment[0])
    return desc_to_vid_pools


def dump_hero_unit_fea_lmdb(model, data_loader, data_dir, fea_type, split):
    model.eval()

    video_db = data_loader.dataset.video_db
    hero_fea_writer = HEROUnitFeaLMDB(data_dir, fea_type, split, tag="unit_fea", encode_method="ndarray", readonly=False)
    video_ids = list(video_db.vid2idx[split].keys())

    keys, values = [], []
    buffer = 2180
    for i, v_id in tqdm(enumerate(video_ids), total=len(video_ids), desc="Calculating node feature"):
        fea_unit = get_subtitle_level_unit_feature(video_db, [v_id], fea_type, model)  # torch.tensor with shape [unit_num, d]
        keys.append(v_id)
        values.append(fea_unit.detach().cpu().numpy())

        if (i % buffer == 0 and i != 0) or (i == len(video_ids) - 1):
            hero_fea_writer.put(keys, values)
            keys, values = [], []


def dump_node_observe_feature(model, data_loader, desc_to_video_pools, desc_to_moment_pools, opts, model_opts):
    model.eval()

    # super parameters
    feature_type = "vt"
    sample_mode = False

    pprint.pprint({
        "sample_mode": sample_mode,
        "feature_type": feature_type
    })

    if sample_mode:
        desc_to_video_pools = sample_dict(desc_to_video_pools, 10, seed=2)

    hero_fea_reader = HEROUnitFeaLMDB(opts.output_dir, feature_type, opts.split, tag="unit_fea", encode_method="ndarray", readonly=True)
    node_observe_vec_writer = \
        HEROUnitFeaLMDB(opts.output_dir, feature_type, opts.split, tag="observe_range_ALPHA_%f_STEPS_%d" % (opts.alpha, opts.observe_steps), encode_method="ndarray", readonly=False)

    query_data = data_loader.dataset.query_data
    video_db = data_loader.dataset.video_db
    v_id_to_duration = load_video2duration(opts.split)
    vid_to_frame_num = video_db.img_db.name2nframe
    max_frame_num = video_db.img_db.max_clip_len
    vid_to_frame_num = {k: min(vid_to_frame_num[k], max_frame_num) for k in vid_to_frame_num}
    video_db.img_db.name2nframe = vid_to_frame_num

    exceed_id = []
    path_len = []
    nei_nums = []
    adj_buff, vec_buff, key_buff = [], [], []
    for desc_idx, desc_id in tqdm(enumerate(desc_to_video_pools), desc="Processing Moment", total=len(desc_to_video_pools)):
        # print("\nDESC_%s" % desc_id)
        query_item = query_data[desc_id]
        vid_tar = query_item["vid_name"]

        # Target moment information
        frame_num_tar = vid_to_frame_num[vid_tar]
        st_tar, ed_tar = query_item["ts"]
        duration_tar = v_id_to_duration[vid_tar]
        frame_st_gt, frame_ed_gt = get_frame_range(st_tar, ed_tar, duration_tar, frame_num_tar)
        assert ed_tar <= duration_tar, f"TARGET: {ed_tar}/{duration_tar}"

        # Initialization Video Information
        vid_ini, st_ini, ed_ini, score_ini = desc_to_moment_pools[desc_id][0]
        frame_num_ini = vid_to_frame_num[vid_ini]
        duration_ini = v_id_to_duration[vid_ini]
        frame_ini = get_mid_frame(st_ini, ed_ini, duration_ini, frame_num_ini)
        assert frame_ini <= frame_num_ini, f"INITIALIZATION: {frame_ini}/{frame_num_ini}"

        video_pool, moment_pool, is_modified = complete_video_and_moment_pool(desc_to_video_pools[desc_id], desc_to_moment_pools[desc_id], vid_tar, duration_tar)
        tar_range = (frame_st_gt, frame_ed_gt)
        vertices, v_color, v_shape, edges, e_color, e_conf, sub_2_vid, frame_to_unit = \
            build_static_graph(video_db, video_pool, vid_tar, tar_range, vid_ini, frame_ini)

        if len(vertices) > 1000:
            exceed_id.append((desc_id, len(vertices)))
            continue

        edges_vcmr, e_color_vcmr, e_conf_vcmr = \
            build_vcmr_edges(moment_pool, frame_to_unit, v_id_to_duration, vid_to_frame_num, frame_interval=model_opts.vfeat_interval)

        nx_graph = build_network(vertices, edges + edges_vcmr)  # initialize the networkx graph

        node_tar = get_tar_node(v_color)
        node_src_hero = get_src_node(v_color)
        if hero_fea_reader:
            unit_fea = hero_fea_reader[video_pool]
        else:
            unit_fea = get_subtitle_level_unit_feature(video_db, video_pool, feature_type, model)

        sp = shortest_path_edges(nx_graph, node_src_hero, node_tar)
        path_len.append(len(sp))

        for _st, _ed in sp:
            state, ground_truth = _st, _ed
            nei_num = len(nx_graph.adj[state].keys()) + 1
            nei_nums.append(nei_num)

        adj_batch, vec_batch, key_batch = ivcml_node_observe_feature(desc_id, nx_graph, node_src_hero, node_tar, unit_fea)
        adj_buff += adj_batch
        vec_buff += vec_batch
        key_buff += key_batch

        buff_size = len(adj_buff)
        if buff_size >= 1024:
            node_size_buff = [mat.shape[0] for mat in adj_buff]
            max_node_size = max(node_size_buff)

            device = torch.device("cuda")
            dtype = adj_buff[0].dtype
            adj_buff_tensor = torch.zeros((buff_size, max_node_size, max_node_size), dtype=dtype, device=device)
            vec_buff_tensor = torch.zeros((buff_size, max_node_size), dtype=dtype, device=device)
            for i in range(buff_size):
                node_size = node_size_buff[i]
                adj_buff_tensor[i, :node_size, :node_size] = adj_buff[i]
                vec_buff_tensor[i, :node_size] = vec_buff[i]

            w_buff = aggregate_batch(adj_buff_tensor, vec_buff_tensor, opts.alpha, opts.observe_steps, norm=True)
            val_batch = w_buff

            for i in range(buff_size):
                key = key_buff[i]
                # if key == "96960_187_465":
                #     print("Found it: 96960_187_465")
                #     exit()
                node_size = node_size_buff[i]
                val = val_batch[i][:node_size]
                node_observe_vec_writer.put([key], [val.detach().cpu().numpy()])

            del adj_buff, vec_buff, key_buff
            adj_buff, vec_buff, key_buff = [], [], []


def dump_ivcml_query_input(model, data_loader, desc_to_video_pools, desc_to_moment_pools, opts, model_opts):
    model.eval()

    # super parameters
    feature_type = "vt"
    sample_mode = False
    ivcml_mode = False

    pprint.pprint({
        "sample_mode": sample_mode,
        "ivcml_mode": ivcml_mode,
        "feature_type": feature_type
    })

    if sample_mode:
        desc_to_video_pools = sample_dict(desc_to_video_pools, 10, seed=2)

    hero_fea_reader = HEROUnitFeaLMDB(opts.output_dir, feature_type, opts.split, tag="unit_fea", encode_method="ndarray", readonly=True)
    ivcml_input_data_writer = HEROUnitFeaLMDB(opts.output_dir, feature_type, opts.split, tag="input", encode_method="json", readonly=False)

    query_data = data_loader.dataset.query_data
    video_db = data_loader.dataset.video_db
    v_id_to_duration = load_video2duration(opts.split)
    vid_to_frame_num = video_db.img_db.name2nframe
    max_frame_num = video_db.img_db.max_clip_len
    vid_to_frame_num = {k: min(vid_to_frame_num[k], max_frame_num) for k in vid_to_frame_num}
    video_db.img_db.name2nframe = vid_to_frame_num

    exceed_id = []
    path_len = []
    nei_nums = []
    for desc_idx, desc_id in tqdm(enumerate(desc_to_video_pools), desc="Processing Moment", total=len(desc_to_video_pools)):
        print("\nDESC_%s" % desc_id)
        query_item = query_data[desc_id]
        vid_tar = query_item["vid_name"]
        desc = query_item["desc"]

        # Target moment information
        frame_num_tar = vid_to_frame_num[vid_tar]
        st_tar, ed_tar = query_item["ts"]
        duration_tar = v_id_to_duration[vid_tar]
        frame_st_gt, frame_ed_gt = get_frame_range(st_tar, ed_tar, duration_tar, frame_num_tar)
        assert ed_tar <= duration_tar, f"TARGET: {ed_tar}/{duration_tar}"

        # Initialization Video Information
        vid_ini, st_ini, ed_ini, score_ini = desc_to_moment_pools[desc_id][0]
        frame_num_ini = vid_to_frame_num[vid_ini]
        duration_ini = v_id_to_duration[vid_ini]
        frame_ini = get_mid_frame(st_ini, ed_ini, duration_ini, frame_num_ini)
        assert frame_ini <= frame_num_ini, f"INITIALIZATION: {frame_ini}/{frame_num_ini}"

        video_pool, moment_pool, is_modified = complete_video_and_moment_pool(desc_to_video_pools[desc_id], desc_to_moment_pools[desc_id], vid_tar, duration_tar)
        tar_range = (frame_st_gt, frame_ed_gt)
        vertices, v_color, v_shape, edges, e_color, e_conf, sub_2_vid, frame_to_unit = \
            build_static_graph(video_db, video_pool, vid_tar, tar_range, vid_ini, frame_ini)

        if len(vertices) > 1000:
            exceed_id.append((desc_id, len(vertices)))
            continue

        edges_vcmr, e_color_vcmr, e_conf_vcmr = \
            build_vcmr_edges(moment_pool, frame_to_unit, v_id_to_duration, vid_to_frame_num, frame_interval=model_opts.vfeat_interval)

        nx_graph = build_network(vertices, edges + edges_vcmr)  # initialize the networkx graph
        coo_adj_zero, coo_mom = compress_adj_mat(nx_graph)

        node_tar = get_tar_node(v_color)
        node_src_hero = get_src_node(v_color)
        if hero_fea_reader:
            unit_fea = hero_fea_reader[video_pool]
        else:
            unit_fea = get_subtitle_level_unit_feature(video_db, video_pool, feature_type, model)

        sp = shortest_path_edges(nx_graph, node_src_hero, node_tar)
        path_len.append(len(sp))

        for _st, _ed in sp:
            state, ground_truth = _st, _ed
            nei_num = len(nx_graph.adj[state].keys()) + 1
            nei_nums.append(nei_num)

        states_node_idx_batch, tar_bias_batch, nei_node_idx_batch, nei_edge_to_remove_batch, neis_rewards_batch = \
            ivcml_preprocessing(nx_graph, node_src_hero, node_tar)

        data_sample = {
            "video_pool": video_pool,
            "num_node": nx_graph.number_of_nodes(),
            # "coo_adj_zero": coo_adj_zero,
            # "coo_mom": coo_mom,
            "nei_edge_to_remove": nei_edge_to_remove_batch,
            "state_node_idx": states_node_idx_batch,
            "nei_node_idx": nei_node_idx_batch,
            "tar_bias": tar_bias_batch,
            "desc": desc,
            "reward": neis_rewards_batch
        }

        ivcml_input_data_writer.put([desc_id], [data_sample])

        del unit_fea
        del coo_adj_zero, coo_mom
        # del states_node_idx_batch, tar_bias_batch
        # del nei_node_idx_batch
        # del nei_edge_to_remove_batch
        # del neis_rewards_batch

    # e9 = percentile(path_len, 0.9)
    # e5 = percentile(path_len, 0.5)
    # list_histogram(path_len, fig_name="dist_shortest_path_length_P_%.1f_LEN_%d_%.1f_LEN_%d.png" % (0.9, e9, 0.5, e5))
    # e9 = percentile(nei_nums, 0.9)
    # e5 = percentile(nei_nums, 0.5)
    # list_histogram(nei_nums, fig_name="dist_node_nei_num_P_%.1f_LEN_%d_%.1f_LEN_%d.png" % (0.9, e9, 0.5, e5))
    # print("Average neighbor number:", sum(nei_nums) / len(nei_nums))


def dump(opts):

    use_hvd = False
    if use_hvd:
        hvd.init()
        device = torch.device("cuda", hvd.local_rank())
        torch.cuda.set_device(hvd.local_rank())
    else:
        device = torch.device("cuda")

    model, model_opts = load_model(opts, device)
    data_loader = build_dataloader(opts)

    # Dump adjacent matrix
    desc_to_moment_pools = load_hero_pred(opts, task="vcmr")
    desc_to_video_pools = get_desc_video_pool(desc_to_moment_pools)

    # dump_ivcml_query_input(model, data_loader, desc_to_video_pools, desc_to_moment_pools, opts, model_opts)
    dump_node_observe_feature(model, data_loader, desc_to_video_pools, desc_to_moment_pools, opts, model_opts)


def main(opts):
    dump(opts)

    # use_hvd = False
    # if use_hvd:
    #     hvd.init()
    #     device = torch.device("cuda", hvd.local_rank())
    #     torch.cuda.set_device(hvd.local_rank())
    # else:
    #     device = torch.device("cuda")
    #
    # model, model_opts = load_model(opts, device)
    # data_loader = build_dataloader(opts)
    #
    # # Process and save LMDB
    # preprocess_hero_unit_fea_lmdb(model, data_loader, opts.output_dir, "vt", opts.split)
    #
    # # similarity-based ad-hoc graph
    # desc_to_video_pools = load_hero_pred(opts, task="vr")
    # desc_to_moment_pools = load_hero_pred(opts, task="vcmr_base_on_vr")
    # data_loader = build_dataloader(opts)
    # model, model_opts = load_model(opts, device)
    # # vid_sub_range = load_subtitle_range(data_loader, opts)
    # vid_to_subs = load_tvr_subtitle()
    # process_ad_hoc_query_graph(model, data_loader, desc_to_video_pools, desc_to_moment_pools, vid_to_subs, opts, model_opts)
    #
    # # VCMR based graph
    # desc_to_moment_pools = load_hero_pred(opts, task="vcmr")
    # desc_to_video_pools = get_desc_video_pool(desc_to_moment_pools)
    # process_vcmr_based_query_graph(model, data_loader, desc_to_video_pools, desc_to_moment_pools, opts, model_opts)
    #
    # # graph statistic visualization
    # moment_num = [len(desc_to_moment_pools[k]) for k in desc_to_moment_pools]
    # p = 0.9
    # x = percentile(moment_num, p)
    # list_histogram(moment_num, title="Number of top-100 retrieved moments using HERO.", fig_name=f"dist_mom_num_top_100_p{p:.2f}_{x}.png")
    #
    # video_num = [len(desc_to_video_pools[k]) for k in desc_to_video_pools]
    # x = percentile(video_num, p)
    # list_histogram(video_num, title="Number of videos in top-100 retrieved moments using HERO.", fig_name=f"dist_vid_num_top_100_p{p:.2f}_{x}.png")


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
    parser.add_argument("--full_eval_tasks", type=str, nargs="+", choices=["VCMR", "SVMR", "VR"], default=["VCMR", "SVMR", "VR"], help="Which tasks to run. VCMR: Video Corpus Moment Retrieval; SVMR: "
                                                                                                                                        "(will be performed automatically with VCMR)")
    parser.add_argument("--output_dir", default=None, type=str, help="The output directory where the model checkpoints will be written.")

    # device parameters
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4, help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true', help="pin memory")

    parser.add_argument("--alpha", default=0.99, type=float, help="decay rate")
    parser.add_argument("--observe_steps", default=20, type=int, help="The number of steps which a node can observe")

    args = parser.parse_args()

    # options safe guard
    # TODO

    main(args)
