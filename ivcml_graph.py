from matplotlib import pyplot as plt
from ivcml_util import save_plt
import networkx as nx
import random
import numpy as np
from ivcml_util import pairwise_equation, expand_window, mean, random_color, is_overlap, subtract


NODE_COLOR = {"target": "orange",
              "default": "green",
              "ini": "cyan",
              "ini_and_tar": "yellow"}

COLOR = {
    "dark red": "#CD2424",
    "light green": "#9FF689"
}


# Graph function
def build_static_graph(video_db, vid_to_subs, vid_pool, tar_vid, range_gt, ini_vid, frame_idx_ini):
    def get_node_color(_vid, _range, _vid_tar, _range_gt, _ini_vid, _frame_ini):
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
            _color = NODE_COLOR["ini_and_tar"]
        elif is_part_of_target:
            _color = NODE_COLOR["target"]
        elif is_ini:
            _color = NODE_COLOR["ini"]
        else:
            _color = NODE_COLOR["default"]

        return _color

    def get_node_shape(_idx):
        if _idx is None:
            return "^"  # triangle for visual only nodes
        return "o"  # circle for visual + text

    vertices, v_color, v_shape = [], [], []
    boundary_indicator = []
    sub_2_vid = []
    for i, vid in enumerate(vid_pool):
        _, _, _, frame_fea_cur, _, _, sub_idx2frame_idx = video_db[vid]
        frame_num_cur = frame_fea_cur.shape[0]
        sub_idx2frame_idx = complement_sub_unit(sub_idx2frame_idx, frame_num_cur)
        for j, (sub_idx, frame_range) in enumerate(sub_idx2frame_idx):
            sub_2_vid.append(vid)
            vertices.append([j * 3, i * 5])
            color = get_node_color(vid, frame_range, tar_vid, range_gt, ini_vid, frame_idx_ini)
            shape = get_node_shape(sub_idx)

            # if vid == ini_vid:
            #     if not j:
            #         print("CURRENT VID:", vid, (vid == ini_vid))
            #     if frame_range:
            #         print(f"{str(sub_idx):4}", f"({frame_range[0]:3}", f"{frame_range[-1]:3})", f"-> {frame_idx_ini:3}", f" max: {frame_num_cur}", vid == ini_vid, color)
            #     else:
            #         print(f"EMPTY", f"-> {frame_idx_ini:3}", f" max: {frame_num_cur}", vid == ini_vid, color)
            #
            # if vid == tar_vid:
            #     if not j:
            #         print("CURRENT VID:", vid, (vid == tar_vid))
            #     if frame_range:
            #         print(f"{str(sub_idx):4}", f"({frame_range[0]:3}", f"{frame_range[-1]:3})", f"-> ({range_gt[0]:3}", f"{range_gt[1]:3})", f" max: {frame_num_cur}", vid == tar_vid, color)
            #     else:
            #         print(f"EMPTY", f"{range_gt[0]:3}", f"{range_gt[1]:3}", f" max: {frame_num_cur}", vid == tar_vid, color)

            v_color.append(color)
            v_shape.append(shape)
            boundary_indicator.append(1)
        boundary_indicator[-1] = 0

    edges_subj = build_subject_edge(vid_pool, vid_to_subs)
    edges_subj_color = [COLOR["light green"]] * len(edges_subj)  # a green
    edges_subj_conf = [0.2] * len(edges_subj)

    temporal_edges = [(node_i, node_i+1) for node_i, is_not_boundary in enumerate(boundary_indicator) if is_not_boundary]
    # temporal_edges_color = ["blue" if st % 2 else "magenta" for st, ed in temporal_edges]
    temporal_edges_color = [COLOR["dark red"]] * len(temporal_edges)
    temporal_edges_conf = [1] * len(temporal_edges)

    static_edges = []
    static_edges_color = []
    static_edges_conf = []

    static_edges += temporal_edges
    static_edges_color += temporal_edges_color
    static_edges_conf += temporal_edges_conf

    # static_edges += edges_subj
    # static_edges_color += edges_subj_color
    # static_edges_conf += edges_subj_conf

    print("|E_subject|: %d" % len(edges_subj))

    return vertices, v_color, v_shape, static_edges, static_edges_color, static_edges_conf, sub_2_vid


def build_subject_edge(vid_pool, vid_to_subs):
    def parse_subtitle_subject(subtitle):
        if ":" in subtitle:
            return subtitle.split(":")[0].strip().lower()
        return "None"

    cur_sub = "None"
    subtitle_subjects = []
    for i, vid in enumerate(vid_pool):
        subs = vid_to_subs[vid]
        for sub in subs:
            subj = parse_subtitle_subject(sub['text'])
            if subj != "None" and len(subj) < 20:
                cur_sub = subj  # update if new subj appears
            subtitle_subjects.append(cur_sub)
    subtitle_subjects = expand_window(subtitle_subjects, window_set=[-2, -1, 0, 1, 2])
    subtitle_subjects = ["%s "*len(i) % i for i in zip(*subtitle_subjects)]
    edges = []
    eq_mat = pairwise_equation(np.array(subtitle_subjects), tok_illegal="None")
    eq_mat = np.tril(eq_mat, -2)
    for x, y in zip(*eq_mat.nonzero()):
        edges.append((x, y))
    return edges


def complement_sub_unit(sub_idx2frame_idx, frame_num):
    """
    :param sub_idx2frame_idx: the mapping from subtitle indices to their corresponding frame range
    :param frame_num: the total number of frames in this video
    :return: the complemented frame range of the whole video (add the intervals without any subtitles matched)
    """
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

    if frame_num > old_ed:
        extended_sub_level_units.append((None, list(range(old_ed+1, frame_num))))

    return extended_sub_level_units


def group_clique_nodes(cliques, vertices, v_color, v_shape, edges, e_color, e_conf, sub_2_vid):
    v_redirect = [i for i in range(len(vertices))]
    for clique in cliques:
        if pairwise_equation(np.array([sub_2_vid[i] for i in clique])).sum() == (len(clique) * (len(clique) - 1) / 2):
            center = random.choice(clique)

            # center coordinates
            x_center = mean([vertices[v][0] for v in clique])
            y_center = mean([vertices[v][1] for v in clique]) + random.uniform(0.8, 1.2)
            vertices[center] = (x_center, y_center)

            # center shape
            v_shape[center] = "*"

            # center color
            clique_colors = [v_color[v] for v in clique]
            is_ini = NODE_COLOR["ini"] in clique_colors
            is_tar = NODE_COLOR["target"] in clique_colors
            is_tar_and_ini = NODE_COLOR["ini_and_tar"] in clique_colors
            if is_tar_and_ini:
                center_color = NODE_COLOR["ini_and_tar"]
            elif is_ini:
                center_color = NODE_COLOR["ini"]
            elif is_tar:
                center_color = NODE_COLOR["target"]
            else:
                center_color = v_color[center]
            v_color[center] = center_color

            # other nodes in the clique
            group_color = "white"  # nodes to remove
            for v in clique:
                if v == center:
                    continue
                v_color[v] = group_color
                v_redirect[v] = center

    all_edges = [(v_redirect[v_st], v_redirect[v_ed]) for v_st, v_ed in edges]

    # Filtering
    _edges, _e_color, _e_conf = [], [], []
    for (v_st, v_ed), color, conf in zip(all_edges, e_color, e_conf):
        if ((v_st, v_ed) not in _edges) and (v_st != v_ed):
            assert v_color[v_st] != "white" and v_color[v_ed] != "white", "(%s -> %s)" % (v_color[v_st], v_color[v_ed])
            _edges.append((v_st, v_ed))
            _e_color.append(color)
            _e_conf.append(conf)
    return vertices, v_color, v_shape, _edges, _e_color, _e_conf


def plot_graph(v, e, markers=None, v_colors=None, e_colors=None, confidence=None, title="Example graph.", fig_name="graph.png", mute=False, full_edge=False):
    """
    :param v: list of vertices [(x1, y1), ...]
    :param e: list of edges [(i, j), ...]
    :param markers: shapes of vertices
    :param v_colors: colors of vertices
    :param e_colors: colors of edges
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
        color = e_colors[i] if e_colors is not None else "red"

        if y[st] != y[ed] or (st == ed-1 and y[st] == y[ed]):  # lines with overlap
            if st == ed-1 and y[st] == y[ed]:
                continue
            ax.plot([x[st], x[ed]], [y[st], y[ed]], color=color, marker='', alpha=p)
        else:
            tmp_x, tmp_y = (x[st] + x[ed]) / 2, y[st] + random.uniform(1, 2)
            ax.plot([x[st], tmp_x], [y[st], tmp_y], color=color, marker='', alpha=p)
            ax.plot([tmp_x, x[ed]], [tmp_y, y[ed]], color=color, marker='', alpha=p)

    for i, v in enumerate(v):
        color = "green" if v_colors is None else v_colors[i]
        marker = "o" if markers is None else markers[i]
        ax.scatter(v[0], v[1], color=color, marker=marker, s=30)

    title = ax.set_title("Fig. " + title, fontdict={'family': 'serif', "verticalalignment": "bottom"}, loc='center', wrap=True)
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    save_plt(fig_name, mute)
    plt.clf()
    plt.close()


# NetworkX functions
def get_src_node(nodes_color):
    if "cyan" in nodes_color:
        node_src_hero = nodes_color.index("cyan")
    else:
        node_src_hero = nodes_color.index("yellow")
    return node_src_hero


def get_tar_node(nodes_color):
    if "orange" in nodes_color:
        node_tar_hero = nodes_color.index("orange")
    else:
        node_tar_hero = nodes_color.index("yellow")
    return node_tar_hero


def build_network(vertices: list, edges: list) -> nx.Graph:
    g = nx.Graph()
    node_ids = list(range(len(vertices)))
    g.add_nodes_from(node_ids)
    g.add_edges_from(edges)
    return g


def graph_diameter(g: nx.Graph):
    """
    :param g: instance of NetworkX graph
    :return: diameter of g
    """
    try:
        dia = nx.algorithms.distance_measures.diameter(g)
    except nx.exception.NetworkXError:
        dia = -1
    return dia


def graph_shortest_distance(g: nx.Graph, src: int, tar: int):
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


def graph_shortest_path(g: nx.Graph, src: int, tar: int):
    """
    :param g: instance of NetworkX graph
    :param src: source node
    :param tar: target node
    :return: the shortest path from src to tar
    """
    try:
        sp = nx.algorithms.shortest_paths.generic.shortest_path(g, source=src, target=tar)
    except nx.exception.NetworkXNoPath:
        sp = None
    return sp


def get_cliques(g: nx.Graph):

    cliques = nx.algorithms.find_cliques(g)
    cliques = sorted(cliques, key=lambda x: len(x))
    unique_cliques = []
    while cliques:
        max_clique = cliques.pop()
        if len(max_clique) > 3:
            unique_cliques.append(max_clique)
        # cliques = [c for c in cliques if not is_overlap(c, max_clique)]  # incorrect logic
        cliques = [subtract(c, max_clique) for c in cliques]
        cliques = sorted(cliques, key=lambda x: len(x))
    return unique_cliques
