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


def load_video2duration(split="train"):
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


def build_ivcml_ground_truth():
    return
