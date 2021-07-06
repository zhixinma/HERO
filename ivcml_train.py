import torch
import argparse
from tqdm import tqdm
import torch.optim as optim
import random
from sklearn.metrics import accuracy_score

from ivcml_model import DeepQNetIVCML
from ivcml_util import cuda
from ivcml_util import show_type_tree
from ivcml_util import stack_tenor_list
from ivcml_data import HEROUnitFeaLMDB
from ivcml_load import get_desc_input_batch
from ivcml_load import get_desc_input_batch_wo_aggregate


def train(opts):
    def batch_generator(_query_ids, _input_reader, _fea_reader, _node_observe_weight_reader, batch_size):
        data_size = len(_query_ids)
        batch_num = data_size // batch_size + int((data_size % batch_size) != 0)
        with tqdm(range(batch_num), total=batch_num) as t:
            for i_batch in t:
                t.set_description(desc="Training samples: ce_loss %.3f l1_loss: %.3f" % (ce_loss, l1_loss))
                st = i_batch * batch_size
                ed = st + batch_size
                batch_q_ids = _query_ids[st: ed]

                text_ids_batch = []
                nei_observe_weight_batch = []
                unit_fea_batch = []
                tar_mask_batch = []
                tar_bias_batch = []
                nei_reward_batch = []
                for q_id in batch_q_ids:
                    text_ids_case, nei_observe_weight_case, unit_fea_case, tar_mask_case, tar_bias_case, nei_reward_case = \
                        get_desc_input_batch_wo_aggregate(q_id, _input_reader, _fea_reader, _node_observe_weight_reader)
                    text_ids_batch.append(text_ids_case)
                    nei_observe_weight_batch.append(nei_observe_weight_case)
                    unit_fea_batch.append(unit_fea_case)
                    tar_mask_batch.append(tar_mask_case)
                    tar_bias_batch.append(tar_bias_case)
                    nei_reward_batch.append(nei_reward_case)

                batch = [text_ids_batch, nei_observe_weight_batch, unit_fea_batch, tar_mask_batch, tar_bias_batch, nei_reward_batch]
                batch = [stack_tenor_list(data) for data in batch]

                yield i_batch, batch

    def iterate(batch):
        text_ids_batch, nei_observe_weight_batch, unit_fea_batch, nei_mask_batch, _cls_gold_batch, _reward_batch = cuda(batch, device)
        _move_pred_batch, _reward_pred_batch = dqn(text_ids_batch, nei_observe_weight_batch, unit_fea_batch, nei_mask_batch, _cls_gold_batch)
        # use mask to remove padded neighbors
        invalid_sample_idx = (1 - nei_mask_batch).nonzero(as_tuple=True)
        _move_pred_batch[invalid_sample_idx] = -1e5
        _reward_pred_batch[invalid_sample_idx] = 0
        # build mask to remove padded time steps
        step_mask_batch = (nei_mask_batch.sum(dim=-1, keepdim=False) > 0).to(nei_mask_batch.dtype)
        return _move_pred_batch, _reward_pred_batch, _cls_gold_batch, _reward_batch, step_mask_batch

    device = torch.device("cuda")
    feature_type = "vt"
    dqn = DeepQNetIVCML(d_fea=768, alpha=opts.alpha, step_num=opts.observe_steps)
    dqn.to(device)

    optimizer = optim.RMSprop(dqn.parameters())
    ce_loss_func = torch.nn.CrossEntropyLoss()
    l1_loss_func = torch.nn.SmoothL1Loss()

    input_reader = HEROUnitFeaLMDB(opts.output_dir, feature_type, opts.split, tag="input", encode_method="json", readonly=True)
    fea_reader = HEROUnitFeaLMDB(opts.output_dir, feature_type, opts.split, tag="unit_fea", encode_method="ndarray", readonly=True)
    node_observe_reader = HEROUnitFeaLMDB(opts.output_dir, feature_type, opts.split, tag="observe_range_ALPHA_%f_STEPS_%d" % (opts.alpha, opts.observe_steps), encode_method="ndarray", readonly=True)

    query_ids = input_reader.keys()

    # # sample
    # random.seed(10)
    # query_ids = random.sample(query_ids, 500)

    data_size_ttl = len(query_ids)
    boundary = int(data_size_ttl * 0.9)
    random.seed(1)
    random.shuffle(query_ids)
    query_ids_trn = query_ids[:boundary]
    query_ids_val = query_ids[boundary:]

    num_epoch = 10
    batch_size = 256
    evaluate_period = 1
    for i_epoch in range(num_epoch):
        print("Epoch %d" % i_epoch)
        dqn.train()
        ce_loss, l1_loss = 0, 0
        for i_batch, batch_data in batch_generator(query_ids_trn, input_reader, fea_reader, node_observe_reader, batch_size):
            move_pred_batch, reward_pred_batch, gold_batch, reward_batch, time_step_mask_batch = iterate(batch_data)

            # use mask to index valid sample
            valid_sample_idx = time_step_mask_batch.nonzero(as_tuple=True)
            move_pred_batch = move_pred_batch[valid_sample_idx]
            reward_pred_batch = reward_pred_batch[valid_sample_idx]
            gold_batch = gold_batch[valid_sample_idx]
            reward_batch = reward_batch[valid_sample_idx]

            if i_batch == 5:
                print(torch.softmax(move_pred_batch[0, :], dim=-1))
                print(gold_batch[0])
                print(reward_pred_batch[0, :])
                print(reward_batch[0, :])
                show_type_tree([move_pred_batch, gold_batch, reward_pred_batch, reward_batch])
                exit()

            ce_loss = ce_loss_func(move_pred_batch, gold_batch)
            l1_loss = l1_loss_func(reward_pred_batch, reward_batch)
            loss = ce_loss + l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i_epoch % evaluate_period == 0:
            dqn.eval()
            pred_val, gold_val = [], []
            for i_batch, batch_data in batch_generator(query_ids_val, input_reader, fea_reader, node_observe_reader, batch_size):
                move_pred_batch, reward_pred_batch, gold_batch, reward_batch, time_step_mask_batch = iterate(batch_data)
                valid_sample_idx = time_step_mask_batch.nonzero(as_tuple=True)

                move_pred_batch = move_pred_batch.argmax(dim=-1)
                move_pred_batch = move_pred_batch[valid_sample_idx]
                gold_batch = gold_batch[valid_sample_idx]
                pred_val.append(move_pred_batch)
                gold_val.append(gold_batch)

            pred_val = torch.cat(pred_val, dim=0)
            gold_val = torch.cat(gold_val, dim=0)
            acc = accuracy_score(gold_val.detach().cpu(), pred_val.detach().cpu())
            print("Epoch: %d Validation accuracy: %.3f" % (i_epoch, acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    # parser.add_argument("--sub_txt_db", default="/txt/tv_subtitles.db", type=str, help="The input video subtitle corpus. (LMDB)")
    # parser.add_argument("--vfeat_db", default="/video/tv", type=str, help="The input video frame features.")
    # parser.add_argument("--query_txt_db", default="/txt/tvr_val.db", type=str, help="The input test query corpus. (LMDB)")
    parser.add_argument("--split", choices=["val", "test_public", "test"], default="val", type=str, help="The input query split")
    # parser.add_argument("--task", choices=["tvr", "how2r", "didemo_video_sub", "didemo_video_only"], default="tvr", type=str, help="The evaluation vcmr task")
    # parser.add_argument("--checkpoint", default=None, type=str, help="pretrained model checkpoint steps")
    # parser.add_argument("--batch_size", default=80, type=int, help="number of queries in a batch")
    # parser.add_argument("--vcmr_eval_video_batch_size", default=50, type=int, help="number of videos in a batch")
    # parser.add_argument("--full_eval_tasks", type=str, nargs="+", choices=["VCMR", "SVMR", "VR"], default=["VCMR", "SVMR", "VR"], help="Which tasks to run. VCMR: Video Corpus Moment Retrieval; SVMR: "
    parser.add_argument("--output_dir", default=None, type=str, help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--alpha", default=0.99, type=float, help="decay rate")
    parser.add_argument("--observe_steps", default=20, type=int, help="The number of steps which a node can observe")

    # # device parameters
    # parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    # parser.add_argument('--n_workers', type=int, default=4, help="number of data workers")
    # parser.add_argument('--pin_mem', action='store_true', help="pin memory")
    parser.add_argument('--log', action='store_true', help="log mode")

    args = parser.parse_args()
    train(args)
