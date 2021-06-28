import torch
import argparse
from tqdm import tqdm
import torch.optim as optim
import random
from sklearn.metrics import accuracy_score

from ivcml_model import DeepQNetIVCML
from ivcml_util import cuda
from ivcml_data import HEROUnitFeaLMDB
from ivcml_load import get_desc_input_batch


def train(opts):
    def iterate(_query_id, _input_reader, _fea_reader):
        batch = get_desc_input_batch(_query_id, _input_reader, _fea_reader)
        text_ids_batch, adj_mat_nei_batch, nei_vec_batch, unit_fea_batch, nei_mask_batch, _cls_gold_batch, _reward_batch = cuda(batch, device)
        _cls_pred_batch = dqn(text_ids_batch, adj_mat_nei_batch, nei_vec_batch, unit_fea_batch, nei_mask_batch)
        _cls_pred_batch = torch.where(nei_mask_batch > 0, _cls_pred_batch, nei_mask_batch)
        return _cls_pred_batch, _cls_gold_batch, _reward_batch

    device = torch.device("cuda")
    feature_type = "vt"
    dqn = DeepQNetIVCML(d_fea=768, alpha=0.99, step_num=20)
    dqn.to(device)

    optimizer = optim.RMSprop(dqn.parameters())
    ce_loss_func = torch.nn.CrossEntropyLoss()
    l1_loss_func = torch.nn.SmoothL1Loss()

    input_reader = HEROUnitFeaLMDB(opts.output_dir, feature_type, opts.split, tag="input", encode_method="json", readonly=True)
    fea_reader = HEROUnitFeaLMDB(opts.output_dir, feature_type, opts.split, tag="unit_fea", encode_method="ndarray", readonly=True)
    query_ids = input_reader.keys()

    # sample
    random.seed(10)
    query_ids = random.sample(query_ids, 500)

    data_size = len(query_ids)
    random.seed(1)
    query_ids_val = random.sample(query_ids, int(data_size*0.1))
    query_ids_trn = [q_id for q_id in query_ids if q_id not in query_ids_val]
    data_size_trn = len(query_ids_trn)

    num_epoch = 10
    evaluate_period = 1
    for i_epoch in range(num_epoch):
        print("Epoch %d" % i_epoch)
        dqn.train()
        ce_loss, l1_loss = 0, 0
        all_loss = 0
        with tqdm(query_ids_trn, total=len(query_ids_trn)) as t:
            for query_id in t:
                t.set_description(desc="Training samples: ce_loss %.3f l1_loss: %.3f" % (ce_loss, l1_loss))
                if opts.log:
                    print("QUERY:", query_id)
                pred_batch, gold_batch, reward_batch = iterate(query_id, input_reader, fea_reader)
                pred_batch = pred_batch.squeeze(-1)

                ce_loss = ce_loss_func(pred_batch, gold_batch)
                l1_loss = l1_loss_func(pred_batch, reward_batch)
                loss = ce_loss + l1_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if i_epoch % evaluate_period == 0:
            dqn.eval()
            pred_val, gold_val = [], []
            for query_id in tqdm(query_ids_val, total=len(query_ids_val), desc="Validating samples"):
                pred_batch, gold_batch, reward_batch = iterate(query_id, input_reader, fea_reader)
                pred_batch = pred_batch.squeeze(-1).argmax(dim=-1)
                pred_val.append(pred_batch)
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

    # # device parameters
    # parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    # parser.add_argument('--n_workers', type=int, default=4, help="number of data workers")
    # parser.add_argument('--pin_mem', action='store_true', help="pin memory")
    parser.add_argument('--log', action='store_true', help="log mode")

    args = parser.parse_args()
    train(args)
