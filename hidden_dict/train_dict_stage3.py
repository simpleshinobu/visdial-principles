import json
import argparse
import time
import datetime
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import yaml
from bisect import bisect
from torch.nn import functional as F
import random
import sys
sys.path.append('../')
from visdialch.data.dataset import VisDialDataset
from visdialch.encoders import Encoder
from visdialch.decoders import Decoder
from visdialch.metrics import SparseGTMetrics, NDCG
from visdialch.model import EncoderDecoderModel
from visdialch.utils.checkpointing import CheckpointManager, load_checkpoint
from visdialch.encoders.dict_encoder import Dict_Encoder
import os
os.chdir('../')
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-yml",
    default="hidden_dict/baseline_stage2.yml",
    help="Path to a config file listing reader, model and solver parameters.",
)
parser.add_argument(
    "--config-dict-yml",
    default="hidden_dict/dict2.yml",
    help="Path to a config file listing reader, model and solver parameters.",
)
parser.add_argument(
    "--train-json",
    default="data/visdial_1.0_train.json",
    help="Path to json file containing VisDial v1.0 training data.",
)
parser.add_argument(
    "--val-json",
    default="data/visdial_1.0_val.json",
    help="Path to json file containing VisDial v1.0 validation data.",
)
parser.add_argument(
    "--val-dense-json",
    default="data/visdial_1.0_val_dense_annotations.json",
    help="Path to json file containing VisDial v1.0 validation dense ground "
         "truth annotations.",
)
parser.add_argument_group(
    "Arguments independent of experiment reproducibility"
)
parser.add_argument(
    "--gpu-ids",
    nargs="+",
    type=int,
    default=[3],
    help="List of ids of GPUs to use.",
)
parser.add_argument(
    "--cpu-workers",
    type=int,
    default=4,
    help="Number of CPU workers for dataloader.",
)
parser.add_argument(
    "--overfit",
    action="store_true",
    help="Overfit model on 5 examples, meant for debugging.",
)
parser.add_argument(
    "--in-memory",
    action="store_true",
    help="Load the whole dataset and pre-extracted image features in memory. "
         "Use only in presence of large RAM, atleast few tens of GBs.",
)
parser.add_argument(
    "--load-pthpath",
    default='',
    help="To continue training, path to .pth file of saved checkpoint.",
)
parser.add_argument(
    "--load-dict-pthpath",
    default='',
    help="To continue training, path to .pth file of saved checkpoint.",
)


manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# =============================================================================
#   INPUT ARGUMENTS AND CONFIG
# =============================================================================

args = parser.parse_args()
# keys: {"dataset", "model", "solver"}
config = yaml.load(open(args.config_yml))
config_dict = yaml.load(open(args.config_dict_yml))

if isinstance(args.gpu_ids, int):
    args.gpu_ids = [args.gpu_ids]
device = (
    torch.device("cuda", args.gpu_ids[0])
    if args.gpu_ids[0] >= 0
    else torch.device("cpu")
)

# Print config and args.
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))

# =============================================================================
#   SETUP DATASET, DATALOADER, MODEL, CRITERION, OPTIMIZER, SCHEDULER
# =============================================================================

train_sample_dataset = VisDialDataset(
    config["dataset"],
    args.train_json,
    overfit=args.overfit,
    in_memory=args.in_memory,
    return_options=True,
    add_boundary_toks=False,
    sample_flag=True  # only train on data with dense annotations
)
train_sample_dataloader = DataLoader(
    train_sample_dataset,
    batch_size=config["solver"]["batch_size"],
    num_workers=args.cpu_workers,
    shuffle=True,
)

val_dataset = VisDialDataset(
    config["dataset"],
    args.val_json,
    args.val_dense_json,
    overfit=args.overfit,
    in_memory=args.in_memory,
    return_options=True,
    add_boundary_toks=False,
    sample_flag=False
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=config["solver"]["batch_size"],
    num_workers=args.cpu_workers,
)

# Pass vocabulary to construct Embedding layer.
encoder_dict = Dict_Encoder(config_dict["model"], train_sample_dataset.vocabulary)
encoder = Encoder(config["model"], train_sample_dataset.vocabulary)
decoder = Decoder(config["model"], train_sample_dataset.vocabulary)
decoder.word_embed = encoder.word_embed
model_dict = encoder_dict.to(device)
# Wrap encoder and decoder in a model.
model = EncoderDecoderModel(encoder, decoder).to(device)
if -1 not in args.gpu_ids:
    model = nn.DataParallel(model, args.gpu_ids)

criterion = nn.CrossEntropyLoss()
criterion_bce = nn.BCEWithLogitsLoss()
iterations = len(train_sample_dataset) // config["solver"]["batch_size"] + 1


def lr_lambda_fun(current_iteration: int) -> float:
    """Returns a learning rate multiplier.

    Till `warmup_epochs`, learning rate linearly increases to `initial_lr`,
    and then gets multiplied by `lr_gamma` every time a milestone is crossed.
    """
    current_epoch = float(current_iteration) / iterations
    if current_epoch < config["solver"]["warmup_epochs"]:
        alpha = current_epoch / float(config["solver"]["warmup_epochs"])
        return config["solver"]["warmup_factor"] * (1.0 - alpha) + alpha
    else:
        idx = bisect(config["solver"]["lr_milestones"], current_epoch)
        return pow(config["solver"]["lr_gamma"], idx)


optimizer = optim.Adamax([
    {'params':model.parameters(), 'lr':config["solver"]["initial_lr"]},
    {'params':model_dict.parameters(), 'lr':config["solver"]["initial_lr"]*0.05} #model is model_dict
])
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fun)  # 可以在一个组里面调节lr参数
start_time = datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')
sparse_metrics = SparseGTMetrics()
ndcg = NDCG()

# If loading from checkpoint, adjust start epoch and load parameters.

start_epoch = 0
model_state_dict, _ = load_checkpoint(args.load_pthpath)
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(model_state_dict)
else:
    model.load_state_dict(model_state_dict)
model_state_dict, _ = load_checkpoint(args.load_dict_pthpath)
if isinstance(model_dict, nn.DataParallel):
    model_dict.module.load_state_dict(model_state_dict)
else:
    model_dict.load_state_dict(model_state_dict)

###start training and set functions used in training
def get_1round_batch_data(batch, rnd):  ##to get 1 round data
    temp_train_batch = {}
    for key in batch:
        if key in ['img_feat']:
            temp_train_batch[key] = batch[key].to(device)
        elif key in ['ques', 'opt', 'ques_len', 'opt_len', 'ans_ind']:
            temp_train_batch[key] = batch[key][:, rnd].to(device)
        elif key in ['hist_len', 'hist']:
            temp_train_batch[key] = batch[key][:, :rnd + 1].to(device)
        else:
            pass
    return temp_train_batch


def get_1round_idx_batch_data(batch, rnd, idx):  ##to get 1 round data with batch_size = 1
    temp_train_batch = {}
    for key in batch:
        if key in ['img_feat']:
            temp_train_batch[key] = batch[key][idx * 2:idx * 2 + 2].to(device)
        elif key in ['ques', 'opt', 'ques_len', 'opt_len', 'ans_ind']:
            temp_train_batch[key] = batch[key][idx * 2:idx * 2 + 2][:, rnd].to(device)
        elif key in ['hist_len', 'hist']:
            temp_train_batch[key] = batch[key][idx * 2:idx * 2 + 2][:, :rnd + 1].to(device)
        else:
            pass
    return temp_train_batch

global_iteration_step = start_epoch * iterations
###load ndcg label list
samplefile = open('data/visdial_1.0_train_dense_sample.json', 'r')
sample = json.loads(samplefile.read())
samplefile.close()
ndcg_id_list = []
for idx in range(len(sample)):
    ndcg_id_list.append(sample[idx]['image_id'])


for epoch in range(start_epoch, config["solver"]["num_epochs"]):
    model.train()
    print('Training for epoch:', epoch, ' time:', time.asctime(time.localtime(time.time())))
    count_loss = 0.0
    for k, batch in enumerate(train_sample_dataloader):
        ##### find the round
        batchsize = batch['img_ids'].shape[0]
        grad_dict = {}
        optimizer.zero_grad()
        for idx in range(int(batchsize / 2)):
            for b in range(2):  # here is because with the batch_size = 1 will raise error
                sample_idx = ndcg_id_list.index(batch['img_ids'][idx * 2 + b].item())
                final_round = sample[sample_idx]['round_id'] - 1
                rnd = final_round
                temp_train_batch = get_1round_idx_batch_data(batch, rnd, idx)
                output_base = model(temp_train_batch)[b]  ## this is only for avoid bug, no other meanings
                output_dict = model_dict(temp_train_batch)[b]
                output = output_base + 0.1 * output_dict
                target = batch["ans_ind"][b, rnd].to(device)
                rs_score = sample[sample_idx]['relevance']
                cuda_device = output.device
                # R3 loss
                batch_loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
                rs_score = torch.tensor(rs_score).to(cuda_device)
                exp_sum = torch.sum(torch.exp(output[[idx for idx in range(len(rs_score)) if rs_score[idx] < 1]]))
                loss_num_count = 0
                for rs_idx in range(len(rs_score)):  # for the candidate with relevance score 1
                    if rs_score[rs_idx] > 0.8:
                        exp_sum = exp_sum + torch.exp(output[rs_idx])
                        batch_loss += (-output[rs_idx] + torch.log(exp_sum))
                        loss_num_count += 1
                        exp_sum = exp_sum - torch.exp(output[rs_idx])
                exp_sum_2 = torch.sum(
                    torch.exp(output[[idx for idx in range(len(rs_score)) if rs_score[idx] < 0.4]]))
                for rs_idx in range(len(rs_score)):  # for the candidate with relevance score 0.5
                    if rs_score[rs_idx] < 0.8 and rs_score[rs_idx] > 0.4:
                        exp_sum_2 = exp_sum_2 + torch.exp(output[rs_idx])
                        batch_loss += (-output[rs_idx] + torch.log(exp_sum_2))
                        loss_num_count += 1
                        exp_sum_2 = exp_sum_2 - torch.exp(output[rs_idx])
                batch_loss = batch_loss / (loss_num_count + 1)
                if batch_loss != 0:  # prevent batch loss = 0
                    batch_loss.backward()
                    count_loss += batch_loss.data.cpu().numpy()
        optimizer.step()  ##accumulate the whole grads in a batch (default is 12) and update weights
        optimizer.zero_grad()

        if k % int(iterations / 5) == 0 and k != 0:
            mean_loss = (count_loss / (float(iterations) / 5)) / 10.0
            print('(step', k, 'in', int(iterations), ') mean_loss:', mean_loss, 'Time:',
                  time.asctime(time.localtime(time.time())), 'lr:', optimizer.param_groups[0]["lr"])
            count_loss = 0.0
        scheduler.step(global_iteration_step)
        global_iteration_step += 1
        # if  k == 5: #for debug
        #     break

model.eval()
for i, batch in enumerate(val_dataloader):
    batchsize = batch['img_ids'].shape[0]
    rnd = 0
    temp_train_batch = get_1round_batch_data(batch, rnd)
    output_temp = model(temp_train_batch).view(-1, 1, 100).detach()
    output_dict = model_dict(temp_train_batch).view(-1, 1, 100).detach()
    output = output_temp + 0.1 * output_dict
    optimizer.zero_grad()
    for rnd in range(1, 10):  # get 10 rounds outputs to evaluate
        temp_train_batch = get_1round_batch_data(batch, rnd)
        output_temp = model(temp_train_batch).view(-1, 1, 100).detach()
        output_dict = model_dict(temp_train_batch).view(-1, 1, 100).detach()
        output = torch.cat((output, output_temp + 0.1 * output_dict), dim=1)
        optimizer.zero_grad()
    sparse_metrics.observe(output, batch["ans_ind"])
    if "relevance" in batch:
        output = output[torch.arange(output.size(0)), batch["round_id"] - 1, :]
        ndcg.observe(output.view(-1, 100), batch["relevance"].contiguous().view(-1, 100))
    # if  i == 5: #for debug
    #     break
all_metrics = {}
all_metrics.update(sparse_metrics.retrieve(reset=True))
all_metrics.update(ndcg.retrieve(reset=True))
for metric_name, metric_value in all_metrics.items():
    print(f"{metric_name}: {metric_value}")
model.train()
