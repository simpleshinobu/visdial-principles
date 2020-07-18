import argparse
import time
import datetime
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import yaml
from bisect import bisect
import random
from visdialch.data.dataset import VisDialDataset
from visdialch.encoders import Encoder
from visdialch.decoders import Decoder
from visdialch.metrics import SparseGTMetrics, NDCG
from visdialch.model import EncoderDecoderModel
from visdialch.utils.checkpointing import CheckpointManager, load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-yml",
    default="configs/baseline.yml",
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
    default=[0, 1],
    help="List of ids of GPUs to use.",
)
parser.add_argument(
    "--cpu-workers",
    type=int,
    default=8,
    help="Number of CPU workers for dataloader.",
)
parser.add_argument(
    "--overfit",
    action="store_true",
    help="Overfit model on 5 examples, meant for debugging.",
)
parser.add_argument(
    "--validate",
    action="store_true",
    help="Whether to validate on val split after every epoch.",
)
parser.add_argument(
    "--in-memory",
    action="store_true",
    help="Load the whole dataset and pre-extracted image features in memory. "
         "Use only in presence of large RAM, atleast few tens of GBs.",
)

parser.add_argument_group("Checkpointing related arguments")
parser.add_argument(
    "--save-dirpath",
    default="checkpoints/",
    help="Path of directory to create checkpoint directory and save "
         "checkpoints.",
)
parser.add_argument(
    "--load-pthpath",
    default="",
    help="To continue training, path to .pth file of saved checkpoint.",
)
parser.add_argument(
    "--save-model",
    action="store_true",
    help="To make the dir clear",
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

train_dataset = VisDialDataset(
    config["dataset"],
    args.train_json,
    overfit=args.overfit,
    in_memory=args.in_memory,
    return_options=True,
    add_boundary_toks=False,
    sample_flag=False
)
train_dataloader = DataLoader(
    train_dataset,
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
    shuffle=True,
)

# Pass vocabulary to construct Embedding layer.
encoder = Encoder(config["model"], train_dataset.vocabulary)
decoder = Decoder(config["model"], train_dataset.vocabulary)
print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))

# Share word embedding between encoder and decoder.
if args.load_pthpath == "":
    print('load glove')
    decoder.word_embed = encoder.word_embed
    glove = np.load('data/glove.npy')
    encoder.word_embed.weight.data = torch.tensor(glove)

# Wrap encoder and decoder in a model.
model = EncoderDecoderModel(encoder, decoder).to(device)
if -1 not in args.gpu_ids:
    model = nn.DataParallel(model, args.gpu_ids)

criterion = nn.CrossEntropyLoss()
iterations = len(train_dataset)// config["solver"]["batch_size"] + 1  # 迭代次数

def lr_lambda_fun(current_iteration: int) -> float:
    """Returns a learning rate multiplier.

    Till `warmup_epochs`, learning rate linearly increases to `initial_lr`,
    and then gets multiplied by `lr_gamma` every time a milestone is crossed.
    """
    current_epoch = float(current_iteration) / iterations
    if current_epoch <= config["solver"]["warmup_epochs"]:
        alpha = current_epoch / float(config["solver"]["warmup_epochs"])
        return config["solver"]["warmup_factor"] * (1.0 - alpha) + alpha
    else:
        idx = bisect(config["solver"]["lr_milestones"], current_epoch)
        return pow(config["solver"]["lr_gamma"], idx)

optimizer = optim.Adamax(model.parameters(), lr=config["solver"]["initial_lr"])
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fun)
# =============================================================================
#   SETUP BEFORE TRAINING LOOP
# =============================================================================
start_time = datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')
checkpoint_dirpath = args.save_dirpath
if checkpoint_dirpath == 'checkpoints/':
    checkpoint_dirpath += '%s+%s/%s' % (config["model"]["encoder"], config["model"]["decoder"], start_time)
if args.save_model:
    summary_writer = SummaryWriter(log_dir=checkpoint_dirpath)
    checkpoint_manager = CheckpointManager(model, optimizer, checkpoint_dirpath, config=config)

sparse_metrics = SparseGTMetrics()
ndcg = NDCG()

# If loading from checkpoint, adjust start epoch and load parameters.
if args.load_pthpath == "":
    start_epoch = 0
else:
    start_epoch = int(args.load_pthpath.split("_")[-1][:-4])  + 1

    model_state_dict, optimizer_state_dict = load_checkpoint(args.load_pthpath)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    print("Loaded model from {}".format(args.load_pthpath))

# =============================================================================
#   TRAINING LOOP
# =============================================================================

# Forever increasing counter to keep track of iterations (for tensorboard log).
global_iteration_step = start_epoch * iterations

###start training and set functions used in training
def get_1round_batch_data(batch, rnd):
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

for epoch in range(start_epoch, config["solver"]["num_epochs"]):
    print('Training for epoch:', epoch, ' time:', time.asctime(time.localtime(time.time())))
    count_loss = 0.0
    for i, batch in enumerate(train_dataloader):
        for rnd in range(10):
            temp_train_batch = get_1round_batch_data(batch, rnd)
            optimizer.zero_grad()
            output = model(temp_train_batch)
            target = batch["ans_ind"][:, rnd].to(device)
            batch_loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            batch_loss.backward()
            count_loss += batch_loss.data.cpu().numpy()
            optimizer.step()
        ##for rva, apply 10 rounds because of the implementation of rva is hard to separate into 10 parts, according to the original authors
        ##note that whether separate into 10 rounds here will not influence the conclusion in our paper
        ##but it is interesting topic in the future
        # for key in batch:
        #     batch[key] = batch[key].to(device)
        # output = model(batch)
        # target = batch["ans_ind"].to(device)
        # batch_loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
        # batch_loss.backward()
        # count_loss += batch_loss.data.cpu().numpy() * 10.0
        # optimizer.step()
        # optimizer.zero_grad()
        ###################whole 10 rounds part end
        if i % int(iterations / 10) == 0 and i != 0:
            mean_loss = (count_loss / float(iterations / 10)) / 10.0
            print('(step', i, 'in', int(iterations), ') mean_loss:', mean_loss, 'Time:',
                  time.asctime(time.localtime(time.time())), 'lr:', optimizer.param_groups[0]["lr"])
            count_loss = 0.0
        if args.save_model:
            summary_writer.add_scalar("train/loss", batch_loss, global_iteration_step)
            summary_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_iteration_step)
            scheduler.step(global_iteration_step)
        global_iteration_step += 1
        # if i > 5: #for debug(like the --overfit)
        #     break
    # -------------------------------------------------------------------------
    #   ON EPOCH END  (checkpointing and validation)
    # -------------------------------------------------------------------------
    if args.save_model:
        checkpoint_manager.step()
    # Validate and report automatic metrics.
    if args.validate:
        print(f"\nValidation after epoch {epoch}:")
        model.eval()
        for i, batch in enumerate(val_dataloader):
            batchsize = batch['img_ids'].shape[0]
            rnd = 0
            temp_train_batch = get_1round_batch_data(batch, rnd)
            output = model(temp_train_batch).view(-1, 1, 100).detach()
            optimizer.zero_grad()
            for rnd in range(1, 10): #should be removed if the input is the whole dialog
                temp_train_batch = get_1round_batch_data(batch, rnd)
                output = torch.cat((output, model(temp_train_batch).view(-1, 1, 100).detach()), dim=1)
                optimizer.zero_grad()
            ###for 10 rounds(rva)
            # with torch.no_grad():
            #     output = model(batch)
            ##end 10 rounds
            sparse_metrics.observe(output, batch["ans_ind"])
            if "relevance" in batch:
                output = output[torch.arange(output.size(0)), batch["round_id"] - 1, :]
                ndcg.observe(output.view(-1, 100), batch["relevance"].contiguous().view(-1, 100))
            # if i > 5: #for debug(like the --overfit)
            #     break
        all_metrics = {}
        all_metrics.update(sparse_metrics.retrieve(reset=True))
        all_metrics.update(ndcg.retrieve(reset=True))
        for metric_name, metric_value in all_metrics.items():
            print(f"{metric_name}: {metric_value}")
        if args.save_model:
            summary_writer.add_scalars("metrics", all_metrics, global_iteration_step)
        model.train()
