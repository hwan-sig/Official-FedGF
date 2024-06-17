from json import load
import os
import random
from copy import deepcopy
from torch import nn
import sys
import torch
import numpy as np
import pandas as pd

sys.path.append("../")  # To use fedlab library
DATA_ROOT = '/data/'
JSON_PATH = os.path.join(os.path.dirname(__file__), 'json_data')
torch.manual_seed(0)

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate, get_best_gpu, val_eval, schedule_cycling_lr

from fedlab.core.standalone import StandalonePipeline

from Lib.algorithms import load_algorithms, average
from Lib.datasets import load_datasets
from Lib.arg_parser import get_parser
from Lib.models import get_model
import wandb

# configuration
args = get_parser()
c_values = []

if 'debug' in args.wandb_project_name:
    mode = 'disabled'
else:
    mode = 'online'

wandb.init(project=args.wandb_project_name, mode=mode)
wandb.config.update(args)
model = get_model(args)

handler, trainer = load_algorithms(args, model)
train_dataset, test_loader = load_datasets(args, DATA_ROOT, JSON_PATH, trainer)

num_rounds = args.com_round
eval_every = args.eval_every

accuracy = []
val_accuracy = []
handler.num_clients = trainer.num_clients

wandb_dict = {}

while handler.if_stop is False:
    wandb_dict.clear()

    sampled_clients = handler.sample_clients()
    if 'debug' in args.wandb_project_name:
        print(f"round:{handler.round},clients:{sampled_clients}")
    broadcast = handler.downlink_package

    trainer.local_process(broadcast, sampled_clients)
    uploads = trainer.uplink_package

    # server side
    for pack in uploads:
        handler.load(pack)

    FLAG = False
    if (args.avg_test and ((handler.round >= args.com_round - 100) or ((handler.round // eval_every) and (handler.round % eval_every) < 100))) or\
            (handler.round >= args.com_round - 100) or ((handler.round // eval_every) and not handler.round % eval_every):

        FLAG = True
        val_loss, val_acc = val_eval(handler._model, nn.CrossEntropyLoss(), sampled_clients, train_dataset)
        loss, acc = evaluate(handler._model, nn.CrossEntropyLoss(), test_loader)

        wandb_dict.update({"test loss": loss, "test acc.": acc * 100, "val loss": val_loss, "val acc": val_acc * 100})

        accuracy.append(acc * 100)
        val_accuracy.append(val_acc * 100)

        print("round {}, Test Accuracy: {:.4f}, Max Acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            handler.round, acc * 100, max(accuracy), val_loss, val_acc))

    elif handler.round == 0:
        FLAG = True
        val_loss, val_acc = val_eval(handler._model, nn.CrossEntropyLoss(), sampled_clients, train_dataset)
        loss, acc = evaluate(handler._model, nn.CrossEntropyLoss(), test_loader)

        wandb_dict.update({"test loss": loss, "test acc.": acc * 100, "val loss": val_loss, "val acc": val_acc * 100})


    if FLAG:
        print(wandb_dict)
        wandb.log(wandb_dict, step=handler.round)

    handler.round += 1

if args.save_model:
    save_info = {'model_state_dict': handler.model.state_dict(),
                 'round': handler.round,
                 'args': args}
    torch.save(save_info, os.path.join(wandb.run.dir, f"Algo{args.alg}_Data{args.dataset}_Alp{args.dir_alpha}.ckpt"))

wandb.log({
    "Eval:Avg Acc.": round(average(accuracy[-100:]), 2),
    "Eval:Max Acc.": round(max(accuracy), 2),
    "Eval:std": round(np.array(accuracy[-100:]).std(), 2),
    "Val:Avg Acc.": round(average(val_accuracy[-100:]), 2),
    "Val:Max Acc.": round(max(val_accuracy), 2),
    "Val:std": round(np.array(val_accuracy[-100:]).std(), 2),
})
