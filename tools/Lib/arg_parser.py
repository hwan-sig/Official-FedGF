import argparse
MODELS = ['FedSAMcnn', 'resnet18_nonorm']

def get_parser():
    parser = argparse.ArgumentParser(description="Standalone training")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--model", type=str, required=True, choices=MODELS)
    parser.add_argument("--pre_trained", action='store_true')
    parser.add_argument("--alg", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--eval_every", type=int, required=True)
    parser.add_argument("--avg_test", action='store_true')
    parser.add_argument("--save_model", action='store_true')

    # dataset distribution
    parser.add_argument("--balance", action='store_true')
    parser.add_argument("--partition", type=str)
    parser.add_argument("--dir_alpha", type=str)
    parser.add_argument("--transform", action='store_true')

    parser.add_argument("--wandb_project_name", type=str, required=True)

    parser.add_argument("--total_client", type=int, default=100)
    parser.add_argument("--com_round", type=int)

    parser.add_argument("--sample_ratio", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float, default=0.0004)
    parser.add_argument("--momentum", type=float, default=0)

    # scaffold
    parser.add_argument("--g_lr", type=float)
    # feddyn
    parser.add_argument("--alpha", type=float)
    # fedprox
    parser.add_argument("--mu", type=float)
    # fedsam
    parser.add_argument("--rho", type=float)
    # fedasam
    parser.add_argument("--eta", type=float)
    # mofedsam, fedavgm
    parser.add_argument("--beta", type=float)
    parser.add_argument("--eta_g", type=float, default=1)
    # FedGF
    parser.add_argument("--T_D", type=float)
    parser.add_argument("--g_rho", type=float)
    parser.add_argument("--W", type=int)

    return parser.parse_args()
