import torch
import argparse
import os
import numpy as np
import torch.multiprocessing as mp
from e3nn.o3 import Irreps, spherical_harmonics
from models.balanced_irreps import BalancedIrreps, WeightBalancedIrreps
import sys
import utils.loggers as loggers
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import segnn_utils


def _find_free_port():
    """ Find free port, so multiple runs don't clash """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


if __name__ == "__main__":
    # sys.argv = [
    #     'main.py', '--dataset=gravity', '--neighbours=4', '--epochs=1000',
    #     '--max_samples=10000', '--model=segnn', '--lmax_h=1', '--lmax_attr=1',
    #     '--layers=4', '--hidden_features=64', '--subspace_type=weightbalanced',
    #     '--norm=none', '--batch_size=100', '--gpu=1', '--weight_decay=1e-12', '--target=pos'
    # ]

    sys.argv = [
        'main.py', '--experiment_name=segnn_runs_v2', '--dataset=gravityV3', '--epochs=500', '--max_samples=10',
        '--model=segnn', '--lmax_h=1', '--lmax_attr=1', '--layers=4',
        '--hidden_features=64', '--subspace_type=weightbalanced', '--norm=none',
        '--batch_size=10', '--gpu=1', '--weight_decay=1e-12', '--target=pos+vel',
        '--random_trajectory_sampling=True'
    ]

    # sys.argv = [
    #     'main.py', '--dataset=nbody', '--epochs=1000', '--max_samples=3000',
    #     '--model=segnn', '--lmax_h=1', '--lmax_attr=1', '--layers=4',
    #     '--hidden_features=64', '--subspace_type=weightbalanced', '--norm=none',
    #     '--batch_size=100', '--gpu=1', '--weight_decay=1e-12'
    # ]

    parser = segnn_utils.create_argparser()

    args = parser.parse_args()

    # Select dataset.
    if args.dataset == "nbody":
        from datasets.nbody.train_nbody import train

        task = "node"
        input_irreps = Irreps("2x1o + 1x0e")
        output_irreps = Irreps("1x1o")
        edge_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
        node_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
        additional_message_irreps = Irreps("2x0e")
    elif args.dataset == "gravity":
        from datasets.nbody.train_gravity import train

        task = "node"
        input_irreps = Irreps("2x1o + 1x0e")
        output_irreps = Irreps("1x1o")
        edge_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
        node_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
        additional_message_irreps = Irreps("2x0e")
    elif args.dataset == "gravityV3":
        from datasets.nbody.train_gravity_V3 import train

        task = "node"
        input_irreps = Irreps("2x1o + 1x0e")
        output_irreps = Irreps("2x1o")
        edge_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
        node_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
        additional_message_irreps = Irreps("2x0e")
    else:
        raise Exception("Dataset could not be found")

    # Create hidden irreps
    if args.subspace_type == "weightbalanced":
        hidden_irreps = WeightBalancedIrreps(
            Irreps("{}x0e".format(args.hidden_features)), node_attr_irreps, sh=True, lmax=args.lmax_h)
    elif args.subspace_type == "balanced":
        hidden_irreps = BalancedIrreps(args.lmax_h, args.hidden_features, True)
    else:
        raise Exception("Subspace type not found")

    # Select model
    if args.model == "segnn":
        from models.segnn.segnn import SEGNN

        model = SEGNN(input_irreps,
                      hidden_irreps,
                      output_irreps,
                      edge_attr_irreps,
                      node_attr_irreps,
                      num_layers=args.layers,
                      norm=args.norm,
                      pool=args.pool,
                      task=task,
                      additional_message_irreps=additional_message_irreps,
                      training_args=args)
        args.ID = "_".join([args.model, args.dataset, args.target, str(np.random.randint(1e4, 1e5))])
    elif args.model == "seconv":
        from models.segnn.seconv import SEConv

        model = SEConv(input_irreps,
                       hidden_irreps,
                       output_irreps,
                       edge_attr_irreps,
                       node_attr_irreps,
                       num_layers=args.layers,
                       norm=args.norm,
                       pool=args.pool,
                       task=task,
                       additional_message_irreps=additional_message_irreps,
                       conv_type=args.conv_type)
        args.ID = "_".join([args.model, args.conv_type, args.dataset, str(np.random.randint(1e4, 1e5))])
    else:
        raise Exception("Model could not be found")

    if "gravity" in args.dataset:
        print("setting model to double precision")
        model.double()

    print(model)
    print("The model has {:,} parameters.".format(sum(p.numel() for p in model.parameters())))

    current_time = datetime.now().strftime('%Y-%m-%d %H-%M')
    run_name = f'{current_time}_{args.dataset}_{args.model}'
    experiment_folder = os.path.join(args.experiment_name, run_name)
    writer = SummaryWriter(experiment_folder)

    log_manager = loggers.LoggingManager()
    log_manager.add_logger(loggers.TensorBoardLogger(writer, None))
    log_manager.log_text('args', ', '.join(f'{k}={v}' for k, v in vars(args).items()))

    if loggers.WandBLogger.get_api_key() is not None:
        wandb_logger = loggers.WandBLogger(project_name=args.experiment_name, run_name=run_name, config=vars(args))
        log_manager.add_logger(wandb_logger)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.gpus == 1 and torch.cuda.is_available():
        print('Starting training on a single gpu...')
        args.mode = 'gpu'
        train(0, model, args, log_manager)
    else:
        print('Starting training on the cpu...')
        args.mode = 'cpu'
        args.gpus = 0
        train(0, model, args, log_manager)

    log_manager.log_model(model)
    log_manager.finish()
