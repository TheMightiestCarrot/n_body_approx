import time
import traceback
import torch
import wandb
from e3nn.o3 import Irreps, spherical_harmonics
from datasets.nbody.dataset_gravity import GravityDataset
from torch import nn, optim
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_scatter import scatter

time_exp_dic = {'time': 0, 'counter': 0}


class O3Transform:
    def __init__(self, lmax_attr, use_force=False):
        self.attr_irreps = Irreps.spherical_harmonics(lmax_attr)
        self.use_force = use_force

    def __call__(self, graph):
        pos = graph.pos
        vel = graph.vel
        mass = graph.mass
        force = graph.force

        prod_mass = mass[graph.edge_index[0]] * mass[graph.edge_index[1]]
        rel_pos = pos[graph.edge_index[0]] - pos[graph.edge_index[1]]
        edge_dist = torch.sqrt(rel_pos.pow(2).sum(1, keepdims=True))

        # steerable edge attributes (relative positions in most cases)
        # but any geometric quantity can be used to steer messages (like relative force or relative velocity)
        graph.edge_attr = spherical_harmonics(self.attr_irreps, rel_pos, normalize=True, normalization='integral')
        # for example adding additional geometric quantities to steer the messages
        # rel_vel = vel[graph.edge_index[0]] - vel[graph.edge_index[1]]
        # rel_force = force[graph.edge_index[0]] - force[graph.edge_index[1]]
        # graph.edge_attr += spherical_harmonics(self.attr_irreps, rel_vel, normalize=True, normalization='integral')
        # graph.edge_attr += spherical_harmonics(self.attr_irreps, rel_force, normalize=True, normalization='integral')

        # steerable node attributes
        vel_embedding = spherical_harmonics(self.attr_irreps, vel, normalize=True, normalization='integral')
        graph.node_attr = scatter(graph.edge_attr, graph.edge_index[1], dim=0, reduce="mean") + vel_embedding

        if self.use_force:
            force_embedding = spherical_harmonics(self.attr_irreps, force, normalize=True,
                                                  normalization='integral')
            graph.node_attr += force_embedding

        vel_abs = torch.sqrt(vel.pow(2).sum(1, keepdims=True))
        mean_pos = pos.mean(1, keepdims=True)

        graph.x = torch.cat((pos - mean_pos, vel, vel_abs), 1)
        graph.additional_message_features = torch.cat((edge_dist, prod_mass), dim=-1)
        return graph


def train(gpu, model, args, log_manager=None):
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    try:
        if args.gpus == 0:
            device = 'cpu'
        else:
            device = torch.device('cuda:' + str(gpu))

        model = model.to(device)

        dataset_train = GravityDataset(partition='train', dataset_name=args.nbody_name,
                                       max_samples=args.max_samples, neighbours=args.neighbours, target=args.target,
                                       steps_to_predict=args.steps_to_predict,
                                       random_trajectory_sampling=args.random_trajectory_sampling)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                   drop_last=True)

        dataset_val = GravityDataset(partition='val', dataset_name=args.nbody_name,
                                     max_samples=args.max_samples, neighbours=args.neighbours, target=args.target,
                                     steps_to_predict=args.steps_to_predict,
                                     random_trajectory_sampling=args.random_trajectory_sampling)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                                 drop_last=False)

        dataset_test = GravityDataset(partition='test', dataset_name=args.nbody_name,
                                      max_samples=args.max_samples, neighbours=args.neighbours, target=args.target,
                                      steps_to_predict=args.steps_to_predict,
                                      random_trajectory_sampling=args.random_trajectory_sampling)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                                  drop_last=False)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_mse = nn.MSELoss()
        transform = O3Transform(args.lmax_attr)

        if args.log and gpu == 0:
            if args.time_exp:
                wandb.init(project="Gravity time", name=args.ID, config=args, entity="segnn")
            else:
                wandb.init(project="SEGNN Gravity", name=args.ID, config=args, entity="segnn")

        for epoch in range(0, args.epochs):
            train_loss = run_epoch(model, optimizer, loss_mse, epoch, loader_train, transform, device, args,
                                   log_manager=log_manager)
            if args.log and gpu == 0:
                wandb.log({"Train MSE": train_loss})
            if epoch % args.test_interval == 0 or epoch == args.epochs - 1:
                # train(epoch, loader_train, backprop=False)
                val_loss = run_epoch(model, optimizer, loss_mse, epoch, loader_val, transform, device, args,
                                     backprop=False,
                                     log_manager=log_manager)
                test_loss = run_epoch(model, optimizer, loss_mse, epoch, loader_test,
                                      transform, device, args, backprop=False, log_manager=log_manager)
                if args.log and gpu == 0:
                    wandb.log({"Val MSE": val_loss})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_test_loss = test_loss
                    best_epoch = epoch
                print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d" %
                      (best_val_loss, best_test_loss, best_epoch))

        if args.log and gpu == 0:
            wandb.log({"Test MSE": best_test_loss})

        log_manager.log_hparams(vars(args), best_test_loss)

    except KeyboardInterrupt:
        print("Keyboard interrupt, saving logs and model state.")
        pass
    except Exception as e:
        print("Error occurred, saving logs and model state.")
        traceback.print_exc()
        log_manager.log_text("ERROR", traceback.format_exc())

    return best_val_loss, best_test_loss, best_epoch


def run_epoch(model, optimizer, criterion, epoch, loader, transform, device, args, backprop=True, log_manager=None):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {
        'epoch': epoch, 'loss': 0, 'counter': 0,
        'loss_pos': 0, 'loss_vel': 0,
        'perc_error_pos': 0, 'perc_error_vel': 0,
        'perc_error_pos_vs_vel': 0
    }

    calculate_pos_and_vel_loss = (args.target == 'pos_dt+vel_dt') or (args.target == 'pos+vel')

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(2)) for d in data]
        loc, vel, force, mass, y = data

        dims = loc.shape[-1]

        optimizer.zero_grad()

        if args.time_exp:
            torch.cuda.synchronize()
            t1 = time.time()

        if args.model == 'segnn' or args.model == 'seconv':
            graph = Data(pos=loc, vel=vel, force=force, mass=mass, y=y)
            batch = torch.arange(0, batch_size).to(device)
            graph.batch = batch.repeat_interleave(n_nodes).long()
            graph.edge_index = knn_graph(loc, args.neighbours, graph.batch)

            graph = transform(graph)  # Add O3 attributes
            graph = graph.to(device)
            pred = model(graph)
        else:
            raise Exception("Wrong model")

        if args.time_exp:
            torch.cuda.synchronize()
            t2 = time.time()
            time_exp_dic['time'] += t2 - t1
            time_exp_dic['counter'] += 1

            if epoch % 100 == 99:
                print("Forward average time: %.6f" % (time_exp_dic['time'] / time_exp_dic['counter']))
                if args.log:
                    wandb.log({"Time": time_exp_dic['time'] / time_exp_dic['counter']})
        loss = criterion(pred, graph.y)

        if calculate_pos_and_vel_loss:
            predicted_pos, predicted_vel = pred[..., :dims], pred[..., dims:]
            target_pos, target_vel = y[..., :dims], y[..., dims:]

            loss_pos = criterion(predicted_pos, target_pos)
            loss_vel = criterion(predicted_vel, target_vel)

            # l1 = MAE, l2 = euclidean distance
            # todo add epsilon to prevent zero division

            pos_l1_error = torch.abs(predicted_pos - target_pos).mean()  # MAE
            pos_l2_error = torch.norm(predicted_pos - target_pos, dim=1)
            pos_target_l2 = torch.norm(target_pos, dim=1)
            perc_error_pos = (pos_l2_error / pos_target_l2).mean() * 100

            vel_l2_error = torch.norm(predicted_vel - target_vel, dim=1)
            vel_target_l2 = torch.norm(target_vel, dim=1)
            perc_error_vel = (vel_l2_error / vel_target_l2).mean() * 100

            perc_error_pos_vs_vel = (pos_l2_error / vel_target_l2).mean() * 100

            # print("predicted_pos:", predicted_pos.tolist())
            # print("target_pos:", target_pos.tolist())
            # print("pos_l1_error:", pos_l1_error.item())
            # print("pos_l2_error:", pos_l2_error.tolist())
            # print("pos_target_l2:", pos_target_l2.tolist())
            # print("perc_error_pos:", perc_error_pos.item())

            # print(vel_l2_error, vel_target_l2)
            # print(perc_error_vel)

        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size

        if calculate_pos_and_vel_loss:
            res['loss_pos'] += loss_pos.item() * batch_size
            res['loss_vel'] += loss_vel.item() * batch_size
            res['perc_error_pos'] += perc_error_pos.item() * batch_size
            res['perc_error_vel'] += perc_error_vel.item() * batch_size
            res['perc_error_pos_vs_vel'] += perc_error_pos_vs_vel.item() * batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""

    # Print average loss with formatted prefix and epoch details
    print(
        f"Dataset Partition: {prefix + loader.dataset.partition}, Epoch: {epoch}, Average Loss:\t{res['loss'] / res['counter']:.5f}")

    # Calculate position and velocity losses if enabled
    if calculate_pos_and_vel_loss:
        # Calculate average metrics, excluding 'counter'
        avg_metrics = {k: v / res['counter'] for k, v in res.items() if k != 'counter'}

        # Print a header for average losses
        print("Average Loss Metrics:")
        print(f"  Total Loss:\t\t{avg_metrics['loss']:.5f}")
        print(f"  Position Loss:\t{avg_metrics['loss_pos']:.5f}")
        print(f"  Velocity Loss:\t{avg_metrics['loss_vel']:.5f}\n")

        # Print a header for average percentage errors
        print("Average Percentage Errors (L2):")
        print(f"  Position:\t\t{avg_metrics['perc_error_pos']:.2f}%")
        print(f"  Velocity:\t\t{avg_metrics['perc_error_vel']:.2f}%\n")
        print(f"  Position vs Velocity:\t\t{avg_metrics['perc_error_pos_vs_vel']:.2f}%\n")

    log_manager.log_scalar(f'{loader.dataset.partition}/loss', res['loss'] / res['counter'], epoch)
    if calculate_pos_and_vel_loss:
        log_manager.log_scalar(f'{loader.dataset.partition}/pos', avg_metrics["loss_pos"], epoch)
        log_manager.log_scalar(f'{loader.dataset.partition}/vel', avg_metrics["loss_vel"], epoch)

        log_manager.log_scalar(f'{loader.dataset.partition}/perc_pos', avg_metrics["perc_error_pos"], epoch)
        log_manager.log_scalar(f'{loader.dataset.partition}/perc_vel', avg_metrics["perc_error_vel"], epoch)
        log_manager.log_scalar(f'{loader.dataset.partition}/perc_pos_vs_vel', avg_metrics["perc_error_pos_vs_vel"],
                               epoch)

    return res['loss'] / res['counter']
