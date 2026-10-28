#######################################
# s04_train_model

##############################################################################
#============================================================================#
# 1) Import Statements
#============================================================================#
##############################################################################

import argparse
import json
import random
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")      # no display on the compute nodes, figures are only saved
import matplotlib.pyplot as plt

import s01_download_data as s01
import s02_preprocessing as s02
import s03_build_training_dataset as s03
import xx_plotting as xx

from pathlib import Path
from datetime import datetime
from pyproj import Geod
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

################################################
# 1.1) Constants
################################################

# geodesic distances and bearings on the WGS84 ellipsoid (project.md §4.2)
GEOD = Geod(ellps="WGS84")


##############################################################################
#============================================================================#
# 2) Helper Functions
#============================================================================#
##############################################################################

################################################
#======== Section 1: Dataset Helpers ==========#
################################################
# Used by build_datasets, in the order they're first called there (a helper's own helpers right after it).

def build_pool_index(graph_files):
    """Lists all pools of the given GRAPH files: reads only the seed rows (seed_pid) of every file.
    In: list of GRAPH parquet paths. Out: polars.DataFrame (graph_file, seed_pid), one row per pool."""
    # for every GRAPH file:
    #     read it lazily (pl.scan_parquet), keep only the rows with seed == 1 and only the column seed_pid
    #     -> every seed is exactly one pool, so this is the list of the file's pools
    #     add the file path as column graph_file
    # concatenate the per file tables into one index and return it
    # hint: only two columns of a few thousand rows per file are read, so this takes seconds for all files
    pass


def sample_pools(pool_index, n_graphs, seed):
    """Draws n_graphs pools at random over all files (fixed seed), all pools if n_graphs is None.
    In: pool index, int or None, int. Out: the sampled rows of the pool index."""
    # if n_graphs is None or larger than the index: return the whole index
    # otherwise: rng = np.random.default_rng(seed)
    #            rows = rng.choice(number of rows, size=n_graphs, replace=False)
    #            return the index rows at these positions
    # the draw goes over all pools of all regions at once, so every region keeps its share on average
    pass


def load_pools(sampled_index, use_gse, min_graph_size):
    """Loads the points of the sampled pools into memory (GSE columns only if use_gse), skips pools < min_graph_size.
    In: sampled pool index, bool, int. Out: list of pools, each a dict of numpy arrays (+ region and orbit)."""
    # columns = latitude, longitude, mean_velocity, incidence_angle, seed_pid (+ gse_0 ... gse_63 if use_gse)
    # for every graph_file in the sampled index (group the index by file, so every file is read only once):
    #     read the columns, keep only the rows whose seed_pid is among the file's sampled pools
    #     split the rows by seed_pid (hint: DataFrame.partition_by("seed_pid", as_dict=True))
    #     for every pool:
    #         skip it if it has fewer than min_graph_size points (the minimum pool size, §4.3)
    #         store a dict: lat, lon, vel, inc as float arrays, gse as int8 array (m, 64) if use_gse,
    #         region = folder name two levels up from the file, orbit = folder name one level up
    # return the list of pool dicts
    # hint: float32 for vel/inc and int8 for gse keep the memory small; lat/lon stay float64 for the geodesics
    pass


def load_norm_stats():
    """Reads the train normalization stats of mean_velocity and incidence_angle written by s02.
    In: nothing (s02.EGMS_NORM_STATS_PATH). Out: dict {column: {"mean", "std"}}."""
    # stats = json.load(open(s02.EGMS_NORM_STATS_PATH))
    # return only the entries of the columns in s02.EGMS_NORM_COLS (mean_velocity, incidence_angle)
    pass


def estimate_edge_stats(dataset, n_draws=2000):
    """Estimates mean/std of the edge distances and of dist_to_coherent from random draws of the train dataset (§4.6).
    In: PoolDataset, int. Out: dict, added to the stats of the dataset."""
    # the dataset has no edge stats yet, so its graphs carry the raw (not normalized) distances here
    graphs = [dataset[i] for i in np.random.default_rng(0).integers(0, len(dataset), n_draws)]
    # collect the raw edge distances of all drawn graphs and the raw dist_to_coherent of their queries
    # distances are strongly skewed (many short, few long edges): normalize log(1 + d) rather than d
    # stats = dataset.stats + {"edge_dist": {"mean", "std"}, "dist_to_coherent": {"mean", "std"}} of the log values
    # return stats (build_datasets hands the same dict to the train and the dev dataset)
    pass


class PoolDataset(torch.utils.data.Dataset):
    """One item = one pool. Every __getitem__ draws a fresh subset, query and k-NN graph (masked target, §7.1).
    In: list of pools, stats, k_graph, subset_min/max, min_graph_size, use_gse. Out per item: torch_geometric Data."""

    def __init__(self, pools, stats, k_graph, subset_min, subset_max, min_graph_size, use_gse, fixed_draws=False):
        # store all arguments as attributes (self.pools = pools, self.stats = stats, ...)
        # nothing is computed here: the graphs are built only when an item is requested
        # fixed_draws=True (used for dev): the random generator of item i is seeded with i, so every epoch evaluates
        # the same subsets and queries and the dev loss of different epochs is comparable for the early stopping
        pass

    def __len__(self):
        # the number of pools: one item per pool and epoch
        pass

    def __getitem__(self, i):
        """Builds one graph from pool i. In: pool number. Out: Data(x, edge_index, edge_attr, y, query_mask, region, orbit)."""
        # pool = self.pools[i], n = number of its points, lat/lon = its coordinates
        # rng = np.random.default_rng(i) if self.fixed_draws else a fresh generator
        # hint: with several DataLoader workers a plain np.random state is copied into every worker, so all workers
        #       would draw the same subsets -> create the generator here (or seed per worker with worker_init_fn)
        idx, q = draw_subset(n, self.subset_min, self.subset_max, self.min_graph_size)
        dist, az = pairwise_geodesic(lat[idx], lon[idx])
        edge_index = knn_edges(dist, self.k_graph)
        x = build_node_features(pool, idx, q, self.stats, self.use_gse)
        edge_attr = build_edge_features(dist, az, edge_index, self.stats)
        # y = the query's normalized velocity (the hidden target), shape (1,)
        # query_mask = bool tensor of length m, True only at q (the model reads its prediction there)
        # return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, query_mask=query_mask,
        #             region=pool["region"], orbit=pool["orbit"])
        # hint: PyG batches several Data objects into one big disconnected graph; string attributes become lists
        pass


def draw_subset(n, subset_min, subset_max, min_graph_size):
    """Draws a random share (subset_min..subset_max, at least min_graph_size) of a pool's n points and the query among them.
    In: pool size and limits. Out: index array of the subset, position of the query in it."""
    # share = uniform(subset_min, subset_max)
    # m = number of points: max(min_graph_size, round(share * n)), but never more than n
    # idx = m point indices drawn without replacement
    # q = one random position in 0 .. m-1 (the query)
    # return idx, q
    # hint: take the generator of __getitem__ as extra argument, so fixed_draws works
    pass


def pairwise_geodesic(lat, lon):
    """Geodesic distance (m) and bearing (deg) between all pairs of the subset's points (GEOD).
    In: lat/lon arrays of the subset. Out: two (m, m) matrices, distance and azimuth."""
    # build all pairs: repeat the coordinates for the first point, tile them for the second (m * m pairs)
    # az12, az21, d = GEOD.inv(lon1, lat1, lon2, lat2)    (pyproj takes lon first)
    # reshape az12 and d to (m, m); the diagonal (a point with itself) has distance 0
    # m is at most k_nearest + 1 = 26, so these are at most 676 pairs per graph -> cheap
    pass


def knn_edges(dist, k):
    """Connects every node to its k nearest nodes, symmetrised (§7.1).
    In: (m, m) distance matrix, int. Out: edge_index (2, n_edges) as torch tensor."""
    # set the diagonal to infinity, so no node picks itself
    # for every node: the k columns with the smallest distance (hint: np.argsort(dist, axis=1)[:, :k])
    # edges neighbour -> node: source = neighbour, target = node (PyG: edge_index[0] = source, edge_index[1] = target)
    # symmetrise: add every edge in the opposite direction too and remove duplicates
    # hint: torch_geometric.utils.to_undirected does exactly this
    pass


def build_node_features(pool, idx, q, stats, use_gse):
    """Node features: normalized velocity (0 for the query), is_query flag, normalized incidence, dist_to_coherent, GSE.
    In: pool, subset index, query position, stats, bool. Out: x (m, n_features) as torch tensor."""
    # vel = (pool vel[idx] - mean) / std of mean_velocity; vel[q] = 0   <- the target is hidden here
    # is_query = 0 everywhere, 1 at q
    # inc = (pool inc[idx] - mean) / std of incidence_angle
    # dist_to_coherent = the query's distance to its nearest other node (log, normalized), 0 for all other nodes
    #     hint: it needs the distance matrix -> pass dist as extra argument or compute it in __getitem__
    gse = dequantise_gse(pool["gse"][idx]) if use_gse else None
    # x = the columns [vel, is_query, inc, dist_to_coherent] (+ the 64 gse columns) side by side, float32
    # -> 4 node features without GSE, 68 with GSE
    pass


def dequantise_gse(v):
    """De-quantises the raw int8 GSE values: x = ((v / 127.5) ** 2) * sign(v) (§4.1).
    In: int8 array (m, 64). Out: float32 array (m, 64)."""
    # convert to float first (int8 would overflow), then apply the formula
    # every de-quantised vector has length ~1 (checked on the GRAPH files)
    pass


def build_edge_features(dist, az, edge_index, stats):
    """Edge features per edge: normalized distance and the unit direction of the bearing (sin, cos).
    In: distance and azimuth matrices, edge_index, stats. Out: edge_attr (n_edges, 3) as torch tensor."""
    # src, dst = edge_index
    # d = (log(1 + dist[src, dst]) - mean) / std of edge_dist (raw log distance if the stats don't exist yet)
    # bearing = az[dst, src]: direction from the receiving node to the sending node, in degrees
    # the bearing is circular (359 deg is next to 1 deg), so it goes in as sin and cos of the angle
    # edge_attr = [d, sin, cos] per edge, float32 -> edge_dim = 3
    pass


################################################
#============= Section 2: Model ===============#
################################################
# Used by main.

class InSARGNN(nn.Module):
    """GATv2 message passing over the drawn graph, scalar heteroscedastic head at the query node (§5).
    In: number of node features, edge feature dim, hidden_dim, n_layers, heads, dropout. Out: the model."""

    def __init__(self, n_node_features, edge_dim, hidden_dim, n_layers, heads, dropout):
        # super().__init__() first (registers the layers, so their weights are trained and saved)
        # encoder: a small MLP (Linear -> GELU -> Linear) from n_node_features to hidden_dim
        # n_layers times: GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        #     -> the heads are concatenated, so the output has hidden_dim again; plus one LayerNorm per layer
        #     hint: keep them in nn.ModuleList, a plain python list is not registered
        # two heads: Linear(hidden_dim, 1) for mu and Linear(hidden_dim, 1) for logvar
        pass

    def forward(self, data):
        """In: a batch of graphs. Out: mu and logvar (normalized velocity) of every query, shape (n_graphs,)."""
        # h = encoder(data.x)
        # for every layer: h = norm(h + activation(conv(h, data.edge_index, data.edge_attr)))   (residual connection)
        # h_query = h[data.query_mask]   -> exactly one node per graph, so one row per graph of the batch
        # mu = head_mu(h_query), logvar = head_logvar(h_query) clamped to [-8, 8] (keeps exp(logvar) finite)
        # return both with the last dimension removed (shape (n_graphs,))
        pass


################################################
#========= Section 3: Training Helpers ========#
################################################
# Used by main and train_model, in the order they're first called.

def set_seed(seed):
    """Seeds python, numpy and torch, so a run is reproducible.
    In: int. Out: nothing."""
    # random.seed(seed), np.random.seed(seed), torch.manual_seed(seed) (and torch.cuda.manual_seed_all on GPU)
    pass


def train_one_epoch(model, loader, optimizer, grad_clip, device):
    """One pass over the train loader with the NLL loss and gradient clipping.
    In: model, loader, optimizer, float, device. Out: mean train loss of the epoch."""
    # model.train()   (switches dropout on)
    # for every batch of the loader:
    #     batch = batch.to(device)
    #     optimizer.zero_grad()
    #     mu, logvar = model(batch), y = batch.y
    loss = loss_fn(mu, logvar, y)
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)   (no single batch can blow up the weights)
    #     optimizer.step()
    #     add loss * number of graphs of the batch to a running sum
    # return the running sum / number of graphs = mean loss per graph
    pass


def loss_fn(mu, logvar, y):
    """Heteroscedastic Gaussian NLL (§6.1).
    In: predicted mu/logvar, true y (normalized). Out: scalar loss tensor."""
    # per query: 0.5 * (logvar + (y - mu)^2 / exp(logvar))   (the constant 0.5 * log(2 pi) is left out)
    # -> a large error costs less if the model also predicts a large variance, but a large variance costs logvar
    # return the mean over the batch
    # hint: torch.nn.GaussianNLLLoss does the same with the variance exp(logvar) as input
    pass


def evaluate(model, loader, device):
    """Predicts all queries of a loader without gradients and computes the metrics (NLL, RMSE, MAE in mm/yr).
    In: model, loader, device. Out: dict of metrics and a DataFrame of predictions (region, orbit, y, mu, sigma)."""
    # model.eval() (dropout off) and torch.no_grad() (no gradients, faster, less memory)
    # for every batch: collect mu, logvar, y, region and orbit
    loss = loss_fn(mu, logvar, y)
    # back to mm/yr for the interpretable metrics: mu_mm = mu * std + mean, y_mm likewise, sigma_mm = exp(0.5 * logvar) * std
    # metrics = mean NLL (normalized units, the early stopping criterion), RMSE and MAE of mu_mm vs y_mm,
    #           and the share of y_mm inside mu_mm ± 1.645 sigma_mm (should be ~90 % if sigma fits; calibrated later in s06)
    # predictions = polars DataFrame with one row per query: region, orbit, y_mm, mu_mm, sigma_mm
    # return metrics, predictions
    # hint: the loader does not know the stats -> pass them in (e.g. via the dataset: loader.dataset.stats)
    pass


def save_checkpoint(model, params, epoch, path):
    """Saves the weights together with the arguments of the run and the epoch, so the model can be rebuilt later.
    In: model, dict, int, Path. Out: nothing (writes the .pt file)."""
    # create path.parent if it doesn't exist
    # torch.save({"state_dict": model.state_dict(), "params": params, "epoch": epoch}, path)
    # params must contain everything needed later (s05/s06/inference): the arguments of the run,
    # n_node_features, edge_dim and the stats (velocity, incidence and edge normalization)
    pass


################################################
#======= Section 4: Markdown Log Helper =======#
################################################
# Used at the end of a run, for the step's part of the markdown log (see s01.write_md_log).

def training_summary_md(results):
    """Builds the s04 part of the markdown log: dataset sizes, best epoch, train/dev metrics, dev metrics per region/orbit.
    In: dict returned by main. Out: markdown body (str) and figures (dict name -> matplotlib figure)."""
    # tables with s01.table_to_md:
    #   data: number of train pools (sampled / available), dev pools, GSE on/off, node features
    #   training: epochs run, best epoch, stopped early or at max_epochs, checkpoint path
    #   metrics: train vs dev (NLL, RMSE, MAE, share inside ±1.645 sigma) -> shows the generalisation gap
    #   dev per region and per orbit
    figures = {"loss_curves": xx.plot_loss_curves(results["history"])}
    # xx.plot_loss_curves (to be written in xx_plotting.py): train and dev loss per epoch, best epoch marked
    # return body, figures
    pass


##############################################################################
#============================================================================#
# 3) Processing
#============================================================================#
##############################################################################

def build_datasets(use_gse, n_graphs, k_graph, subset_min, subset_max, min_graph_size, seed):
    """Builds the train and dev datasets from the GRAPH files of the region table (n_graphs only limits train).
    In: the data arguments. Out: train PoolDataset, dev PoolDataset, stats dict."""
    study_areas_gdf = s01.study_areas_io("read", base_path=Path(__file__).parent / "../../assets/study_areas_gdf")
    # train_files = all paths in graph_file_list of the regions with set == 0, dev_files likewise for set == 1
    train_index = sample_pools(build_pool_index(train_files), n_graphs, seed)
    dev_index = build_pool_index(dev_files)
    stats = load_norm_stats()
    train_ds = PoolDataset(load_pools(train_index, use_gse, min_graph_size), stats, k_graph, subset_min, subset_max, min_graph_size, use_gse)
    dev_ds = PoolDataset(load_pools(dev_index, use_gse, min_graph_size), stats, k_graph, subset_min, subset_max, min_graph_size, use_gse,
                         fixed_draws=True)
    # the edge stats only come from train draws (never from dev, cal or val, §4.6)
    stats = estimate_edge_stats(train_ds)
    # give both datasets the complete stats: train_ds.stats = dev_ds.stats = stats
    # return train_ds, dev_ds, stats
    pass


def train_model(model, train_loader, dev_loader, lr, weight_decay, max_epochs, patience, grad_clip,
                device, checkpoint_path, params):
    """Trains with AdamW and early stopping on the dev loss; the best epoch is saved as checkpoint.
    In: model, loaders and the training arguments. Out: history (losses per epoch), best epoch."""
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # best_loss = infinity, best_epoch = None, epochs_without_improvement = 0, history = []
    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, grad_clip, device)
        dev_metrics, _ = evaluate(model, dev_loader, device)
        # history.append({"epoch", "train_loss", "dev_loss" = dev_metrics NLL, "dev_rmse"}); print it (job output)
        # if the dev loss is a new best:
        #     best_loss, best_epoch = dev loss, epoch; epochs_without_improvement = 0
        save_checkpoint(model, params, epoch, checkpoint_path)
        # else: epochs_without_improvement += 1
        # stop (break) when epochs_without_improvement reaches patience
    # return history, best_epoch
    pass


def evaluate_on_dev(model, checkpoint_path, train_loader, dev_loader, device):
    """Reloads the best checkpoint and evaluates it on train and dev, overall and per dev region and orbit.
    In: model, checkpoint path, loaders, device. Out: dict of metric tables (train vs dev, per region, per orbit)."""
    # model.load_state_dict(torch.load(checkpoint_path)["state_dict"])   -> the best epoch, not the last one
    train_metrics, _ = evaluate(model, train_loader, device)
    dev_metrics, dev_predictions = evaluate(model, dev_loader, device)
    # per region: group dev_predictions by region and compute RMSE, MAE and the ±1.645 sigma share per group
    # per orbit: the same grouped by orbit (ascending, descending)
    # return {"train": train_metrics, "dev": dev_metrics, "dev_per_region": ..., "dev_per_orbit": ...}
    pass


##############################################################################
#============================================================================#
# 4) Main
#============================================================================#
##############################################################################

def main(project_dir, use_gse, n_graphs, k_graph, subset_min, subset_max, min_graph_size,
         hidden_dim, n_layers, heads, dropout,
         lr, weight_decay, batch_size, max_epochs, patience, grad_clip,
         seed, num_workers, run_name):
    """Trains one model with one hyperparameter setting and evaluates it on dev (one SLURM array task of the search).
    In: the arguments of the run. Out: results dict (dataset sizes, history, best epoch, metrics) for the markdown log."""
    set_seed(seed)
    # run_name = timestamp if None; device = "cuda" if torch.cuda.is_available() else "cpu"
    # the weights go to <project-dir>/checkpoints/<run name>/ (the folder is created by s01)
    checkpoint_path = Path(project_dir) / "checkpoints" / run_name / "best_model.pt"
    train_ds, dev_ds, stats = build_datasets(use_gse, n_graphs, k_graph, subset_min, subset_max, min_graph_size, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # n_node_features = number of columns of train_ds[0].x (4, or 68 with GSE); edge_dim = 3
    model = InSARGNN(n_node_features, edge_dim, hidden_dim, n_layers, heads, dropout).to(device)
    # params = the arguments of the run + n_node_features, edge_dim and stats (saved with every checkpoint)
    history, best_epoch = train_model(model, train_loader, dev_loader, lr, weight_decay, max_epochs, patience, grad_clip,
                                      device, checkpoint_path, params)
    metrics = evaluate_on_dev(model, checkpoint_path, train_loader, dev_loader, device)
    # return {"n_train_pools", "n_dev_pools", "history", "best_epoch", "checkpoint_path", "metrics"}
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--project-dir", dest="project_dir", type=str, required=True,
                        help="project root (same as in s01), the weights go to <project-dir>/checkpoints/<run-name>/")
    parser.add_argument("--use-gse", dest="use_gse", action="store_true", default=False,
                        help="use the 64 GSE columns as node features (off by default: first runs without GSE)")
    parser.add_argument("--n-graphs", dest="n_graphs", type=int, default=None,
                        help="number of train pools drawn at random over all train regions (default: all)")
    parser.add_argument("--k-graph", dest="k_graph", type=int, default=3,
                        help="every node of a drawn subset is connected to its k nearest nodes")
    parser.add_argument("--subset-min", dest="subset_min", type=float, default=0.6,
                        help="smallest share of a pool's points drawn as subset per epoch")
    parser.add_argument("--subset-max", dest="subset_max", type=float, default=0.9,
                        help="largest share of a pool's points drawn as subset per epoch")
    parser.add_argument("--min-graph-size", dest="min_graph_size", type=int, default=7,
                        help="pools with fewer points are skipped, and no drawn subset gets smaller than this")

    # model
    parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=96,
                        help="width of the node embeddings")
    parser.add_argument("--n-layers", dest="n_layers", type=int, default=2,
                        help="number of message passing layers (hops)")
    parser.add_argument("--heads", dest="heads", type=int, default=4,
                        help="attention heads per GATv2 layer (hidden-dim must be divisible by it)")
    parser.add_argument("--dropout", dest="dropout", type=float, default=0.1,
                        help="dropout rate in the message passing layers")

    # training
    parser.add_argument("--lr", dest="lr", type=float, default=3e-4,
                        help="learning rate of AdamW")
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4,
                        help="weight decay of AdamW")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64,
                        help="number of graphs per batch")
    parser.add_argument("--max-epochs", dest="max_epochs", type=int, default=100,
                        help="upper bound on the number of epochs")
    parser.add_argument("--patience", dest="patience", type=int, default=10,
                        help="early stopping: epochs without a new best dev loss before training stops")
    parser.add_argument("--grad-clip", dest="grad_clip", type=float, default=5.0,
                        help="maximum gradient norm")

    # run
    parser.add_argument("--seed", dest="seed", type=int, default=42,
                        help="seed for the pool sample, the draws and the weight initialisation")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=8,
                        help="DataLoader worker processes")
    parser.add_argument("--run-name", dest="run_name", type=str, default=None,
                        help="name of the run for the checkpoint and the log, e.g. the SLURM array index (default: timestamp)")

    args = parser.parse_args()

    # Make sure the arguments have a valid value range
    if not 0 < args.subset_min <= args.subset_max <= 1:
        parser.error("--subset-min and --subset-max must satisfy 0 < min <= max <= 1")
    if args.min_graph_size < args.k_graph + 1:
        parser.error("--min-graph-size must be at least --k-graph + 1 (the query plus its k neighbours)")
    if args.hidden_dim % args.heads != 0:
        parser.error("--hidden-dim must be divisible by --heads")

    start = datetime.now()
    results = main(**vars(args))

    # the markdown log of the run, written only if the run got this far
    body, figures = training_summary_md(results)
    s01.write_md_log("s04_train_model", start, vars(args), body=body, figures=figures)
