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
from torch_geometric.utils import to_undirected

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
    """
    Lists all pools of the given GRAPH files. Only the seed rows of every file are read: every pool has
    exactly one seed, so the seeds' seed_pid values are exactly the pools of the file.

    Arguments:
        graph_files -- list of str or Path, GRAPH parquet files (of train or of dev)

    Returns:
        pool_index -- polars.DataFrame with the columns seed_pid and graph_file, one row per pool
    """

    # what happens in steps:
    # 1) the seed_pid column is read only where the seed flag is set (same as using unique on the pid col in our case)
    # 2) with .lit and alias every pid is associated with a filename, so later it is clear from which file to read (done in load_pools)
    # 3) .collect executes; for every graph file given to the function (either train or dev)
    # 4) pl.concat builds one list for all of train or dev
    tables = [pl.scan_parquet(path).filter(pl.col("seed") == 1).select("seed_pid").with_columns(pl.lit(str(path)).alias("graph_file")).collect() for path in graph_files]

    return pl.concat(tables)


def sample_pools(pool_index, n_graphs, seed):
    """
    Draws n_graphs pools at random over all files at once (fixed seed), so every region keeps its share of
    pools on average. Returns all pools if n_graphs is None or not smaller than the number of pools.

    Arguments:
        pool_index -- polars.DataFrame, the pool index (see build_pool_index)
        n_graphs -- int or None, number of pools to keep (--n-graphs)
        seed -- int, seed of the random draw

    Returns:
        sampled_index -- polars.DataFrame, the sampled rows of pool_index
    """

    # If not training dataset size is specified or the number exceeds the available 
    # number of graphs, return all available indices
    if n_graphs is None or n_graphs >= pool_index.height:
        return pool_index

    # local random generator
    rng = np.random.default_rng(seed)

    # Draws row positions not rows; replace=F to ensure no graph appears twice
    positions = rng.choice(pool_index.height, size=n_graphs, replace=False)

    # a random subsample of the input df 
    return pool_index[positions]


def load_pools(sampled_index, use_gse, min_graph_size):
    """
    Loads the points of the sampled pools into memory, reading every GRAPH file only once. Pools with fewer
    than min_graph_size points are skipped.

    Arguments:
        sampled_index -- polars.DataFrame, the pools to load (see sample_pools)
        use_gse -- bool, also load the 64 GSE columns
        min_graph_size -- int, minimum number of points of a pool

    Returns:
        pools -- list of dicts, one per pool: lat, lon (float64 arrays), vel, inc (float32 arrays), gse (int8 array of
            shape (m, 64), only with use_gse), region and orbit (str, from the folder names of the file)
    """

    # List of necessary columns to read for the process
    cols = ["latitude", "longitude", "mean_velocity", "incidence_angle", "seed_pid"]
    gse_cols = [f"gse_{i}" for i in range(64)]
    # if gse is used, the 64 dimensions are appended
    if use_gse:
        cols += gse_cols

    pools = []

    # grouped by graph_file so every file is opened once
    for (graph_file,), rows in sampled_index.group_by("graph_file"):
        # read the cols of a file
        df = pl.read_parquet(graph_file, columns=cols)
        # keep only the rows of the pools of this file that were sampled
        df = df.filter(pl.col("seed_pid").is_in(rows["seed_pid"].implode()))
        # one dataframe per pool; returns a dict.
        pools_in_file = df.partition_by("seed_pid", as_dict=True)
        # only accept pools that reach the minimum size
        for pool_df in pools_in_file.values():
            if pool_df.height < min_graph_size:
                continue
            # Adding the point information to the dictionary           
            pool = {
                    "lat": pool_df["latitude"].to_numpy(),                           # float64, needed for geodesics
                    "lon": pool_df["longitude"].to_numpy(),
                    "vel": pool_df["mean_velocity"].to_numpy().astype(np.float32),
                    "inc": pool_df["incidence_angle"].to_numpy().astype(np.float32),
                    "region": Path(graph_file).parent.parent.name,
                    "orbit": Path(graph_file).parent.name,
                    }
            if use_gse:
                pool["gse"] = pool_df.select(gse_cols).to_numpy()    # shape (m, 64)

            pools.append(pool)

    return pools


def load_norm_stats():
    """
    Reads the normalization stats of mean_velocity and incidence_angle that s02 computed on the train regions.

    Arguments:
        None -- reads s02.EGMS_NORM_STATS_PATH

    Returns:
        stats -- dict, {column: {"mean": float, "std": float}} for the columns in s02.EGMS_NORM_COLS
    """

    # open the json file where the velocity and incidence angle stats are saved
    with open(s02.EGMS_NORM_STATS_PATH) as f:
        stats = json.load(f)
    # EGMS_NORM_COLS are just velocity and incidence
    # Returns a dict that has mean & std for only those two
    return {c: stats[c] for c in s02.EGMS_NORM_COLS}


def estimate_edge_stats(dataset, n_draws=20000):
    """
    Estimates mean and std of the log edge distances from random draws of the train dataset (§4.6), so the
    edge distances can be normalized like the node features. Must be called before the dataset has edge stats,
    so the drawn graphs still carry the raw log(1 + distance).

    Arguments:
        dataset -- PoolDataset, the train dataset (never dev, cal or val)
        n_draws -- int, number of graphs drawn for the estimate

    Returns:
        stats -- dict, the dataset's stats with "edge_dist": {"mean", "std"} added
    """

    # every item of the dataset is one freshly drawn graph; n_draws of them (with replacement) give the sample for the stats
    graphs = [dataset[i] for i in np.random.default_rng(0).integers(0, len(dataset), n_draws)]

    # column 0 of edge_attr is the raw log(1 + dist); all edges of all drawn graphs are put into one long array
    d = torch.cat([g.edge_attr[:, 0] for g in graphs]).numpy()

    stats_d = {"mean": float(d.mean()), "std": float(d.std())}

    return {**dataset.stats, "edge_dist": stats_d}


class PoolDataset(torch.utils.data.Dataset):
    """
    The dataset of one set (train or dev): one item = one pool. Every __getitem__ builds a graph from its pool:
    a random subset of the points, one of them as query with its velocity hidden, k-nearest-neighbour edges and the
    node and edge features (masked target, §7.1). Train draws change every epoch, dev draws are fixed.

    Arguments (of __init__):
        pools -- list of dicts, the loaded pools (see load_pools)
        stats -- dict, normalization stats (see load_norm_stats, estimate_edge_stats)
        k_graph -- int, every node is connected to its k_graph nearest nodes
        subset_min, subset_max -- float, range of the share of a pool's points drawn per item
        min_graph_size -- int, minimum number of points of a drawn subset
        use_gse -- bool, add the 64 GSE values to the node features
        fixed_draws -- bool, True for dev: pool i always gives the same graph
        seed -- int, run seed of the train draws (draw of pool i in an epoch = f(seed, epoch, i))
    """

    def __init__(self, pools, stats, k_graph, subset_min, subset_max, min_graph_size, use_gse, fixed_draws=False, seed=0):
        # store all arguments as attributes (self.pools = pools, self.stats = stats, ...)
        # nothing is computed here: the graphs are built only when an item is requested
        # fixed_draws=True (used for dev): the random generator of item i is seeded with i, so every epoch evaluates
        # the same subsets and queries and the dev loss of different epochs is comparable for the early stopping
        # seed and epoch (train): the draw of pool i in an epoch only depends on (seed, epoch, i), see __getitem__
        self.seed = seed
        self.epoch = 0
        self.pools = pools
        self.stats = stats
        self.k_graph = k_graph
        self.subset_min = subset_min
        self.subset_max = subset_max
        self.min_graph_size = min_graph_size
        self.use_gse = use_gse
        self.fixed_draws = fixed_draws

    def set_epoch(self, epoch):
        # train_model calls this before every epoch; the DataLoader starts its workers anew every epoch (no
        # persistent_workers), so every worker copies the dataset with the current epoch
        """
        Sets the epoch the next train draws belong to, so every epoch draws new subsets and queries.

        Arguments:
            epoch -- int, the current epoch (see train_model)

        Returns:
            None -- updates self.epoch
        """
        self.epoch = epoch

    def __len__(self):
        # the number of pools: one item per pool and epoch
        """
        Number of items of the dataset, used by the DataLoader to know how many graphs one epoch has.

        Arguments:
            None

        Returns:
            n -- int, the number of pools (one graph per pool and epoch)
        """
        return len(self.pools)

    def __getitem__(self, i):
        """
        Builds one graph from pool i: random subset, query, k-NN edges, node and edge features, target.

        Arguments:
            i -- int, the pool number

        Returns:
            data -- torch_geometric Data with x (m, n_node_features), edge_index (2, n_edges), edge_attr (n_edges, 3),
                y (1,) the query's normalized velocity, query_mask (m,) True at the query, region and orbit (str)
        """

        # pulls one candidate pool from all pools to build a graph from it
        pool = self.pools[i]
        n = len(pool["vel"])
        lat, lon = pool["lat"], pool["lon"]

        # train: the generator is seeded with (run seed, epoch, pool), so every epoch draws a different subset and query,
        #        reproducibly and independent of which DataLoader worker draws the pool (and of --num-workers)
        # dev: the generator is seeded with i, so pool i always gives the same graph and the dev metrics stay comparable
        rng = np.random.default_rng(i) if self.fixed_draws else np.random.default_rng([self.seed, self.epoch, i])
        # draws the subset of the candidate pool for per epoch variation
        idx, q = draw_subset(n, self.subset_min, self.subset_max, self.min_graph_size, rng)

        # compute the distance and the orientation of the points
        dist, az = pairwise_geodesic(lat[idx], lon[idx])
        edge_index = knn_edges(dist, self.k_graph)

        # the edge and node features are put together using the normalization stats. 
        x = build_node_features(pool, idx, q, self.stats, self.use_gse)
        edge_attr = build_edge_features(dist, az, edge_index, self.stats)
        y = torch.tensor([(pool["vel"][idx][q] - self.stats["mean_velocity"]["mean"]) / self.stats["mean_velocity"]["std"]], dtype=torch.float32)

        # Building the query mask
        query_mask = torch.zeros(len(idx), dtype=torch.bool)
        query_mask[q] = True

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, query_mask=query_mask,
            region=pool["region"], orbit=pool["orbit"])


def draw_subset(n, subset_min, subset_max, min_graph_size, rng):
    """
    Draws the random subset of a pool's points and the query among them. The size of the subset is a random share
    of the pool, but at least min_graph_size points and at most all of them.

    Arguments:
        n -- int, number of points of the pool
        subset_min, subset_max -- float, range of the share of points drawn
        min_graph_size -- int, minimum size of the subset
        rng -- numpy.random.Generator, the generator of the item (see PoolDataset.__getitem__)

    Returns:
        idx -- numpy int array (m,), positions of the subset's points in the pool
        q -- int, position of the query within idx
    """
    
    # this draws a random fraction between the lower and upper limit
    share = rng.uniform(subset_min, subset_max)
    # convert the relative share into an integer; but always larger than min graph size and smaller than pool size
    m = min(n, max(min_graph_size, round(share * n)))

    # select the points that are in the subset
    idx = rng.choice(n, size=m, replace=False)
    # choose which node is the query node (also random per epoch)
    q = rng.integers(m)

    return idx, q


def pairwise_geodesic(lat, lon):
    """
    Computes the geodesic distance and bearing between all pairs of the subset's points (WGS84, GEOD).

    Arguments:
        lat, lon -- numpy float arrays (m,), coordinates of the subset's points

    Returns:
        dist -- numpy array (m, m), distance in metres from point i (row) to point j (column)
        az -- numpy array (m, m), bearing in degrees from point i to point j
    """

    # number of points for matrix dimension
    m = len(lat)
    # create all point combinations
    lat1, lon1 = np.repeat(lat, m), np.repeat(lon, m)
    lat2, lon2 = np.tile(lat, m), np.tile(lon, m)

    # compute all distances in one call (pyproj takes lon before lat); az12 = bearing from point 1 to point 2,
    # the back azimuth (point 2 to point 1) is not needed, it is the same matrix transposed
    az12, _, d = GEOD.inv(lon1, lat1, lon2, lat2)

    # back to (m, m): row i, column j = from point i to point j; the diagonal (a point with itself) has distance 0
    return d.reshape(m, m), az12.reshape(m, m)


def knn_edges(dist, k):
    """
    Connects every node to its k nearest nodes and symmetrises the edges (§7.1).

    Arguments:
        dist -- numpy array (m, m), pairwise distances (see pairwise_geodesic)
        k -- int, number of nearest neighbours per node

    Returns:
        edge_index -- torch long tensor (2, n_edges), row 0 = source, row 1 = target node of every edge
    """

    m = len(dist)

    # work on a copy, the caller still needs the original matrix for the edge features
    dist = dist.copy()
    # the diagonal is zero which would make the nodes nearest neighbour itself
    np.fill_diagonal(dist, np.inf)

    # computes the k nearest neighbours based on the distance matrix: row a holds the k nodes closest to a
    neighbours = np.argsort(dist, axis=1)[:, :k]

    # Turn it into a list of edges in the PyG convention: edge_index[0] = source, edge_index[1] = target,
    # messages flow from the source to the target, so every node receives from its k nearest neighbours
    target = np.repeat(np.arange(m), k)      # node a, repeated k times
    source = neighbours.reshape(-1)          # its k neighbours
    edge_index = torch.tensor(np.stack([source, target]), dtype=torch.long)

    # symmetrise: every edge also in the other direction, duplicates removed (a node can be among the k nearest of a
    # node that is not among its own k nearest, so some nodes end up with more than k edges)
    edge_index = to_undirected(edge_index)

    return edge_index


def build_node_features(pool, idx, q, stats, use_gse):
    """
    Builds the node feature matrix of a drawn subset: normalized velocity (0 at the query, its value is hidden),
    the is_query flag, normalized incidence angle and, optionally, the de-quantised GSE vector.

    Arguments:
        pool -- dict, the pool (see load_pools)
        idx -- numpy int array, positions of the subset's points in the pool
        q -- int, position of the query within idx
        stats -- dict, normalization stats of mean_velocity and incidence_angle
        use_gse -- bool, append the 64 GSE values

    Returns:
        x -- torch float32 tensor (m, 3), or (m, 67) with GSE
    """

    # get number of points for matrix dimensions
    m = len(idx)
    # get the normalization stats. 
    v_stats, i_stats = stats["mean_velocity"], stats["incidence_angle"]

    # normalize the velocity values here
    vel = (pool["vel"][idx] - v_stats["mean"]) / v_stats["std"]
    # set the query node to 0
    vel[q] = 0.0

    # creating the query flag column that tell the model where the query nodes sits
    is_query = np.zeros(m, dtype=np.float32)
    is_query[q] = 1.0

    # same normalization for the incidence angle column
    inc = (pool["inc"][idx] - i_stats["mean"]) / i_stats["std"]

    # if gse requested
    gse = dequantise_gse(pool["gse"][idx]) if use_gse else None

    # putting the columns together
    cols = [vel, is_query, inc]
    x = np.column_stack(cols + ([gse] if use_gse else []))

    return torch.tensor(x, dtype=torch.float32)


def dequantise_gse(v):
    """
    De-quantises the raw int8 GSE values to the unit-length embedding: x = ((v / 127.5) ** 2) * sign(v) (§4.1).

    Arguments:
        v -- numpy int8 array (m, 64), the raw GSE values

    Returns:
        x -- numpy float32 array (m, 64)
    """

    # convert to float32 what the model also takes
    v = v.astype(np.float32)
    # dequantize
    x = (v / 127.5) ** 2 * np.sign(v)

    return x


def build_edge_features(dist, az, edge_index, stats):
    """
    Builds the edge features: the normalized log distance and the bearing from the receiving to the sending node
    as its sine and cosine (circular, so 1° and 359° end up close together).

    Arguments:
        dist, az -- numpy arrays (m, m), pairwise distances and bearings (see pairwise_geodesic)
        edge_index -- torch long tensor (2, n_edges), the edges (see knn_edges)
        stats -- dict, normalization stats; without "edge_dist" the raw log(1 + distance) is returned

    Returns:
        edge_attr -- torch float32 tensor (n_edges, 3): [distance, sin, cos]
    """
    
    # the distance measure between two nodes is symmetrical, it doesnt matter whether you go ab or ba
    # the orientation measure is not, there the direction matters.
    # here we split the edge points into senders and receivers
    src, dst = edge_index.numpy()

    # picks the dist between a source and destination point and transforms it to log space
    d = np.log1p(dist[src, dst])

    # apply the edge statistics previously estimated
    if "edge_dist" in stats:
        d = (d - stats["edge_dist"]["mean"]) / stats["edge_dist"]["std"]

    # bearing from the receiving to the sending node, in radians because sin and cos expect radians
    bearing = np.radians(az[dst, src])

    # sin and cos describe the bearing well because they are naturally between -1 and 1,
    # and 1° and 359° are close together, which sin and cos convey (the raw angles do not)
    sin, cos = np.sin(bearing), np.cos(bearing)

    edge_attr = np.column_stack([d, sin, cos])
    return torch.tensor(edge_attr, dtype=torch.float32)


################################################
#============= Section 2: Model ===============#
################################################
# Used by main.

class InSARGNN(nn.Module):
    """
    The model: an MLP encoder, n_layers GATv2 message passing layers with residual connections and LayerNorm, and a
    heteroscedastic head that predicts the mean and log variance of the query's velocity (§5).

    Arguments (of __init__):
        n_node_features -- int, columns of x (3, or 67 with GSE)
        edge_dim -- int, columns of edge_attr (3)
        hidden_dim -- int, width of the node embeddings
        n_layers -- int, number of message passing layers
        heads -- int, attention heads per layer (hidden_dim must be divisible by it)
        dropout -- float, dropout rate of the attention coefficients
    """

    def __init__(self, n_node_features, edge_dim, hidden_dim, n_layers, heads, dropout):
        # registers the layers below as part of the model, so their weights are trained, moved to the GPU and saved
        super().__init__()

        # small MLP that lifts the few raw node features (3, or 67 with GSE) to hidden_dim dimensions
        self.encoder = nn.Sequential(nn.Linear(n_node_features, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

        # the message passing layers: every node aggregates its neighbours, weighted by attention that depends on
        # the node and edge features (distance, bearing); the heads are concatenated, so the output is hidden_dim again.
        # nn.ModuleList instead of a python list, otherwise the layers would not be registered
        self.convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
            for _ in range(n_layers)])

        # one LayerNorm per layer keeps the scale of the node embeddings stable over the layers
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        # the heteroscedastic head: one output for the mean and one for the log variance of the query's velocity
        self.head_mu = nn.Linear(hidden_dim, 1)
        self.head_logvar = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        # encode the node features
        """
        Runs a batch of graphs through the model and reads the prediction out at the query nodes.

        Arguments:
            data -- torch_geometric Batch with x, edge_index, edge_attr and query_mask

        Returns:
            mu -- torch tensor (n_graphs,), predicted normalized velocity of every query
            logvar -- torch tensor (n_graphs,), predicted log variance, clamped to [-8, 8]
        """
        h = self.encoder(data.x)

        # Message passing with residual connections: the layer's output is added to its input (h + ...), so a layer only
        # has to learn a correction and the information of the node itself is not lost
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + nn.functional.gelu(conv(h, data.edge_index, data.edge_attr)))

        # Read out the query node: every graph of the batch has exactly one, so this gives one row per graph
        h_query = h[data.query_mask]

        # squeeze removes the last dimension of size 1 -> shape (n_graphs,); the clamp keeps exp(logvar) finite
        mu = self.head_mu(h_query).squeeze(-1)
        logvar = self.head_logvar(h_query).squeeze(-1).clamp(-8, 8)

        return mu, logvar


################################################
#========= Section 3: Training Helpers ========#
################################################
# Used by main and train_model, in the order they're first called.

def set_seed(seed):
    """
    Seeds python, numpy and torch, so the weight initialisation and the order of the pools are reproducible.

    Arguments:
        seed -- int, the run seed (--seed)

    Returns:
        None
    """

    # The different random initializations here cover randomization of different libraries
    # or instances throughout the script
    # For pythons built in random
    random.seed(seed)
    # Numpy's global generator
    np.random.seed(seed)
    # torch's random generator (for the weight initialization for example)
    torch.manual_seed(seed)
    # torch.manual_seed already seeds the GPU as well, this call only makes it explicit (skipped without a GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_one_epoch(model, loader, optimizer, grad_clip, device):
    """
    Trains the model for one pass over the train loader: NLL loss, backward pass, gradient clipping, optimizer step.

    Arguments:
        model -- InSARGNN
        loader -- DataLoader over the train PoolDataset
        optimizer -- torch.optim.Optimizer
        grad_clip -- float, maximum gradient norm
        device -- str, "cuda" or "cpu"

    Returns:
        train_loss -- float, mean loss per graph of the epoch
    """

    # switch the model to training mode
    model.train()
    # set up counters
    total_loss, n_graphs = 0.0, 0

    # every batch is moved to the device (the GPU if available)
    for batch in loader:
        batch = batch.to(device)

        # set the gradients to zero, otherwise pytorch would add them up over the previous batches
        optimizer.zero_grad()

        # forward pass through the model and comparison with ground truth to compute loss
        mu, logvar = model(batch)
        loss = loss_fn(mu, logvar, batch.y)
        # backward pass
        loss.backward()
        # gradient clipping stops them from exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # update the weights
        optimizer.step()

        # loss is the mean over the batch: weighted with the batch size, so the epoch loss is the mean per graph
        # and a smaller last batch does not count as much as a full one
        total_loss += loss.item() * batch.num_graphs
        n_graphs += batch.num_graphs

    return total_loss / n_graphs


def loss_fn(mu, logvar, y):
    """
    Heteroscedastic Gaussian negative log likelihood (§6.1), without the constant term.

    Arguments:
        mu, logvar -- torch tensors (n,), predicted mean and log variance (normalized units)
        y -- torch tensor (n,), true normalized velocity

    Returns:
        loss -- torch scalar tensor, mean over the n queries
    """

    # per query: a large error costs less if the predicted variance is large too, but a large variance costs logvar,
    # so the model learns an honest uncertainty (the constant 0.5 * log(2 pi) is left out, it doesn't change the gradients)
    loss = 0.5 * (logvar + (y - mu) ** 2 / torch.exp(logvar))

    # mean over the queries of the batch
    return loss.mean()


def evaluate(model, loader, device):
    """
    Predicts all queries of a loader without gradients and computes the metrics: the NLL (normalized units, the
    early stopping criterion) and RMSE, MAE and the share inside mu ± 1.645 sigma (in mm/yr).

    Arguments:
        model -- InSARGNN
        loader -- DataLoader over a PoolDataset (its stats are used to go back to mm/yr)
        device -- str, "cuda" or "cpu"

    Returns:
        metrics -- dict with nll, rmse, mae and inside_90
        predictions -- polars.DataFrame, one row per query: region, orbit, y_mm, mu_mm, sigma_mm
    """

    # switch model to evaluation mode and turn off gradients
    model.eval()
    with torch.no_grad():
        # the results of every batch are collected 
        mus, logvars, ys, regions, orbits = [], [], [], [], []
        for batch in loader:
            batch = batch.to(device)
            mu, logvar = model(batch)
            mus.append(mu.cpu()); logvars.append(logvar.cpu()); ys.append(batch.y.cpu())
            regions.extend(batch.region); orbits.extend(batch.orbit)

        # One long tensor. One entry in the tensor is one query not an entire graph
        mu, logvar, y = torch.cat(mus), torch.cat(logvars), torch.cat(ys)

        nll = loss_fn(mu, logvar, y).item()

        # Undo the normalization back to interpretable units again
        v = loader.dataset.stats["mean_velocity"]
        mu_mm = mu * v["std"] + v["mean"]
        y_mm = y * v["std"] + v["mean"]
        sigma_mm = torch.exp(0.5 * logvar) * v["std"]

        # compute the metrics
        err = mu_mm - y_mm
        rmse = torch.sqrt((err ** 2).mean()).item()
        mae = err.abs().mean().item()
        # share of queries whose true value lies inside conf interval
        inside = (err.abs() <= 1.645 * sigma_mm).float().mean().item()

        # metrics to dict
        metrics = {"nll": nll, "rmse": rmse, "mae": mae, "inside_90": inside}

        # building prediction table
        predictions = pl.DataFrame({"region": regions, "orbit": orbits, "y_mm": y_mm.numpy(),
                            "mu_mm": mu_mm.numpy(), "sigma_mm": sigma_mm.numpy()})

    return metrics, predictions


def save_checkpoint(model, params, epoch, path):
    """
    Saves the weights together with everything needed to rebuild and use the model later (s05, s06, inference).

    Arguments:
        model -- InSARGNN
        params -- dict, the arguments of the run plus n_node_features, edge_dim and stats (see main)
        epoch -- int, the epoch of these weights
        path -- Path, the .pt file; its folder is created if missing

    Returns:
        None -- writes the checkpoint
    """

    # ensure the output path exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # state.dict() holds learned weights; params is the dict from main; epoch is at which epoch the best state was reached
    torch.save({"state_dict": model.state_dict(), "params": params, "epoch": epoch}, path)



################################################
#======= Section 4: Markdown Log Helper =======#
################################################
# Used at the end of a run, for the step's part of the markdown log (see s01.write_md_log).

def training_summary_md(results):
    """
    Builds the s04 part of the markdown log: data, training (epochs, best epoch, checkpoint), train vs dev metrics
    and the dev metrics per region and orbit, plus the loss curves as figure.

    Arguments:
        results -- dict, returned by main

    Returns:
        body -- str, markdown
        figures -- dict, {"loss_curves": matplotlib.figure.Figure} (see xx.plot_loss_curves)
    """
    history, metrics = results["history"], results["metrics"]

    # rounds the floats of a metrics row, so the tables stay readable (names and counts stay as they are)
    def rounded(row):
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}

    body = "\n\n".join([
        "## Data",
        s01.table_to_md([{"train pools": results["n_train_pools"], "dev pools": results["n_dev_pools"],
                          "GSE": "on" if results["use_gse"] else "off", "node features": results["n_node_features"]}]),
        "## Training",
        s01.table_to_md([{"epochs run": len(history), "best epoch": results["best_epoch"],
                          "stopped": "early" if len(history) < results["max_epochs"] else "at max_epochs",
                          "checkpoint": results["checkpoint_path"]}]),
        # train vs dev shows the generalisation gap
        "## Metrics, train vs dev",
        "NLL in normalized units, RMSE and MAE in mm/yr, inside_90 = share of queries inside mu ± 1.645 sigma.",
        s01.table_to_md([{"set": name, **rounded(metrics[name])} for name in ["train", "dev"]]),
        "## Dev per region",
        s01.table_to_md([rounded(r) for r in metrics["dev_per_region"]]),
        "## Dev per orbit",
        s01.table_to_md([rounded(r) for r in metrics["dev_per_orbit"]]),
    ])

    # train and dev loss per epoch, best epoch marked (xx.plot_loss_curves is still to be written in xx_plotting.py)
    figures = {"loss_curves": xx.plot_loss_curves(history)}

    return body, figures


##############################################################################
#============================================================================#
# 3) Processing
#============================================================================#
##############################################################################

def build_datasets(use_gse, n_graphs, k_graph, subset_min, subset_max, min_graph_size, seed):
    """
    Builds the train and dev datasets from the GRAPH files of the region table: pool index, sample (train only),
    loading the pools, the normalization stats from s02 and the edge stats estimated from train draws.

    Arguments:
        use_gse, n_graphs, k_graph, subset_min, subset_max, min_graph_size -- see main
        seed -- int, seed of the pool sample and of the train draws

    Returns:
        train_ds, dev_ds -- PoolDataset, with the complete stats
        stats -- dict, the normalization stats (velocity, incidence, edge distance)
    """

    # col "graph_file_list" holds a list of graph files per region 
    study_areas_gdf = s01.study_areas_io("read", base_path=Path(__file__).parent / "../../assets/study_areas_gdf")

    # if skips None regions; returns a flat list of graph files
    train_files = [f for files in study_areas_gdf.loc[study_areas_gdf["set"] == 0, "graph_file_list"] if files for f in files]
    dev_files = [f for files in study_areas_gdf.loc[study_areas_gdf["set"] == 1, "graph_file_list"] if files for f in files]

    # build_pool_index lists all the pools; sample_pools keeps a part of them in case we want to vary the 
    # size of the training dataset between runs.
    train_index = sample_pools(build_pool_index(train_files), n_graphs, seed)
    # dev is static to guarantee stable and comparable results
    dev_index = build_pool_index(dev_files)

    # Loading the velocity and incidence stats, no edge normalization stats yet. 
    stats = load_norm_stats()

    # Load pools reads the points into memory and PoolDataset creates the graphs (in dev, fixed_draws = True to keep them static)
    train_ds = PoolDataset(load_pools(train_index, use_gse, min_graph_size), stats, k_graph, subset_min, subset_max, min_graph_size, use_gse,
                           seed=seed)
    dev_ds = PoolDataset(load_pools(dev_index, use_gse, min_graph_size), stats, k_graph, subset_min, subset_max, min_graph_size, use_gse,
                             fixed_draws=True)
    
    # Now time to estimate the edge normalization statistics and assign them to the dataset
    stats = estimate_edge_stats(train_ds)
    train_ds.stats = dev_ds.stats = stats

    return train_ds, dev_ds, stats


def train_model(model, train_loader, dev_loader, lr, weight_decay, max_epochs, patience, grad_clip,
                device, checkpoint_path, params):
    """
    Trains with AdamW and early stopping on the dev NLL: every epoch new train draws, one training pass, one
    evaluation on dev; every new best epoch is saved as checkpoint, training stops after patience epochs without one.

    Arguments:
        model -- InSARGNN, already on device
        train_loader, dev_loader -- DataLoader
        lr, weight_decay -- float, AdamW settings
        max_epochs, patience -- int, upper bound on the epochs and the early stopping patience
        grad_clip -- float, maximum gradient norm
        device -- str, "cuda" or "cpu"
        checkpoint_path -- Path, file of the best checkpoint
        params -- dict, saved with the checkpoint (see save_checkpoint)

    Returns:
        history -- list of dicts, one per epoch: epoch, train_loss, dev_loss, dev_rmse
        best_epoch -- int, the epoch of the saved checkpoint
    """
    
    # create instance of the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # set up the bookkeeping for the early stopping
    best_loss, best_epoch, epochs_without_improvement, history = float("inf"), None, 0, []
    
    for epoch in range(max_epochs):
        # new draws for this epoch (the train draws depend on the run seed, the epoch and the pool, see PoolDataset)
        train_loader.dataset.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, grad_clip, device)
        dev_metrics, _ = evaluate(model, dev_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                "dev_loss": dev_metrics["nll"], "dev_rmse": dev_metrics["rmse"]})
        print(history[-1], flush=True)
        if dev_metrics["nll"] < best_loss:
            best_loss, best_epoch = dev_metrics["nll"], epoch
            epochs_without_improvement = 0
            save_checkpoint(model, params, epoch, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    return history, best_epoch


def evaluate_on_dev(model, checkpoint_path, train_loader, dev_loader, device):
    """
    Reloads the best checkpoint and evaluates it on train and dev, and on dev per region and per orbit.

    Arguments:
        model -- InSARGNN, its weights are replaced by the checkpoint's
        checkpoint_path -- Path, the best checkpoint (see train_model)
        train_loader, dev_loader -- DataLoader
        device -- str, "cuda" or "cpu"

    Returns:
        metrics -- dict with train and dev (see evaluate) and dev_per_region, dev_per_orbit (lists of dicts with
            n, rmse, mae, inside_90 per group)
    """

    # loads the best checkpoint not the last one 
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    # evaluates the model on train and dev
    train_metrics, _ = evaluate(model, train_loader, device)
    dev_metrics, dev_predictions = evaluate(model, dev_loader, device)

    # helper to aggregate the dev predictions by region or orbit: number of queries, RMSE, MAE and the share inside
    # mu ± 1.645 sigma per group, sorted by name, as list of dicts for the markdown tables
    def per_group(col):
        err = pl.col("mu_mm") - pl.col("y_mm")
        return (dev_predictions.group_by(col)
                .agg(pl.len().alias("n"),
                    (err ** 2).mean().sqrt().alias("rmse"),
                    err.abs().mean().alias("mae"),
                    (err.abs() <= 1.645 * pl.col("sigma_mm")).mean().alias("inside_90"))
                .sort(col).to_dicts())

    return {"train": train_metrics, "dev": dev_metrics,
        "dev_per_region": per_group("region"), "dev_per_orbit": per_group("orbit")}


##############################################################################
#============================================================================#
# 4) Main
#============================================================================#
##############################################################################

def main(project_dir, use_gse, n_graphs, k_graph, subset_min, subset_max, min_graph_size,
         hidden_dim, n_layers, heads, dropout,
         lr, weight_decay, batch_size, max_epochs, patience, grad_clip,
         seed, num_workers, run_name):
    """
    Trains one model with one hyperparameter setting and evaluates it on dev; one run of the hyperparameter search
    (one SLURM array task) or a single run.

    Arguments:
        project_dir -- str, project root, the weights go to <project_dir>/checkpoints/<run_name>/
        use_gse, n_graphs, k_graph, subset_min, subset_max, min_graph_size -- data arguments (see the argparse help)
        hidden_dim, n_layers, heads, dropout -- model arguments
        lr, weight_decay, batch_size, max_epochs, patience, grad_clip -- training arguments
        seed -- int, run seed
        num_workers -- int, DataLoader worker processes
        run_name -- str or None, name of the run (None = timestamp)

    Returns:
        results -- dict for the markdown log: dataset sizes, history, best epoch, checkpoint path, metrics,
            use_gse, n_node_features, max_epochs
    """
    
    # Setting the global seed eg. for weight initialization
    set_seed(seed)

    # If run_name wasnt specified, it will be the time stamp
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Computing on the gpu node if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(project_dir) / "checkpoints" / run_name / "best_model.pt"
                                           
    # Building the datasets from the precomputed candidate pools (the graphs and their edge features are built on the fly)
    # The normalization stats of the node features come from s02, the ones of the edge distances are estimated here
    train_ds, dev_ds, stats = build_datasets(use_gse, n_graphs, k_graph, subset_min, subset_max, min_graph_size, seed)
    
    # Defining the Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # To initialize the model, we need to know the dimensions of node and edge features 
    # extracting an example graph to derive the dimensions from it
    sample = train_ds[0]
    n_node_features = sample.x.shape[1]       # 3, or 67 with GSE
    edge_dim = sample.edge_attr.shape[1]      
    
    # n_node_features = number of columns of train_ds[0].x (3, or 67 with GSE); edge_dim = 3
    model = InSARGNN(n_node_features, edge_dim, hidden_dim, n_layers, heads, dropout).to(device)

    # everything needed to rebuild the model later (s05/s06/inference), saved with every checkpoint
    params = dict(use_gse=use_gse, n_graphs=n_graphs, k_graph=k_graph, subset_min=subset_min,
                  subset_max=subset_max, min_graph_size=min_graph_size, hidden_dim=hidden_dim,
                  n_layers=n_layers, heads=heads, dropout=dropout, lr=lr, weight_decay=weight_decay,
                  batch_size=batch_size, max_epochs=max_epochs, patience=patience, grad_clip=grad_clip,
                  seed=seed, n_node_features=n_node_features, edge_dim=edge_dim, stats=stats)

    # params = the arguments of the run + n_node_features, edge_dim and stats (saved with every checkpoint)
    history, best_epoch = train_model(model, train_loader, dev_loader, lr, weight_decay, max_epochs, patience, grad_clip,
                                      device, checkpoint_path, params)

    # Reload the best checkpoint and evaluate it on train and dev (overall, per dev region and orbit)
    metrics = evaluate_on_dev(model, checkpoint_path, train_loader, dev_loader, device)

    return {"n_train_pools": len(train_ds), "n_dev_pools": len(dev_ds), "history": history,
            "best_epoch": best_epoch, "checkpoint_path": checkpoint_path, "metrics": metrics,
            "use_gse": use_gse, "n_node_features": n_node_features, "max_epochs": max_epochs}


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
