import torch
from torch_geometric.utils import subgraph

# ----------------------------
# BFS removal (true version)
# ----------------------------
def bfs_remove(adj, start, target):
    visited = set()
    queue = [start]
    removed = []

    while queue and len(removed) < target:
        node = queue.pop(0)

        if node in visited:
            continue

        visited.add(node)
        removed.append(node)

        for nbr in adj[node]:
            if nbr not in visited:
                queue.append(nbr)

    return removed


# ----------------------------
# transform function (SINGLE GRAPH)
# ----------------------------
def subgraph_removal_transform(data, percent=0.2):
    device = data.x.device
    num_nodes = data.num_nodes
    target_remove = max(1, int(num_nodes * percent))

    # ---- build adjacency (global index) ----
    adj = {i: [] for i in range(num_nodes)}

    src, dst = data.edge_index.tolist()
    for s, d in zip(src, dst):
        adj[s].append(d)

    # ---- BFS start ----
    start = torch.randint(0, num_nodes, (1,), device=device).item()
    remove_nodes = bfs_remove(adj, start, target_remove)

    remove_nodes = torch.tensor(remove_nodes, dtype=torch.long, device=device)

    # ---- keep nodes ----
    mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    mask[remove_nodes] = False
    keep_nodes = mask.nonzero(as_tuple=False).view(-1)

    # ---- subgraph (IMPORTANT: NO offset, NO local index) ----
    edge_index, edge_attr = subgraph(
        keep_nodes,
        data.edge_index,
        edge_attr=data.edge_attr if hasattr(data, "edge_attr") else None,
        relabel_nodes=True,
        num_nodes=num_nodes
    )

    # ---- build new graph ----
    data = data.clone()
    data.x = data.x[keep_nodes]
    data.edge_index = edge_index
    data.edge_attr = edge_attr

    # ---- fix batch if exists ----
    if hasattr(data, "batch") and data.batch is not None:
        data.batch = data.batch[keep_nodes]

    return data