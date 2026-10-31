"""Fixed sensor graph for the SPIN-H Task 1 experiment.

The released LWR topology lists road links, while SPIN-H operates on detector
nodes.  Some consecutive detectors are separated by unobserved connector links.
We therefore connect consecutive ``order_index`` detectors in travel direction,
then add every explicit sensor-to-sensor edge in ``incoming_link_ids`` and
``outgoing_link_ids``.  The official SPIN runner requests a symmetric sensor
graph, so each physical connection is represented by two directed messages.
No traffic measurements or target labels enter the graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from src.io import paths
from src.task1.staeformer_data import PanelInfo


@dataclass(frozen=True)
class SensorGraph:
    panel: str
    edge_index: np.ndarray  # [2, E], source then destination
    ordered_edges: int
    explicit_edges: int
    symmetric: bool

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]


def _split_links(value: str | None) -> list[str]:
    return [] if not value else [part for part in value.split(";") if part]


def build_sensor_graph(info: PanelInfo, *, force_symmetric: bool = True) -> SensorGraph:
    topology = pl.read_csv(paths.network_dir(info.panel) / "lwr_mainline_topology.csv")
    required = {"link_id", "order_index", "incoming_link_ids", "outgoing_link_ids"}
    if not required.issubset(topology.columns):
        raise ValueError(f"{info.panel}: topology lacks {required - set(topology.columns)}")
    ordered = topology.sort("order_index")["link_id"].to_list()
    if tuple(ordered) != info.link_ids or len(set(ordered)) != info.n_links:
        raise ValueError(f"{info.panel}: topology order differs from PanelInfo")
    node = {link: i for i, link in enumerate(info.link_ids)}

    # order_index follows milepost.  North/east travel increases it; south/west
    # travel decreases it, as confirmed by the topology's explicit out-edges.
    forward = info.panel.endswith(("N", "E"))
    ordered_edges = {
        (i, i + 1) if forward else (i + 1, i)
        for i in range(info.n_links - 1)
    }
    explicit: set[tuple[int, int]] = set()
    for row in topology.select("link_id", "incoming_link_ids", "outgoing_link_ids").iter_rows(named=True):
        here = node[row["link_id"]]
        for upstream in _split_links(row["incoming_link_ids"]):
            if upstream in node and node[upstream] != here:
                explicit.add((node[upstream], here))
        for downstream in _split_links(row["outgoing_link_ids"]):
            if downstream in node and node[downstream] != here:
                explicit.add((here, node[downstream]))

    edges_set = ordered_edges | explicit
    if force_symmetric:
        edges_set |= {(dst, src) for src, dst in tuple(edges_set)}
    edges = sorted(edges_set)
    edge_index = np.asarray(edges, dtype=np.int64).T.copy()
    if edge_index.shape != (2, len(edges)) or len(edges) < info.n_links - 1:
        raise ValueError(f"{info.panel}: invalid sensor graph")
    return SensorGraph(info.panel, edge_index, len(ordered_edges), len(explicit), force_symmetric)
