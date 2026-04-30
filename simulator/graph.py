from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NODE_TYPE_TRANSFORMER = "transformer"
NODE_TYPE_HOME = "home"
NODE_TYPE_BATTERY = "battery"


@dataclass
class GraphNode:
    node_id: int
    node_type: str
    home_index: int | None = None


@dataclass
class GraphEdge:
    parent: int
    child: int
    length_m: float
    resistance_ohm: float


class ResidentialFeederGraph:
    def __init__(
        self,
        nodes: dict[int, GraphNode],
        parent_by_node: np.ndarray,
        edge_by_child: dict[int, GraphEdge],
        home_node_ids: np.ndarray,
        battery_node_id_by_home: np.ndarray,
        base_voltage_v: float,
        line_ampacity_a: float,
    ) -> None:
        self.nodes = nodes
        self.parent_by_node = parent_by_node
        self.edge_by_child = edge_by_child
        self.home_node_ids = home_node_ids
        self.battery_node_id_by_home = battery_node_id_by_home
        self.base_voltage_v = base_voltage_v
        self.line_ampacity_a = line_ampacity_a
        self.num_homes = len(home_node_ids)
        self.num_nodes = len(nodes)

    @classmethod
    def build_random_radial(
        cls,
        num_homes: int,
        home_has_battery: np.ndarray,
        resistance_ohm_per_km: float,
        min_edge_length_m: float,
        max_edge_length_m: float,
        base_voltage_v: float,
        line_ampacity_a: float,
        random_seed: int,
    ) -> "ResidentialFeederGraph":
        if home_has_battery.shape[0] != num_homes:
            raise ValueError("home_has_battery shape must match num_homes")

        rng = np.random.default_rng(random_seed)
        battery_count = int(np.sum(home_has_battery.astype(np.int64)))
        total_nodes = 1 + num_homes + battery_count

        parent_by_node = np.zeros(total_nodes, dtype=np.int64)
        parent_by_node[0] = -1

        nodes: dict[int, GraphNode] = {0: GraphNode(node_id=0, node_type=NODE_TYPE_TRANSFORMER)}
        edge_by_child: dict[int, GraphEdge] = {}
        home_node_ids = np.arange(1, num_homes + 1, dtype=np.int64)
        battery_node_id_by_home = np.full(num_homes, -1, dtype=np.int64)

        for home_idx, node in enumerate(home_node_ids):
            if home_idx == 0:
                parent = 0
            else:
                lower_bound = max(0, int(node) - 4)
                parent = int(rng.integers(lower_bound, int(node)))

            length_m = float(rng.uniform(min_edge_length_m, max_edge_length_m))
            resistance_ohm = resistance_ohm_per_km * (length_m / 1000.0)
            parent_by_node[int(node)] = parent
            nodes[int(node)] = GraphNode(
                node_id=int(node),
                node_type=NODE_TYPE_HOME,
                home_index=home_idx,
            )
            edge_by_child[int(node)] = GraphEdge(
                parent=parent,
                child=int(node),
                length_m=length_m,
                resistance_ohm=resistance_ohm,
            )

        next_node_id = num_homes + 1
        for home_idx, node in enumerate(home_node_ids):
            if not bool(home_has_battery[home_idx]):
                continue

            battery_node_id = int(next_node_id)
            next_node_id += 1
            battery_node_id_by_home[home_idx] = battery_node_id
            parent_by_node[battery_node_id] = int(node)
            nodes[battery_node_id] = GraphNode(
                node_id=battery_node_id,
                node_type=NODE_TYPE_BATTERY,
                home_index=home_idx,
            )
            edge_by_child[battery_node_id] = GraphEdge(
                parent=int(node),
                child=battery_node_id,
                length_m=1.0,
                resistance_ohm=0.001,
            )

        return cls(
            nodes=nodes,
            parent_by_node=parent_by_node,
            edge_by_child=edge_by_child,
            home_node_ids=home_node_ids,
            battery_node_id_by_home=battery_node_id_by_home,
            base_voltage_v=base_voltage_v,
            line_ampacity_a=line_ampacity_a,
        )

    def compute_network_state(
        self,
        net_grid_kw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        if net_grid_kw.shape[0] != self.num_homes:
            raise ValueError("net_grid_kw shape does not match number of homes")

        subtree_kw = np.zeros(self.num_nodes, dtype=float)
        subtree_kw[self.home_node_ids] = net_grid_kw

        for node in range(self.num_nodes - 1, 0, -1):
            parent = self.parent_by_node[node]
            subtree_kw[parent] += subtree_kw[node]

        branch_flow_kw = np.zeros(self.num_nodes, dtype=float)
        branch_loading = np.zeros(self.num_nodes, dtype=float)
        voltage_pu = np.ones(self.num_nodes, dtype=float)

        for node in range(1, self.num_nodes):
            parent = self.parent_by_node[node]
            edge = self.edge_by_child[node]

            branch_kw = subtree_kw[node]
            branch_flow_kw[node] = branch_kw
            current_a = (branch_kw * 1000.0) / max(self.base_voltage_v, 1e-6)
            delta_v = current_a * edge.resistance_ohm
            voltage_pu[node] = voltage_pu[parent] - (delta_v / max(self.base_voltage_v, 1e-6))
            branch_loading[node] = abs(current_a) / max(self.line_ampacity_a, 1e-6)

        feeder_total_kw = subtree_kw[0]
        home_voltage_pu = voltage_pu[self.home_node_ids]
        home_branch_flow_kw = branch_flow_kw[self.home_node_ids]
        max_branch_loading_pu = float(np.max(branch_loading[1:])) if self.num_nodes > 1 else 0.0
        return home_voltage_pu, home_branch_flow_kw, float(feeder_total_kw), max_branch_loading_pu
