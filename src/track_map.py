"""
track_map.py - VectorVance Track Graph Definition
──────────────────────────────────────────────────
Defines the physical track layout as a weighted graph.
Nodes = physical locations (start, forks, waypoints, destination).
Edges = road segments with distances (cm) and turn directions.

USAGE:
  Edit build_default_track() to match YOUR printed/taped track.
  Or call TrackMap() and add nodes/edges programmatically.

COORDINATE SYSTEM:
  x = left/right (cm from left edge of board)
  y = up/down   (cm from top edge of board, y increases downward)
  These are only for visualization — the car doesn't use absolute position.
"""


class Node:
    """A point on the track (intersection, waypoint, start, or destination)."""

    def __init__(self, name, x=0, y=0, node_type="waypoint"):
        """
        name:      unique string ID (e.g. "START", "FORK_1", "DEST")
        x, y:      position on the board in cm (for visualization only)
        node_type: "start" | "fork" | "waypoint" | "destination"
        """
        self.name = name
        self.x = x
        self.y = y
        self.node_type = node_type

    def __repr__(self):
        return f"Node({self.name}, type={self.node_type}, pos=({self.x},{self.y}))"


class Edge:
    """A road segment connecting two nodes."""

    def __init__(self, from_node, to_node, distance, turn_direction="STRAIGHT"):
        """
        from_node:      name of the source node
        to_node:        name of the destination node
        distance:       length of this road segment in cm
        turn_direction: what the car must do to enter this edge from the fork
                        "STRAIGHT" | "TURN_LEFT" | "TURN_RIGHT"
        """
        self.from_node = from_node
        self.to_node = to_node
        self.distance = distance
        self.turn_direction = turn_direction

    def __repr__(self):
        return (f"Edge({self.from_node} -> {self.to_node}, "
                f"{self.distance}cm, {self.turn_direction})")


class TrackMap:
    """
    Graph representation of the physical track.
    Supports bidirectional edges (the car can potentially go either way).
    """

    def __init__(self):
        self.nodes = {}       # name -> Node
        self.edges = []       # list of Edge
        self.adjacency = {}   # name -> [(neighbor_name, distance, turn_direction)]

    def add_node(self, name, x=0, y=0, node_type="waypoint"):
        """Add a location to the track."""
        node = Node(name, x, y, node_type)
        self.nodes[name] = node
        if name not in self.adjacency:
            self.adjacency[name] = []
        return node

    def add_edge(self, from_name, to_name, distance,
                 turn_from="STRAIGHT", turn_to="STRAIGHT",
                 bidirectional=True):
        """
        Connect two nodes with a road segment.

        turn_from: direction to take when traveling from_name -> to_name
        turn_to:   direction to take when traveling to_name -> from_name
        bidirectional: if True, car can travel both directions
        """
        assert from_name in self.nodes, f"Node '{from_name}' not found"
        assert to_name in self.nodes, f"Node '{to_name}' not found"

        edge_forward = Edge(from_name, to_name, distance, turn_from)
        self.edges.append(edge_forward)
        self.adjacency[from_name].append((to_name, distance, turn_from))

        if bidirectional:
            edge_back = Edge(to_name, from_name, distance, turn_to)
            self.edges.append(edge_back)
            self.adjacency[to_name].append((from_name, distance, turn_to))

    def get_neighbors(self, node_name):
        """Return list of (neighbor_name, distance, turn_direction)."""
        return self.adjacency.get(node_name, [])

    def get_node(self, name):
        return self.nodes.get(name)

    def get_forks(self):
        """Return all fork nodes (where a routing decision happens)."""
        return [n for n in self.nodes.values() if n.node_type == "fork"]

    def get_start(self):
        """Return the start node (assumes exactly one)."""
        for n in self.nodes.values():
            if n.node_type == "start":
                return n
        return None

    def get_destination(self):
        """Return the destination node (assumes exactly one)."""
        for n in self.nodes.values():
            if n.node_type == "destination":
                return n
        return None

    def print_map(self):
        """Pretty-print the track layout."""
        print("=" * 55)
        print("  TRACK MAP")
        print("=" * 55)
        print(f"  Nodes: {len(self.nodes)}")
        for n in self.nodes.values():
            marker = {"start": "🟢", "destination": "🏁",
                      "fork": "🔀", "waypoint": "⚪"}.get(n.node_type, "?")
            print(f"    {marker} {n.name:15s} ({n.x}, {n.y})  [{n.node_type}]")
        print(f"\n  Edges: {len(self.edges)}")
        for e in self.edges:
            print(f"    {e.from_node:15s} --> {e.to_node:15s}  "
                  f"{e.distance:5.1f}cm  {e.turn_direction}")
        print("=" * 55)


# ─────────────────────────────────────────────────────────────────────
#  PRE-BUILT TRACK LAYOUTS
#  Edit these to match your actual printed board!
# ─────────────────────────────────────────────────────────────────────

def build_default_track():
    """
    Default Y-shaped track:

        [DEST B] ←─ short path (45cm) ←─┐
                                         │
                                     [FORK F]
                                         │
                                     (30cm straight)
                                         │
                                     [START A]

        [DEST B] ←─ long path via W (70cm total) ←─ [FORK F]

    Distances are in cm — measure your actual track and update!
    """
    track = TrackMap()

    # ── NODES ────────────────────────────────────────────────────────
    track.add_node("START",   x=50, y=90, node_type="start")
    track.add_node("FORK",    x=50, y=60, node_type="fork")
    track.add_node("WP_LEFT", x=20, y=30, node_type="waypoint")    # short path waypoint
    track.add_node("WP_RIGHT",x=80, y=40, node_type="waypoint")    # long path waypoint
    track.add_node("DEST",    x=20, y=10, node_type="destination")

    # ── EDGES ────────────────────────────────────────────────────────
    # START → FORK (straight road, no turn needed)
    track.add_edge("START", "FORK", distance=30,
                   turn_from="STRAIGHT", turn_to="STRAIGHT")

    # FORK → left path (short route to destination)
    track.add_edge("FORK", "WP_LEFT", distance=25,
                   turn_from="TURN_LEFT", turn_to="TURN_RIGHT")

    track.add_edge("WP_LEFT", "DEST", distance=20,
                   turn_from="STRAIGHT", turn_to="STRAIGHT")

    # FORK → right path (long route to destination)
    track.add_edge("FORK", "WP_RIGHT", distance=35,
                   turn_from="TURN_RIGHT", turn_to="TURN_LEFT")

    track.add_edge("WP_RIGHT", "DEST", distance=35,
                   turn_from="TURN_LEFT", turn_to="TURN_RIGHT")

    return track


def build_complex_track():
    """
    More complex track with 3 possible routes:

    START ──(30)── FORK_1 ──(20)── FORK_2 ──(15)── DEST    (shortest: 65cm)
                     │                │
                     │(40)            │(25)
                     │                │
                   WP_A ───(30)──── WP_B ──(20)── DEST     (medium: 95cm)
                     │
                     │(50)
                     │
                   WP_C ──────────(35)──────────── DEST     (longest: 115cm)

    Edit distances to match your board!
    """
    track = TrackMap()

    track.add_node("START",  x=10, y=80, node_type="start")
    track.add_node("FORK_1", x=10, y=55, node_type="fork")
    track.add_node("FORK_2", x=50, y=55, node_type="fork")
    track.add_node("WP_A",   x=10, y=30, node_type="waypoint")
    track.add_node("WP_B",   x=50, y=30, node_type="waypoint")
    track.add_node("WP_C",   x=10, y=10, node_type="waypoint")
    track.add_node("DEST",   x=80, y=10, node_type="destination")

    track.add_edge("START",  "FORK_1", distance=30,
                   turn_from="STRAIGHT", turn_to="STRAIGHT")

    # Top route (shortest): FORK_1 → FORK_2 → DEST
    track.add_edge("FORK_1", "FORK_2", distance=20,
                   turn_from="TURN_RIGHT", turn_to="TURN_LEFT")
    track.add_edge("FORK_2", "DEST", distance=15,
                   turn_from="STRAIGHT", turn_to="STRAIGHT")

    # Middle route: FORK_1 → WP_A → WP_B → DEST
    track.add_edge("FORK_1", "WP_A", distance=40,
                   turn_from="STRAIGHT", turn_to="STRAIGHT")
    track.add_edge("WP_A",   "WP_B", distance=30,
                   turn_from="TURN_RIGHT", turn_to="TURN_LEFT")

    # Also connect FORK_2 to WP_B (cross-path)
    track.add_edge("FORK_2", "WP_B", distance=25,
                   turn_from="TURN_LEFT", turn_to="TURN_RIGHT")
    track.add_edge("WP_B",   "DEST", distance=20,
                   turn_from="STRAIGHT", turn_to="STRAIGHT")

    # Bottom route (longest): WP_A → WP_C → DEST
    track.add_edge("WP_A",   "WP_C", distance=50,
                   turn_from="STRAIGHT", turn_to="STRAIGHT")
    track.add_edge("WP_C",   "DEST", distance=35,
                   turn_from="TURN_RIGHT", turn_to="TURN_LEFT")

    return track


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── Default Y-Track ──")
    t1 = build_default_track()
    t1.print_map()

    print("\n── Complex Track ──")
    t2 = build_complex_track()
    t2.print_map()
