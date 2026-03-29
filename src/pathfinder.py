"""
pathfinder.py - VectorVance Shortest Path Solver
─────────────────────────────────────────────────
Uses Dijkstra's algorithm to find the shortest route
between any two nodes on the track graph.

Returns a turn-by-turn command list that the navigator can execute.

USAGE:
    from track_map import build_default_track
    from pathfinder import find_shortest_path, get_all_routes

    track = build_default_track()
    route = find_shortest_path(track, "START", "DEST")
    print(route)
"""

import heapq


def find_shortest_path(track_map, start_name, dest_name):
    """
    Dijkstra's algorithm — find the shortest path between two nodes.

    Returns:
        {
            'found':     True/False,
            'distance':  total distance in cm,
            'path':      ["START", "FORK", "WP_LEFT", "DEST"],
            'commands':  [
                {"action": "STRAIGHT", "to": "FORK", "distance": 30},
                {"action": "TURN_LEFT", "to": "WP_LEFT", "distance": 25},
                {"action": "STRAIGHT", "to": "DEST", "distance": 20},
            ]
        }
    """
    if start_name not in track_map.nodes:
        return {"found": False, "error": f"Start node '{start_name}' not in map"}
    if dest_name not in track_map.nodes:
        return {"found": False, "error": f"Destination '{dest_name}' not in map"}

    # ── Dijkstra ─────────────────────────────────────────────────────
    # dist[node] = shortest known distance from start
    dist = {name: float('inf') for name in track_map.nodes}
    dist[start_name] = 0

    # prev[node] = (previous_node, turn_direction, edge_distance)
    prev = {name: None for name in track_map.nodes}

    # Priority queue: (distance, node_name)
    pq = [(0, start_name)]
    visited = set()

    while pq:
        current_dist, current = heapq.heappop(pq)

        if current in visited:
            continue
        visited.add(current)

        # Found destination — reconstruct path
        if current == dest_name:
            return _reconstruct(track_map, prev, dist, start_name, dest_name)

        # Explore neighbors
        for neighbor, edge_dist, turn_dir in track_map.get_neighbors(current):
            if neighbor in visited:
                continue

            new_dist = current_dist + edge_dist
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = (current, turn_dir, edge_dist)
                heapq.heappush(pq, (new_dist, neighbor))

    # No path found
    return {"found": False, "error": f"No route from '{start_name}' to '{dest_name}'"}


def _reconstruct(track_map, prev, dist, start_name, dest_name):
    """Reconstruct the path and build turn-by-turn commands."""
    path = []
    commands = []
    current = dest_name

    while current != start_name:
        path.append(current)
        prev_node, turn_dir, edge_dist = prev[current]
        commands.append({
            "action": turn_dir,
            "to": current,
            "distance": edge_dist
        })
        current = prev_node

    path.append(start_name)
    path.reverse()
    commands.reverse()

    return {
        "found": True,
        "distance": dist[dest_name],
        "path": path,
        "commands": commands,
    }


def get_all_routes(track_map, start_name, dest_name, max_routes=10):
    """
    Find ALL possible routes using DFS, sorted by distance.
    Useful for comparing routes and showing why Dijkstra picked one.

    Returns list of:
        {"distance": cm, "path": [...], "commands": [...]}
    """
    if start_name not in track_map.nodes or dest_name not in track_map.nodes:
        return []

    all_routes = []

    def dfs(current, dest, visited, path, commands, total_dist):
        if current == dest:
            all_routes.append({
                "distance": total_dist,
                "path": list(path),
                "commands": list(commands),
            })
            return

        for neighbor, edge_dist, turn_dir in track_map.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                commands.append({
                    "action": turn_dir,
                    "to": neighbor,
                    "distance": edge_dist
                })

                dfs(neighbor, dest, visited, path, commands, total_dist + edge_dist)

                path.pop()
                commands.pop()
                visited.remove(neighbor)

    visited = {start_name}
    dfs(start_name, dest_name, visited, [start_name], [], 0)

    # Sort by distance (shortest first)
    all_routes.sort(key=lambda r: r["distance"])

    return all_routes[:max_routes]


def print_route(route, label=""):
    """Pretty-print a route result."""
    if label:
        print(f"\n── {label} ──")

    if not route.get("found", True):
        print(f"  ✗ {route.get('error', 'No route found')}")
        return

    path_str = " → ".join(route["path"])
    print(f"  Route:    {path_str}")
    print(f"  Distance: {route['distance']:.1f} cm")
    print(f"  Steps:    {len(route['commands'])}")

    for i, cmd in enumerate(route["commands"], 1):
        icon = {"STRAIGHT": "⬆️ ", "TURN_LEFT": "⬅️ ", "TURN_RIGHT": "➡️ "}.get(
            cmd["action"], "  "
        )
        print(f"    {i}. {icon}{cmd['action']:12s} → {cmd['to']:15s} ({cmd['distance']}cm)")


# ─────────────────────────────────────────────────────────────────────
#  SELF-TEST
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from track_map import build_default_track, build_complex_track

    # ── Test 1: Y-shaped track ───────────────────────────────────────
    print("=" * 60)
    print("  TEST 1: Y-Shaped Track (Default)")
    print("=" * 60)

    track = build_default_track()
    track.print_map()

    shortest = find_shortest_path(track, "START", "DEST")
    print_route(shortest, "SHORTEST PATH (Dijkstra)")

    all_routes = get_all_routes(track, "START", "DEST")
    print(f"\n── ALL {len(all_routes)} ROUTES (sorted by distance) ──")
    for i, route in enumerate(all_routes):
        tag = " ★ SHORTEST" if i == 0 else ""
        path_str = " → ".join(route["path"])
        print(f"  {i+1}. {route['distance']:5.1f}cm  {path_str}{tag}")

    # ── Test 2: Complex track ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TEST 2: Complex Track (3 routes)")
    print("=" * 60)

    track2 = build_complex_track()
    track2.print_map()

    shortest2 = find_shortest_path(track2, "START", "DEST")
    print_route(shortest2, "SHORTEST PATH (Dijkstra)")

    all_routes2 = get_all_routes(track2, "START", "DEST")
    print(f"\n── ALL {len(all_routes2)} ROUTES ──")
    for i, route in enumerate(all_routes2):
        tag = " ★ SHORTEST" if i == 0 else ""
        path_str = " → ".join(route["path"])
        print(f"  {i+1}. {route['distance']:5.1f}cm  {path_str}{tag}")
