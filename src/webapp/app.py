"""
app.py - VectorVance Web Dashboard Server
INSTALL:  pip install flask opencv-python numpy scikit-image
USAGE:    python app.py  ->  http://localhost:3000
"""

from flask import Flask, render_template, request, jsonify
import argparse, json, time, heapq, math
import cv2
import numpy as np

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────
#  IN-MEMORY STATE
# ─────────────────────────────────────────────────────────────────────
track_state = {
    "nodes": [
        {"id": "START", "x": 300, "y": 520, "type": "start"},
        {"id": "FORK",  "x": 300, "y": 340, "type": "fork"},
        {"id": "WP_L",  "x": 130, "y": 180, "type": "waypoint"},
        {"id": "WP_R",  "x": 480, "y": 220, "type": "waypoint"},
        {"id": "DEST",  "x": 130, "y": 60,  "type": "destination"},
    ],
    "edges": [
        {"from": "START", "to": "FORK", "dist": 30, "sign": None},
        {"from": "FORK",  "to": "WP_L", "dist": 25, "sign": "GREEN"},
        {"from": "FORK",  "to": "WP_R", "dist": 35, "sign": "BLUE"},
        {"from": "WP_L",  "to": "DEST", "dist": 20, "sign": None},
        {"from": "WP_R",  "to": "DEST", "dist": 35, "sign": None},
    ],
    "pi_ip":   "192.168.1.100",
    "pi_port": "5000",
}
last_command_sent = {"command": None, "timestamp": 0}


# ─────────────────────────────────────────────────────────────────────
#  PATHFINDING
# ─────────────────────────────────────────────────────────────────────
def dijkstra(nodes, edges, start_id, dest_id):
    adj = {n["id"]: [] for n in nodes}
    for e in edges:
        adj[e["from"]].append({"to": e["to"],  "dist": e["dist"], "sign": e.get("sign")})
        adj[e["to"]].append(  {"to": e["from"], "dist": e["dist"], "sign": e.get("sign")})
    dist    = {n["id"]: float("inf") for n in nodes}
    dist[start_id] = 0
    prev    = {n["id"]: None for n in nodes}
    visited = set()
    pq      = [(0, start_id)]
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u)
        if u == dest_id: break
        for nb in adj.get(u, []):
            nd = d + nb["dist"]
            if nd < dist[nb["to"]]:
                dist[nb["to"]] = nd; prev[nb["to"]] = u
                heapq.heappush(pq, (nd, nb["to"]))
    if dist[dest_id] == float("inf"): return None
    path, cur = [], dest_id
    while cur: path.append(cur); cur = prev[cur]
    path.reverse()
    path_edges = []
    for i in range(len(path) - 1):
        for e in edges:
            if (e["from"]==path[i] and e["to"]==path[i+1]) or \
               (e["to"]==path[i]   and e["from"]==path[i+1]):
                path_edges.append(e); break
    return {"path": path, "edges": path_edges, "distance": dist[dest_id]}


def all_routes(nodes, edges, start_id, dest_id):
    adj = {n["id"]: [] for n in nodes}
    for e in edges:
        adj[e["from"]].append({"to": e["to"],  "dist": e["dist"], "sign": e.get("sign"), "edge": e})
        adj[e["to"]].append(  {"to": e["from"], "dist": e["dist"], "sign": e.get("sign"), "edge": e})
    results = []
    def dfs(cur, vis, path, epath, total):
        if cur == dest_id:
            results.append({"path": list(path), "edges": list(epath), "distance": total})
            return
        for nb in adj.get(cur, []):
            if nb["to"] not in vis:
                vis.add(nb["to"]); path.append(nb["to"]); epath.append(nb["edge"])
                dfs(nb["to"], vis, path, epath, total + nb["dist"])
                path.pop(); epath.pop(); vis.discard(nb["to"])
    dfs(start_id, {start_id}, [start_id], [], 0)
    results.sort(key=lambda r: r["distance"])
    return results


# ─────────────────────────────────────────────────────────────────────
#  CV — ZHANG-SUEN THINNING (fallback if scikit-image absent)
# ─────────────────────────────────────────────────────────────────────
def _zhang_suen_thinning(binary):
    img = (binary > 0).astype(np.uint8)
    while True:
        prev = img.copy()
        for step in (1, 2):
            p  = img[1:-1,1:-1]
            p2,p3 = img[0:-2,1:-1],img[0:-2,2:]
            p4,p5 = img[1:-1,2:],  img[2:,2:]
            p6,p7 = img[2:,1:-1],  img[2:,0:-2]
            p8,p9 = img[1:-1,0:-2],img[0:-2,0:-2]
            B = (p2+p3+p4+p5+p6+p7+p8+p9).astype(int)
            seq = [p2,p3,p4,p5,p6,p7,p8,p9]
            A = sum(((seq[i]==0)&(seq[(i+1)%8]==1)).astype(int) for i in range(8))
            base = (p==1)&(B>=2)&(B<=6)&(A==1)
            cond = base&((p2*p4*p6==0)&(p4*p6*p8==0)) if step==1 \
                else base&((p2*p4*p8==0)&(p2*p6*p8==0))
            img[1:-1,1:-1] = p&(~cond)
        if np.array_equal(img, prev): break
    return img


# ─────────────────────────────────────────────────────────────────────
#  CV — ADAPTIVE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────
def _preprocess_track(img):
    """
    Returns (binary_mask uint8 0/255, dark_ratio float).
    Thin-outline (<5% dark): dilate lines before skeletonization.
    Thick-road  (>=5% dark): light morphological cleanup only.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_ratio = float(np.sum(binary > 0)) / (h * w)

    if dark_ratio < 0.05:
        r = max(12, min(h, w) // 55)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r*2+1, r*2+1))
        binary = cv2.dilate(binary, k)
        k_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_s, iterations=2)
    else:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=1)

    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n_lab > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary  = ((labels == largest) * 255).astype(np.uint8)
    return binary, dark_ratio


# ─────────────────────────────────────────────────────────────────────
#  CV — GRAPH SIMPLIFICATION PASSES
# ─────────────────────────────────────────────────────────────────────
_PRI = {"start": 3, "destination": 3, "fork": 2, "waypoint": 1}


def _assign_start_dest(nodes):
    """
    Mark the leftmost node as start, rightmost as destination.
    Never overwrites an already-assigned start/destination.
    """
    if not any(n["type"] == "start" for n in nodes):
        min(nodes, key=lambda n: n["x"])["type"] = "start"
    if not any(n["type"] == "destination" for n in nodes):
        for nd in sorted(nodes, key=lambda n: n["x"], reverse=True):
            if nd["type"] != "start":
                nd["type"] = "destination"; break
    return nodes


def _merge_symmetric_forks(nodes, edges):
    """
    Detect fork nodes that have the EXACT same set of neighbors and merge them.

    Why this matters
    ────────────────
    A Y-shaped starting corridor produces two skeleton junctions — one for the
    top wall and one for the bottom wall — that both connect to the same two
    other nodes (start and end).  These are topologically the same node and
    must be merged so Dijkstra sees two genuine parallel paths fork→end, not
    two separate intermediate nodes.

    After merging:
      - All edges from the dropped node are redirected to the kept node.
      - This intentionally creates DUPLICATE edges for the same node pair
        (e.g. two fork→end edges), which represent alternative paths.
      - Position is averaged.

    Runs iteratively until no symmetric pairs remain.
    """
    changed = True
    while changed:
        changed = False
        adj = {n["id"]: set() for n in nodes}
        for e in edges:
            adj[e["from"]].add(e["to"])
            adj[e["to"]].add(e["from"])

        fork_nodes = [n for n in nodes if n["type"] == "fork"]
        for i in range(len(fork_nodes)):
            for j in range(i + 1, len(fork_nodes)):
                ni, nj = fork_nodes[i], fork_nodes[j]
                if adj[ni["id"]] != adj[nj["id"]]:
                    continue   # different neighbors → not symmetric

                # Merge nj INTO ni (average position)
                ni["x"] = (ni["x"] + nj["x"]) // 2
                ni["y"] = (ni["y"] + nj["y"]) // 2
                keep_id, drop_id = ni["id"], nj["id"]

                nodes = [n for n in nodes if n["id"] != drop_id]
                new_edges = []
                for e in edges:
                    nf = keep_id if e["from"] == drop_id else e["from"]
                    nt = keep_id if e["to"]   == drop_id else e["to"]
                    if nf == nt: continue   # self-loop → remove
                    new_edges.append({**e, "from": nf, "to": nt})
                edges = new_edges
                changed = True
                break
            if changed: break

    return nodes, edges


def _collapse_start_side_edges(nodes, edges):
    """
    For each pair of nodes with more than one edge between them:
      - If the START node is one endpoint → corridor walls → keep the LONGEST only.
      - Otherwise                         → genuine alternative paths → keep ALL.

    This distinction is crucial:
      start→fork edges are corridor duplicates (top and bottom wall of the same
      channel), while fork→dest edges are real path choices (path A vs path B).
    """
    start_ids = {n["id"] for n in nodes if n["type"] == "start"}

    groups: dict = {}
    for e in edges:
        key = tuple(sorted([e["from"], e["to"]]))
        groups.setdefault(key, []).append(e)

    result = []
    for key, group in groups.items():
        if len(group) == 1:
            result.extend(group); continue
        nA, nB = key
        if nA in start_ids or nB in start_ids:
            # Corridor walls → keep one (longest = most representative path length)
            result.append(max(group, key=lambda x: x["dist"]))
        else:
            # Parallel alternative paths → keep all
            result.extend(group)
    return result


def _merge_nearby_nodes(nodes, edges, threshold=80):
    """Merge any two nodes within threshold SVG pixels of each other."""
    changed = True
    while changed:
        changed = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                ni, nj = nodes[i], nodes[j]
                if math.hypot(ni["x"]-nj["x"], ni["y"]-nj["y"]) >= threshold:
                    continue
                keep, drop = (j,i) if _PRI.get(nj["type"],0) > _PRI.get(ni["type"],0) else (i,j)
                keep_id, drop_id = nodes[keep]["id"], nodes[drop]["id"]
                kn = nodes[keep]; dn = nodes[drop]
                kn["x"] = (kn["x"] + dn["x"]) // 2
                kn["y"] = (kn["y"] + dn["y"]) // 2
                nodes = [n for n in nodes if n["id"] != drop_id]
                new_edges, seen = [], set()
                for e in edges:
                    nf = keep_id if e["from"]==drop_id else e["from"]
                    nt = keep_id if e["to"]  ==drop_id else e["to"]
                    if nf == nt: continue
                    new_edges.append({**e, "from": nf, "to": nt})
                edges = new_edges
                changed = True; break
            if changed: break
    return nodes, edges


def _assign_signs(nodes, edges):
    """
    For each fork node, find outgoing edges that share the same other endpoint.
    Sort by distance and assign GREEN (shorter) / BLUE (longer).
    """
    for e in edges: e["sign"] = None
    adj = {n["id"]: [] for n in nodes}
    for e in edges: adj[e["from"]].append(e); adj[e["to"]].append(e)

    for n in nodes:
        if n["type"] != "fork": continue
        fid = n["id"]
        groups: dict = {}
        for e in adj[fid]:
            other = e["to"] if e["from"] == fid else e["from"]
            groups.setdefault(other, []).append(e)
        for group in groups.values():
            if len(group) < 2: continue
            group.sort(key=lambda x: x["dist"])
            group[0]["sign"] = "GREEN"
            group[1]["sign"] = "BLUE"
    return edges


# ─────────────────────────────────────────────────────────────────────
#  CV — SKELETON GRAPH EXTRACTION
# ─────────────────────────────────────────────────────────────────────
def _extract_graph(skel_01, orig_h, orig_w, svg_w=620, svg_h=600,
                   min_edge_px=20, is_thin_outline=False):
    """
    Skeleton image → clean VectorVance node/edge graph.

    Pipeline
    ────────
    1.  Per-pixel 8-neighbor count → junction / endpoint masks.
    2.  Cluster key pixels into raw nodes.
        Thin-outline: larger cluster radius so end-circles collapse early.
    3.  Strip node blobs → edge segments; touch-detect node pairs.
    4.  Map to SVG coordinates.
    5.  Assign start (leftmost x) / destination (rightmost x) early.
    6.  _merge_symmetric_forks: merge fork nodes with identical neighbors.
        This is the key step that converts the "diamond" topology of a
        parallel-corridor track into the correct 3-node fork graph.
    7.  _collapse_start_side_edges: collapse corridor-wall duplicates
        (start-connected pairs keep only one); keep fork→dest pairs.
    8.  _merge_nearby_nodes: final proximity safety net.
    9.  Re-confirm start/dest; assign GREEN/BLUE signs.
    """
    h, w = skel_01.shape
    sx, sy = svg_w / orig_w, svg_h / orig_h

    # 1. Neighbor masks
    k_nb = np.ones((3,3), np.float32); k_nb[1,1] = 0.0
    nb   = (cv2.filter2D(skel_01.astype(np.float32), -1, k_nb) * skel_01).astype(np.int32)
    junc_mask = (nb >= 3) & (skel_01 > 0)
    endp_mask = (nb == 1) & (skel_01 > 0)
    key_mask  = junc_mask | endp_mask

    # 2. Cluster into raw nodes
    if is_thin_outline:
        cluster_r = max(20, min(h, w) // 14)   # ~77 px for 1080-tall image
    else:
        cluster_r = max(15, min(h, w) // 30)

    dil_k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cluster_r*2+1, cluster_r*2+1))
    key_dil = cv2.dilate(key_mask.astype(np.uint8), dil_k)
    n_comp, comp_labels = cv2.connectedComponents(key_dil)

    raw_nodes, raw_types = [], []
    for cid in range(1, n_comp):
        cm = comp_labels == cid
        ky, kx = np.where(cm & key_mask)
        if len(ky) == 0: ky, kx = np.where(cm)
        raw_nodes.append((int(np.mean(ky)), int(np.mean(kx))))
        raw_types.append("junction" if np.any(cm & junc_mask) else "endpoint")

    if len(raw_nodes) < 2: return [], []

    # 3. Edge segments → node-pair detection
    edge_skel     = (skel_01 & (~key_dil)).astype(np.uint8)
    n_e, e_labels = cv2.connectedComponents(edge_skel)
    node_masks    = [(comp_labels==cid).astype(np.uint8) for cid in range(1, n_comp)]
    touch_k       = np.ones((9, 9), np.uint8)
    raw_edges, seen_pairs = [], set()

    for eid in range(1, n_e):
        em = (e_labels==eid).astype(np.uint8)
        px = int(np.sum(em))
        if px < min_edge_px: continue
        ed = cv2.dilate(em, touch_k)
        touching = [i for i, nm in enumerate(node_masks) if np.any(ed & nm)]
        if len(touching) == 2:
            pair = tuple(sorted(touching))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                raw_edges.append({"from": pair[0], "to": pair[1], "px_len": px})

    # 4. SVG mapping
    max_px = max((e["px_len"] for e in raw_edges), default=1)
    nodes  = [{
        "id":   f"N{i}",
        "x":    int(min(svg_w-20, max(20, cx*sx))),
        "y":    int(min(svg_h-20, max(20, cy*sy))),
        "type": "fork" if t=="junction" else "waypoint",
    } for i, ((cy,cx), t) in enumerate(zip(raw_nodes, raw_types))]
    edges  = [{
        "from": nodes[e["from"]]["id"],
        "to":   nodes[e["to"]]["id"],
        "dist": max(5, int(e["px_len"] * 100 / max_px)),
        "sign": None,
    } for e in raw_edges]

    # 5. Assign start/dest early (leftmost/rightmost x)
    nodes = _assign_start_dest(nodes)

    # 6. Merge symmetric fork pairs → creates parallel fork→dest edges
    nodes, edges = _merge_symmetric_forks(nodes, edges)
    if not nodes: return [], []

    # 7. Collapse corridor duplicates on start side; keep fork→dest parallels
    edges = _collapse_start_side_edges(nodes, edges)

    # 8. Proximity merge (safety net)
    nodes, edges = _merge_nearby_nodes(nodes, edges, threshold=min(svg_w, svg_h)//8)
    if not nodes: return [], []

    # 9. Re-confirm start/dest, assign GREEN/BLUE
    nodes = _assign_start_dest(nodes)
    edges = _assign_signs(nodes, edges)

    return nodes, edges


# ─────────────────────────────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/track", methods=["GET"])
def get_track():
    nodes = track_state["nodes"]
    edges = track_state["edges"]
    start = next((n for n in nodes if n["type"]=="start"),       None)
    dest  = next((n for n in nodes if n["type"]=="destination"), None)
    shortest, routes, chosen_sign = None, [], None
    if start and dest:
        shortest = dijkstra(nodes, edges, start["id"], dest["id"])
        routes   = all_routes(nodes, edges, start["id"], dest["id"])
        if shortest:
            fe = next((e for e in shortest["edges"] if e.get("sign")), None)
            chosen_sign = fe["sign"] if fe else None
    return jsonify({
        "nodes": nodes, "edges": edges, "shortest": shortest,
        "routes": routes, "chosen_sign": chosen_sign,
        "pi_ip": track_state["pi_ip"], "pi_port": track_state["pi_port"],
    })


@app.route("/api/track", methods=["POST"])
def update_track():
    data = request.get_json()
    for k in ("nodes","edges","pi_ip","pi_port"):
        if k in data: track_state[k] = data[k]
    return jsonify({"status": "ok"})


@app.route("/api/send", methods=["POST"])
def send_to_pi():
    import urllib.request, urllib.error
    data  = request.get_json()
    color, route = data.get("color"), data.get("route", [])
    if not color: return jsonify({"error": "No color specified"}), 400
    pi_url  = f"http://{track_state['pi_ip']}:{track_state['pi_port']}/command"
    payload = json.dumps({"action":"FOLLOW_COLOR","color":color,"route":route}).encode()
    try:
        req = urllib.request.Request(pi_url, data=payload,
              headers={"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read().decode())
        last_command_sent.update({"command":{"color":color,"route":route},"timestamp":time.time()})
        return jsonify({"status":"sent","pi_response":result})
    except urllib.error.URLError as e:
        return jsonify({"status":"error","message":f"Cannot reach Pi: {e}"}), 502
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500


# ─────────────────────────────────────────────────────────────────────
#  SCAN-TRACK
# ─────────────────────────────────────────────────────────────────────
@app.route("/api/scan-track", methods=["POST"])
def scan_track():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400

    h, w = img.shape[:2]
    binary, dark_ratio = _preprocess_track(img)
    is_thin = dark_ratio < 0.05
    track_style = "thin-outline" if is_thin else "thick-road"

    try:
        from skimage.morphology import skeletonize as sk_sk
        skel_01 = sk_sk(binary // 255).astype(np.uint8)
        method  = f"scikit-image [{track_style}]"
    except ImportError:
        skel_01 = _zhang_suen_thinning(binary)
        method  = f"zhang-suen [{track_style}]"

    if not np.any(skel_01):
        return jsonify({"error":
            f"Skeletonization produced no output. "
            f"Ensure the track is darker than the background. "
            f"[style={track_style}, dark_ratio={dark_ratio:.3f}]"}), 400

    min_edge_px = max(15, (h * w) // 80_000)
    new_nodes, new_edges = _extract_graph(
        skel_01, h, w, min_edge_px=min_edge_px, is_thin_outline=is_thin
    )

    if not new_nodes:
        return jsonify({"error":
            f"Could not detect any track nodes. "
            f"Ensure junctions/endpoints are visible. "
            f"[style={track_style}, skel_px={int(skel_01.sum())}]"}), 400

    track_state["nodes"] = new_nodes
    track_state["edges"] = new_edges

    return jsonify({
        "status":      "ok",
        "nodes_found": len(new_nodes),
        "edges_found": len(new_edges),
        "method":      method,
        "dark_ratio":  round(dark_ratio, 4),
    })


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VectorVance Dashboard")
    parser.add_argument("--port",  type=int, default=3000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"\n  VectorVance  http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)