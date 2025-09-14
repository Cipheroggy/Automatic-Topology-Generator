import os
import ipaddress
import random
import networkx as nx
import matplotlib.pyplot as plt

# ==============================
# CONFIG — adjust as you like
# ==============================
CONFIG_DIR = r"D:\CISCO\Conf"   # Conf/R1/config.dump, Conf/R2/config.dump, ...
SAVE_PNG = True
OUTPUT_PNG = "topology.png"

# Traffic modes: "apps" (application-aware), "random" (range), "fixed" (single fixed value)
LOAD_MODE = "apps"

# Random traffic range (kbps) if LOAD_MODE == "random"
TRAFFIC_MIN = 2000
TRAFFIC_MAX = 15000

# Fixed traffic (kbps) if LOAD_MODE == "fixed"
FIXED_LOAD = 8000

# If you want reproducible loads for screenshots, set a seed (None = fresh every run)
RANDOM_SEED = None  # e.g., 42

# Default bandwidth assumptions (kbps) when config lacks 'bandwidth' line
DEFAULT_ROUTER_IF_BW = 10000     # router↔router or router↔switch FastEthernet
DEFAULT_ENDPOINT_LINK_BW = 1000  # switch↔endpoint

# How many endpoints (PC/Server) to hang per inferred access subnet
ENDPOINTS_PER_LAN = 2

# Application profiles (OPTIONAL feature from the PDF):
APP_PROFILES = {
    "Web Browsing":    {"peak": 2000, "avg": 500},
    "Video Streaming": {"peak": 8000, "avg": 5000},
    "VoIP":            {"peak": 512,  "avg": 256},
    "File Transfer":   {"peak": 9000, "avg": 6000},
    "Cloud Backup":    {"peak": 7000, "avg": 4000}
}
USE_PEAK = True  # if LOAD_MODE == "apps": use peak (True) or average (False)


# ==============================
# STEP 1: Parse router configs
# ==============================
def parse_config(file_path):
    """
    Parse one router config.dump to extract:
    - hostname
    - interfaces: name, ip, mask, bandwidth (kbps if 'bandwidth <num>' present)
    """
    data = {"hostname": None, "interfaces": []}
    current = None
    with open(file_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("hostname"):
                parts = line.split()
                if len(parts) >= 2:
                    data["hostname"] = parts[1]

            elif line.startswith("interface"):
                if current:
                    data["interfaces"].append(current)
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    current = {"name": parts[1], "ip": None, "mask": None, "bandwidth": None}
                else:
                    current = {"name": "UNKNOWN", "ip": None, "mask": None, "bandwidth": None}

            elif line.startswith("ip address") and current:
                parts = line.split()
                if len(parts) >= 4:
                    current["ip"] = parts[2]
                    current["mask"] = parts[3]

            elif line.startswith("bandwidth") and current:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        current["bandwidth"] = int(parts[1])
                    except ValueError:
                        pass

    if current:
        data["interfaces"].append(current)
    return data


def load_all_configs(config_dir):
    """
    Load all router configs from subfolders (R1, R2, R3... each containing a .dump/.txt).
    """
    routers = []
    for folder in os.listdir(config_dir):
        folder_path = os.path.join(config_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        picked = None
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(".dump") or fname.lower().endswith(".txt"):
                picked = os.path.join(folder_path, fname)
                break
        if picked:
            routers.append(parse_config(picked))
    return routers


# ==============================
# STEP 2: Router↔Router link detection
# ==============================
def find_router_links(routers):
    """
    Match subnets between router interfaces. If two interfaces share a subnet, they are linked.
    Returns list of (routerA, routerB, bandwidth).
    """
    links = []
    # Build helper list: (routerName, ifaceDict, network)
    all_ifaces = []
    for r in routers:
        for iface in r["interfaces"]:
            if iface["ip"] and iface["mask"]:
                net = ipaddress.IPv4Network(f'{iface["ip"]}/{iface["mask"]}', strict=False)
                all_ifaces.append((r["hostname"], iface, net))

    # Find matches
    for i in range(len(all_ifaces)):
        ra, ia, na = all_ifaces[i]
        for j in range(i + 1, len(all_ifaces)):
            rb, ib, nb = all_ifaces[j]
            if na == nb:  # same subnet → link
                bw = ia["bandwidth"] or ib["bandwidth"] or DEFAULT_ROUTER_IF_BW
                links.append((ra, rb, bw))
    return links


# ==============================
# STEP 3: Infer Access LANs → add Switches & Endpoints
# ==============================
def infer_access_lans(routers):
    """
    LAN subnets are those present on exactly ONE router interface (i.e., not shared with other routers).
    For each such subnet, create one Access Switch (SW_<router>_<iface>) and a few endpoints (PCs).
    Returns:
      switches: list of dicts {name, router, lan_net}
      endpoints: list of endpoint names
      access_links: list of (router, switch, bw) and (switch, endpoint, bw)
    """
    # Map: network -> list of (routerName, ifaceName, ifaceBW)
    net_map = {}
    for r in routers:
        for iface in r["interfaces"]:
            if iface["ip"] and iface["mask"]:
                net = ipaddress.IPv4Network(f'{iface["ip"]}/{iface["mask"]}', strict=False)
                net_map.setdefault(str(net), []).append(
                    (r["hostname"], iface["name"], iface["bandwidth"] or DEFAULT_ROUTER_IF_BW)
                )

    switches = []
    endpoints = []
    access_links = []

    for net_str, attaches in net_map.items():
        # LAN if only one router has this subnet
        if len(attaches) == 1:
            rname, iname, rbw = attaches[0]
            sw_name = f"SW_{rname}_{iname.replace('/', '_')}"
            switches.append({"name": sw_name, "router": rname, "lan_net": net_str})

            # Router ↔ Switch link (use router iface bandwidth or default)
            access_links.append((rname, sw_name, rbw))

            # Add endpoints under switch
            for idx in range(1, ENDPOINTS_PER_LAN + 1):
                ep_name = f"PC_{sw_name}_{idx}"
                endpoints.append(ep_name)
                access_links.append((sw_name, ep_name, DEFAULT_ENDPOINT_LINK_BW))

    return switches, endpoints, access_links


# ==============================
# STEP 4: Layering (Core / Dist / Access / Endpoints)
# ==============================
def auto_assign_layers(router_links, switches, endpoints):
    """
    Routers: decide Core / Distribution automatically from router links.
    Switches: Access layer.
    Endpoints: Endpoint layer (below Access).
    """
    layer = {}

    if router_links:
        bws = [bw for _, _, bw in router_links]
        max_bw, min_bw = max(bws), min(bws)

        # Special case: if all router link BWs equal, pick first router as Core
        if max_bw == min_bw:
            core = router_links[0][0]
            layer[core] = 0
            for a, b, _ in router_links:
                if a != core:
                    layer.setdefault(a, 1)
                if b != core:
                    layer.setdefault(b, 1)
        else:
            # Core: routers on highest-BW links
            for a, b, bw in router_links:
                if bw == max_bw:
                    layer[a] = 0
                    layer[b] = 0
            # Dist: routers directly connected to Core
            for a, b, _ in router_links:
                if a not in layer and b in layer:
                    layer[a] = 1
                if b not in layer and a in layer:
                    layer[b] = 1
            # Any stragglers → Access (unlikely for routers)
            for a, b, _ in router_links:
                layer.setdefault(a, 2)
                layer.setdefault(b, 2)

    # Access switches → layer 2
    for sw in switches:
        layer[sw["name"]] = 2

    # Endpoints → layer 3
    for ep in endpoints:
        layer[ep] = 3

    return layer


# ==============================
# STEP 5: Load assignment (apps/random/fixed) + overload check
# ==============================
def compute_load(bw, app_choice=None):
    """
    Decide the load value (kbps) per LOAD_MODE.
    """
    if LOAD_MODE == "fixed":
        return FIXED_LOAD
    if LOAD_MODE == "random":
        return random.randint(TRAFFIC_MIN, TRAFFIC_MAX)
    # Application-aware
    if app_choice is None:
        app_choice = random.choice(list(APP_PROFILES.keys()))
    profile = APP_PROFILES[app_choice]
    return profile["peak"] if USE_PEAK else profile["avg"]


def annotate_links_with_load(links, is_access=False):
    """
    For a list of links [(A, B, bw)], compute loads and status.
    If is_access=True, we’ll pick lighter app types more often (just for realism).
    Returns list of (A, B, bw, load, overloaded, app)
    """
    annotated = []
    for a, b, bw in links:
        app = None
        if LOAD_MODE == "apps":
            # Bias app selection by link type (purely cosmetic, you can remove this)
            if is_access:
                # access links more likely to be Web/VoIP
                app = random.choices(
                    population=list(APP_PROFILES.keys()),
                    weights=[4, 2, 4, 2, 1],  # Web, Video, VoIP, File, Backup
                    k=1
                )[0]
            else:
                # router links see heavier mix (Video/File/Backup)
                app = random.choices(
                    population=list(APP_PROFILES.keys()),
                    weights=[2, 4, 1, 4, 3],
                    k=1
                )[0]
        load = compute_load(bw, app_choice=app)
        overloaded = (bw > 0) and (load > bw)
        annotated.append((a, b, bw, load, overloaded, app))
    return annotated


# ==============================
# STEP 6: Build graph, draw, save PNG
# ==============================
def build_and_draw(router_links_annot, access_links_annot, layer_map):
    """
    Combine router and access links; draw hierarchical topology with labels & legend.
    """
    G = nx.Graph()

    # Add nodes (layer attribute for layout)
    for node, layer in layer_map.items():
        ntype = ("Endpoint" if layer == 3 else "Switch" if layer == 2 else "Router")
        G.add_node(node, layer=layer, ntype=ntype)

    # Add edges
    for (a, b, bw, load, overloaded, app) in router_links_annot + access_links_annot:
        G.add_edge(a, b, bandwidth=bw, load=load, overloaded=overloaded, app=app)

    # Position by layers (x = index in the layer, y = -layer)
    pos = {}
    layers = {}
    for n, d in G.nodes(data=True):
        layers.setdefault(d["layer"], []).append(n)
    for ly in sorted(layers.keys()):
        row = layers[ly]
        for i, n in enumerate(row):
            pos[n] = (i, -ly)

    # Node style
    node_sizes = []
    node_colors = []
    for n, d in G.nodes(data=True):
        if d["ntype"] == "Router":
            node_sizes.append(1600)
            node_colors.append("#8ecae6")
        elif d["ntype"] == "Switch":
            node_sizes.append(1300)
            node_colors.append("#bde0fe")
        else:
            node_sizes.append(900)
            node_colors.append("#d9f99d")

    plt.figure(figsize=(10, 7))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=node_sizes,
        node_color=node_colors,
        font_size=9,
        edge_color=["red" if G[u][v]["overloaded"] else "black" for u, v in G.edges()]
    )

    # Edge labels: "BW / Load (App)"
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        app_str = f" ({d['app']})" if d['app'] else ""
        edge_labels[(u, v)] = f"{d['bandwidth']} / {d['load']}{app_str}"
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Legend (simple text)
    plt.title("Hierarchical Network Topology (Core → Dist → Access → Endpoints)")
    legend_text = "Edge label: Bandwidth kbps / Load kbps (App)\nRed = Overloaded, Black = OK"
    plt.gcf().text(0.01, 0.01, legend_text, fontsize=8, va="bottom")

    if SAVE_PNG:
        plt.tight_layout()
        plt.savefig(OUTPUT_PNG, dpi=150)
        print(f"\nSaved diagram → {OUTPUT_PNG}")

    plt.show()


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # 1) Routers from config
    routers = load_all_configs(CONFIG_DIR)
    print("\nParsed routers:", len(routers))
    for r in routers:
        print(r)

    # 2) Router links
    rtr_links = find_router_links(routers)
    print("\nRouter↔Router Links:", rtr_links if rtr_links else "None")

    # 3) Infer Access: switches + endpoints + links
    switches, endpoints, access_links = infer_access_lans(routers)
    if switches:
        print("\nInferred Access Switches:")
        for sw in switches:
            print(f"  {sw['name']} (LAN {sw['lan_net']}) under {sw['router']}")
    else:
        print("\nNo Access LANs inferred (no single-router subnets found).")

    if endpoints:
        print("\nEndpoints:", endpoints)

    # 4) Layers
    layer_map = auto_assign_layers(rtr_links, switches, endpoints)
    print("\nLayer Map:", layer_map)

    # 5) Loads + overload checks
    rtr_links_annot = annotate_links_with_load(rtr_links, is_access=False)
    acc_links_annot = annotate_links_with_load(access_links, is_access=True)

    print("\nTraffic Load Report (Application-Aware)" if LOAD_MODE == "apps" else "\nTraffic Load Report")
    for (a, b, bw, load, ov, app) in rtr_links_annot + acc_links_annot:
        status = "OVERLOADED" if ov else "OK"
        if app:
            print(f"{a} ↔ {b} [{app}]: BW={bw} kbps, Load={load} kbps → {status}")
        else:
            print(f"{a} ↔ {b}: BW={bw} kbps, Load={load} kbps → {status}")

    # 6) Draw & save
    build_and_draw(rtr_links_annot, acc_links_annot, layer_map)
