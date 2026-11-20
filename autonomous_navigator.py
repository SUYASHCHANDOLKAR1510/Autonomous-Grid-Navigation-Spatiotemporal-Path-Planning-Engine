"""Autonomous Grid Navigation & Dynamic Path Planning Controller.

Connects to a local MQTT broker, loads the 15x15 grid environment, receives
dynamic target coordinates, and publishes differential-drive wheel velocity
commands on every 100 Hz sensor frame.

Architecture:
    - Real-time A* path planning over (row, col, heading, collected_targets,
      predicted_obstacle_states) with dynamic replanning and evasion.
    - Closed-loop dead-reckoning using wheel encoders and bias-calibrated yaw
      gyroscope, corrected against known map geometry via 4 axis-aligned ToF
      rangefinders.
    - Smooth arc-cornering and trapezoidal velocity profiling.

Usage:
    Ensure Mosquitto MQTT broker is running (`sudo systemctl enable --now mosquitto`),
    launch the physics simulator in terminal 1, and run this script in terminal 2:

        python3 autonomous_navigator.py
"""
import json
import math

import paho.mqtt.client as mqtt

import grid_map as grid_env
from grid_map import Maze


MQTT_HOST = "localhost"
MQTT_PORT = 1883
TOPIC_SENSORS = "pacbot/sensors"
TOPIC_PELLETS = "pacbot/pellets"
TOPIC_GHOST = "pacbot/ghost"
TOPIC_GHOST2 = "pacbot/ghost2"
TOPIC_RESULT = "pacbot/result"
TOPIC_WHEEL_VEL = "pacbot/wheel_vel"

WHEEL_RADIUS = 0.017  # m
HALF_TRACK = 0.039    # m, wheel centre to centreline
WHEEL_LIMIT = 30.0    # rad/s, actuator clamp limit
TICKS_PER_REV = 360   # encoder counts per wheel revolution
TICK_RAD = 2.0 * math.pi / TICKS_PER_REV   # radians of wheel turn per tick


def _mqtt_client():
    # paho-mqtt >= 2.0 requires picking a callback API version explicitly.
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        return mqtt.Client()


def wheels(forward, turn):
    """Differential drive mixer: +turn steers LEFT (counter-clockwise)."""
    return forward - turn, forward + turn


class Controller:
    def __init__(self, maze):
        self.maze = maze
        self.pellets = []       # target cells still on the grid; filled from MQTT
        self.have_pellets = False
        self.elapsed = 0.0
        self.ghost = None       # latest dynamic obstacle 1 state
        self.ghost2 = None      # latest dynamic obstacle 2 state
        self.prev_ticks = None  # (left, right) counts from the previous frame

    def on_pellets(self, msg):
        """Called whenever the remaining-target list arrives or changes."""
        self.pellets = [tuple(c) for c in msg.get("pellets", [])]
        self.have_pellets = True
        print(f"targets remaining: {self.pellets}")

    def decide(self, s):
        """One sensor frame in, one wheel command out.

        `s` is the decoded sensor payload. Return (left, right) in rad/s.
        """
        self.elapsed += s["dt"]

        left_dist = s["fl"]          # m to the wall on the left
        right_dist = s["fr"]         # m to the wall on the right
        front_dist = min(s["sl"], s["sr"])   # m to the wall ahead
        yaw_rate = s["gyro"][2]      # rad/s about z
        tl, tr = s["ticks_l"], s["ticks_r"]
        if self.prev_ticks is None or s["dt"] <= 0:
            speed = 0.0
        else:
            dl, dr = tl - self.prev_ticks[0], tr - self.prev_ticks[1]
            speed = (WHEEL_RADIUS * ((dl + dr) / 2.0) * TICK_RAD) / s["dt"]
        self.prev_ticks = (tl, tr)

        # Wait until target coordinates arrive
        if not self.have_pellets:
            return 0.0, 0.0

        st = self.__dict__.setdefault("st", {})
        P, AB = 0.22, 0.033
        VMAX = 0.55
        VARC = 0.28
        CU, T90, UT = 12, 26, 46
        TS = 0.05
        HOR, FHOR, ACCEPT, TRIG = 90.0, 10.0, 6.0, 5.0
        ENC_RUN = 0.90
        ENC_ARC = 0.90
        GAP = 0.05
        EXITS = ((7, 14, 0), (0, 7, 1))
        U = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
        STEP = ((0, 1), (-1, 0), (0, -1), (1, 0))
        SENS = (("sl", -0.005, 0.034, 0.0), ("sr", -0.005, -0.034, 0.0),
                ("fl", 0.036, 0.005, math.pi / 2), ("fr", 0.036, -0.005, -math.pi / 2))
        wrap = lambda a: math.atan2(math.sin(a), math.cos(a))
        clamp = lambda x, lo, hi: max(lo, min(hi, x))
        near_cell = lambda x, y: (clamp(14 - int(round(y / P)), 0, 14), clamp(int(round(x / P)), 0, 14))

        # ---- one-time setup: wall lookup table from the grid environment ----
        if not st:
            dirs = (Maze.EAST, Maze.NORTH, Maze.WEST, Maze.SOUTH)
            wall = {}
            for r in range(15):
                for c in range(15):
                    for d in range(4):
                        nr, nc = r + STEP[d][0], c + STEP[d][1]
                        if 0 <= nr < 15 and 0 <= nc < 15:
                            wall[r, c, d] = not self.maze.can_move((r, c), dirs[d])
                        else:
                            wall[r, c, d] = (r, c, d) not in ((14, 0, 2),) + EXITS
            st.update(wall=wall, calib=0.0, bsum=0.0, bn=0, bias=0.0, pose=None, way=None,
                      dir=0, acts=None, idx=0, started=-1, prev=None, need=True,
                      cell=(14, 0), hd=0, first=True, t=0.0,
                      wait=0.0, settle=0, edge={}, v=0.0,
                      route=[], ri=0, cool=0.0, brake=0.0, rp=0.0, bfs={},
                      align=False, at=0.0, fleeing=False, pm=0,
                      gl=[None, None], glog=[[], []], gmc={},
                      glast=[None, None], rdirs=[], lastchk=-9.0, npel=None, repl=False)
            print("walls built; open sides of start cell (14,0):",
                  [d for d in range(4) if not wall[14, 0, d]])
        wall = st["wall"]
        st["t"] += s["dt"]
        st["simt"] = s["t"]

        # ---- dynamic obstacles: cells and empirical speed (cells/s) ----
        ghs = [g for g in (self.ghost, self.ghost2) if g is not None]
        gcells = tuple(near_cell(g["x"], g["y"]) for g in ghs)
        for gi, g in enumerate((self.ghost, self.ghost2)):
            if g is None:
                continue
            gc_ = near_cell(g["x"], g["y"])
            if gc_ != st["gl"][gi]:
                if st["gl"][gi] is not None:
                    st["glog"][gi].append(st["t"])
                    st["glast"][gi] = st["t"]
                st["gl"][gi] = gc_
            st["glog"][gi] = [x for x in st["glog"][gi] if st["t"] - x < 6.0]
        win = clamp(st["t"], 2.0, 6.0)
        gvs = [clamp(1.25 * len(lg) / win, 0.9, 3.0) for lg in st["glog"]]
        gv = max(gvs)

        # ---- status log every 2 s: robot cell and obstacle cells ----
        if st["pose"] is not None and int(st["t"] / 2.0) != st.setdefault("lp", -1):
            st["lp"] = int(st["t"] / 2.0)
            print(f"[{st['t']:6.1f}s] status: robot {near_cell(st['pose'][0], st['pose'][1])} "
                  f"obstacles {gcells} gv {gv:.1f}")

        def gbfs(src):
            """graph distance (in cells) from src to every cell, through open sides"""
            cache = st["bfs"]
            if src in cache:
                return cache[src]
            dm = {src: 0}
            q = [src]
            for cur in q:
                for dd in range(4):
                    if wall[cur[0], cur[1], dd]:
                        continue
                    nb = (cur[0] + STEP[dd][0], cur[1] + STEP[dd][1])
                    if 0 <= nb[0] < 15 and 0 <= nb[1] < 15 and nb not in dm:
                        dm[nb] = dm[cur] + 1
                        q.append(nb)
            cache[src] = dm
            return dm

        def gmin(cells):
            """graph distance from the nearest dynamic obstacle to every cell"""
            key = tuple(sorted(cells))
            cache = st["gmc"]
            if key in cache:
                return cache[key]
            dm = {}
            for cc in cells:
                for k_, v_ in gbfs(cc).items():
                    if v_ < dm.get(k_, 99):
                        dm[k_] = v_
            if len(cache) > 500:
                cache.clear()
            cache[key] = dm
            return dm

        GPER = 4.0 / 3.0

        def gstate():
            """each obstacle as (cell, committed next cell or None, seconds until transition)"""
            out = []
            ph = st["simt"] % GPER
            for g in (self.ghost, self.ghost2):
                if g is None:
                    continue
                X = near_cell(g["x"], g["y"])
                nxt = None
                if ph < GPER / 2:
                    dx, dy = g["x"] - X[1] * P, g["y"] - (14 - X[0]) * P
                    if abs(dx) + abs(dy) > 0.01:
                        d = (0 if dx > 0 else 2) if abs(dx) > abs(dy) else (1 if dy > 0 else 3)
                        if not wall[X[0], X[1], d]:
                            nxt = (X[0] + STEP[d][0], X[1] + STEP[d][1])
                    dtn = GPER / 2 - ph
                else:
                    dtn = 1.5 * GPER - ph
                out.append((X, nxt, dtn))
            return out

        def timeline(c0, h0, path, t0=0.1, moving=False):
            """cells the robot visits along path and the time it reaches each centre"""
            cells, times, t_, h_, c_ = [c0], [0.0], t0, h0, c0
            room = 9.0 if moving else 0.0
            still = not moving
            for d in path:
                if d == h_:
                    t_ += 0.6 + (0.3 if still else 0.0)
                    room += P
                elif (d - h_) % 4 == 2:
                    t_ += 2.3 + 0.9
                    room = P
                elif room >= P / 2 + GAP:
                    t_ += 0.8
                    room = P / 2
                else:
                    t_ += 0.9 + 0.9
                    room = P
                still = False
                h_ = d
                nc_ = (c_[0] + STEP[d][0], c_[1] + STEP[d][1])
                if not (0 <= nc_[0] < 15 and 0 <= nc_[1] < 15):
                    break
                cells.append(nc_)
                times.append(t_)
                c_ = nc_
            return cells, times

        def chase(cells, times, gs, horizon, hold, margin=1):
            """first time a pursuing obstacle meets the robot along the path"""
            if not gs:
                return 1e9
            end = horizon if hold else min(horizon, times[-1] + 1.0)
            n_ = len(cells)
            ptr = [0, 0]

            def rcell(t_, w):
                i_ = ptr[w] if 0.5 * (times[ptr[w]] + times[min(ptr[w] + 1, n_ - 1)]) <= t_ or ptr[w] == 0 else 0
                while i_ + 1 < n_ and 0.5 * (times[i_] + times[i_ + 1]) <= t_:
                    i_ += 1
                ptr[w] = i_
                return i_

            gp = [g[0] for g in gs]
            gc = [g[1] for g in gs]
            gn = [g[2] for g in gs]
            t_ = 0.0
            while t_ <= end:
                ri_ = rcell(t_, 0)
                R = cells[ri_]
                Rn = cells[ri_ + 1] if ri_ + 1 < n_ else R
                dR = gbfs(R)
                for k in range(len(gs)):
                    while gn[k] <= t_:
                        if gc[k] is not None:
                            gp[k], gc[k] = gc[k], None
                        else:
                            dT = gbfs(cells[rcell(max(0.0, gn[k] - GPER / 2), 1)])
                            r0, c0 = gp[k]
                            bc, bk = gp[k], dT.get(gp[k], 99)
                            for d in range(4):
                                if wall[r0, c0, d]:
                                    continue
                                nb = (r0 + STEP[d][0], c0 + STEP[d][1])
                                if 0 <= nb[0] < 15 and 0 <= nb[1] < 15 and dT.get(nb, 99) < bk:
                                    bc, bk = nb, dT.get(nb, 99)
                            gp[k] = bc
                        gn[k] += GPER
                    if dR.get(gp[k], 99) <= margin or gp[k] == Rn:
                        return t_
                t_ += 0.05
            return 1e9

        # ---- encoders and gyro ----
        ticks = (s["ticks_l"], s["ticks_r"])
        if st["prev"] is None:
            st["prev"] = ticks
        d_left = (ticks[0] - st["prev"][0]) * WHEEL_RADIUS * TICK_RAD
        d_right = (ticks[1] - st["prev"][1]) * WHEEL_RADIUS * TICK_RAD
        st["prev"] = ticks
        if st["calib"] < 0.5:
            st["calib"] += s["dt"]
            st["bsum"] += s["gyro"][2]
            st["bn"] += 1
            st["bias"] = st["bsum"] / st["bn"]
            return 0.0, 0.0
        rate = s["gyro"][2] - st["bias"]

        # ---- plan (initially, on leftover targets, and upon obstacle proximity) ----
        if st["need"] and not st["align"]:
            if st["rp"] > 0:
                st["rp"] -= s["dt"]
                return 0.0, 0.0
            if st["first"] and len(ghs) < 2 and st["t"] < 3.0:
                return 0.0, 0.0
            gs = gstate()
            pel = {tuple(p): i for i, p in enumerate(self.pellets)}
            full = (1 << len(pel)) - 1
            cell, hd = st["cell"], st["hd"]
            n0 = (cell[0], cell[1], hd, (1 << pel[cell]) if cell in pel else 0)

            def search(forbid):
                dist = {n0: 0}
                prv = {}
                buckets = {0: [n0]}
                cost = 0
                found = False
                while buckets and not found:
                    for node in buckets.pop(cost, []):
                        if dist.get(node) != cost:
                            continue
                        if node == "OUT":
                            found = True
                            break
                        r, c, h, m = node
                        for d in range(4):
                            if wall[r, c, d]:
                                continue
                            tc = CU if d == h else (UT if (d - h) % 4 == 2 else T90)
                            if (r, c, d) in EXITS:
                                if m != full:
                                    continue
                                nxt = "OUT"
                            else:
                                nr, nc = r + STEP[d][0], c + STEP[d][1]
                                if (nr, nc) == (14, 0) and d == 2:
                                    continue
                                if not (0 <= nr < 15 and 0 <= nc < 15) or (nr, nc) in forbid:
                                    continue
                                nxt = (nr, nc, d, m | (1 << pel[(nr, nc)]) if (nr, nc) in pel else m)
                            if cost + tc < dist.get(nxt, 10 ** 9):
                                dist[nxt] = cost + tc
                                prv[nxt] = (node, d)
                                buckets.setdefault(cost + tc, []).append(nxt)
                    cost += 1
                path, node = [], "OUT"
                while found and node != n0:
                    node, d = prv[node]
                    path.append(d)
                path.reverse()
                return found, path

            def flee_plan():
                """reachable cell where the robot stays clear of pursuing obstacles longest"""
                f0 = (cell[0], cell[1], hd)
                dist = {f0: 0}
                prv = {}
                buckets = {0: [f0]}
                cost = 0
                while buckets and cost <= 100:
                    for node in buckets.pop(cost, []):
                        if dist.get(node) != cost:
                            continue
                        r, c, h = node
                        for d in range(4):
                            if wall[r, c, d]:
                                continue
                            nr, nc = r + STEP[d][0], c + STEP[d][1]
                            if (nr, nc) == (14, 0) and d == 2:
                                continue
                            if not (0 <= nr < 15 and 0 <= nc < 15):
                                continue
                            tc = CU if d == h else (UT if (d - h) % 4 == 2 else T90)
                            nxt = (nr, nc, d)
                            if cost + tc < dist.get(nxt, 10 ** 9):
                                dist[nxt] = cost + tc
                                prv[nxt] = (node, d)
                                buckets.setdefault(cost + tc, []).append(nxt)
                    cost += 1
                dg = gmin(tuple(g[0] for g in gs))
                best, bkey, seen = None, None, set()
                for node in sorted(dist, key=dist.get):
                    if node == f0 or node[:2] in seen or dg.get(node[:2], 99) < 3:
                        continue
                    seen.add(node[:2])
                    path, nd = [], node
                    while nd != f0:
                        nd, d = prv[nd]
                        path.append(d)
                    path.reverse()
                    cl, tl = timeline(cell, hd, path)
                    ct = chase(cl, tl, gs, FHOR, True)
                    key = (min(ct, FHOR), dg.get(node[:2], 99), -dist[node])
                    if bkey is None or key > bkey:
                        best, bkey = (ct, tl[-1], path, node[:2]), key
                return best

            def astar():
                """time-optimal route over (robot, targets, dynamic obstacles)"""
                import heapq
                bits = [1 << pel[c_] for c_ in sorted(pel, key=pel.get)]
                cost_s, cost_a, cost_u = 0.6, 0.8, 3.2
                hcost = {}
                heap = []
                for (er, ec, ed) in EXITS:
                    for hh in range(4):
                        tcs = cost_s if hh == ed else (cost_u if (ed - hh) % 4 == 2 else cost_a)
                        k0 = (er, ec, hh, full)
                        if tcs < hcost.get(k0, 1e9):
                            hcost[k0] = tcs
                            heapq.heappush(heap, (tcs, k0))
                while heap:
                    cst, (r, c, h, m) = heapq.heappop(heap)
                    if cst > hcost.get((r, c, h, m), 1e9):
                        continue
                    pr, pc = r - STEP[h][0], c - STEP[h][1]
                    if not (0 <= pr < 15 and 0 <= pc < 15) or wall[pr, pc, h]:
                        continue
                    pms = [m]
                    if (r, c) in pel and m & (1 << pel[(r, c)]):
                        pms.append(m ^ (1 << pel[(r, c)]))
                    for pm_ in pms:
                        if (pr, pc) in pel and not pm_ & (1 << pel[(pr, pc)]):
                            continue
                        for ph_ in range(4):
                            tcs = cost_s if ph_ == h else (cost_u if (h - ph_) % 4 == 2 else cost_a)
                            k1 = (pr, pc, ph_, pm_)
                            if cst + tcs < hcost.get(k1, 1e9):
                                hcost[k1] = cst + tcs
                                heapq.heappush(heap, (cst + tcs, k1))

                def gstep(gx, gy, ta, tb, rfn):
                    """advance obstacles over (ta, tb]; None if one comes within a cell of robot"""
                    gx, gy = list(gx), list(gy)
                    k = math.floor(ta / (GPER / 2)) + 1
                    while True:
                        e = k * (GPER / 2)
                        if e > tb:
                            break
                        rc_ = rfn(e)
                        dR = gbfs(rc_)
                        centre_ev = k % 2 == 0
                        for i_ in range(len(gx)):
                            if centre_ev:
                                if gy[i_] is None:
                                    X = gx[i_]
                                    bc, bk = X, dR.get(X, 99)
                                    for d in range(4):
                                        if wall[X[0], X[1], d]:
                                            continue
                                        nb = (X[0] + STEP[d][0], X[1] + STEP[d][1])
                                        if 0 <= nb[0] < 15 and 0 <= nb[1] < 15 and dR.get(nb, 99) < bk:
                                            bc, bk = nb, dR.get(nb, 99)
                                    gy[i_] = bc if bc != X else None
                            elif gy[i_] is not None:
                                gx[i_], gy[i_] = gy[i_], None
                            if dR.get(gx[i_], 99) <= 1:
                                return None
                        k += 1
                    dR = gbfs(rfn(tb))
                    for i_ in range(len(gx)):
                        if dR.get(gx[i_], 99) <= 1:
                            return None
                    return tuple(gx), tuple(gy)

                T0 = st["simt"] + 0.15 + (0.5 if st["first"] else 0.0)
                gx0 = tuple(g[0] for g in gs)
                gy0 = tuple(g[1] for g in gs)
                m0 = (1 << pel[cell]) if cell in pel else 0
                start = (cell[0], cell[1], hd, m0, 0)
                nodes = [(start, gx0, gy0, T0, -1, -1)]
                seen = {}
                h0 = hcost.get((cell[0], cell[1], hd, m0), 1e9)
                if h0 >= 1e9:
                    return None
                openl = [(T0 + 1.2 * h0, 0)]
                expanded = 0
                while openl and expanded < 6000:
                    _, ni = heapq.heappop(openl)
                    (r, c, h, m, af), gx, gy, t, par, dd = nodes[ni]
                    if t - T0 > 90.0:
                        continue
                    expanded += 1
                    for d in range(4):
                        if wall[r, c, d]:
                            continue
                        still = af == 0
                        if d == h:
                            dur, cross, naf = 0.6 + (0.3 if still else 0.0), 0.5, 2
                        elif (d - h) % 4 == 2:
                            dur, cross, naf = 3.2, (3.2 - 0.35) / 3.2, 3
                        elif af == 2:
                            dur, cross, naf = 0.8, 0.5, 1
                        else:
                            dur, cross, naf = 1.8, (1.8 - 0.35) / 1.8, 3
                        tb = t + dur
                        if (r, c, d) in EXITS:
                            if m != full:
                                continue
                            ok = gstep(gx, gy, t, tb, lambda e, a=(r, c): a)
                            if ok is None:
                                continue
                            path, j = [d], ni
                            while nodes[j][4] >= 0:
                                path.append(nodes[j][5])
                                j = nodes[j][4]
                            path.reverse()
                            return path, tb - T0, expanded
                        nr, nc = r + STEP[d][0], c + STEP[d][1]
                        if (nr, nc) == (14, 0) and d == 2:
                            continue
                        if not (0 <= nr < 15 and 0 <= nc < 15):
                            continue
                        tx = t + cross * dur
                        res = gstep(gx, gy, t, tb, lambda e, a=(r, c), b=(nr, nc), tx=tx: a if e < tx else b)
                        if res is None:
                            continue
                        nm = m | (1 << pel[(nr, nc)]) if (nr, nc) in pel else m
                        hv = hcost.get((nr, nc, d, nm), 1e9)
                        if hv >= 1e9:
                            continue
                        key = (nr, nc, d, nm, naf, res[0], res[1], int((tb % GPER) / 0.1))
                        if seen.get(key, 1e9) <= tb:
                            continue
                        seen[key] = tb
                        nodes.append(((nr, nc, d, nm, 2 if naf == 3 else naf), res[0], res[1], tb, ni, d))
                        heapq.heappush(openl, (tb + 1.2 * hv, len(nodes) - 1))
                return None

            cands, forbid = [], set()
            ap = astar() if gs else None
            if ap is not None:
                cands.append((1e9, ap[1], ap[0], None))
            for _ in range(0 if ap is not None else 6):
                ok, path = search(forbid)
                if not ok:
                    break
                cl, tl = timeline(cell, hd, path)
                ct = chase(cl, tl, gs, HOR, False)
                cands.append((ct, tl[-1], path, None))
                if ct >= HOR:
                    break
                j = max(i for i in range(len(tl)) if tl[i] <= ct)
                bad = cl[j + 1] if j + 1 < len(cl) else cl[j]
                if bad == cell or bad in forbid:
                    break
                forbid.add(bad)
            safe = [c_ for c_ in cands if c_[0] >= HOR]
            if safe:
                best = min(safe, key=lambda c_: c_[1])
            else:
                best = max(cands, key=lambda c_: (c_[0], -c_[1])) if cands else None
                if gs and (best is None or best[0] < ACCEPT):
                    fb = flee_plan()
                    if fb is not None and (best is None or fb[0] > best[0] + 0.5):
                        best = fb
            if best is None:
                st["rp"] = 0.2
                return 0.0, 0.0
            ct, _, path, flee = best
            if flee:
                print(f"[{st['t']:6.1f}s] evading to {flee}")

            route, _ = timeline(cell, hd, path)
            st["route"], st["rdirs"], st["ri"] = route, path, 0

            acts, cur = [], hd
            for i, d in enumerate(path):
                if d != cur:
                    acts.append(["turn", d])
                    cur = d
                step_len = P
                if i == len(path) - 1 and flee is None:
                    step_len += 0.15
                if acts and acts[-1][0] == "run":
                    acts[-1][1] += step_len
                else:
                    acts.append(["run", step_len])
            if st["first"]:
                if acts and acts[0][0] == "run":
                    acts[0][1] += 0.15 + AB
                else:
                    acts.insert(0, ["run", 0.15 + AB])
                st["first"] = False
            out, pos, h = [], st["way"] or (-0.15 - AB, 0.0), hd
            for i, a in enumerate(acts):
                if a[0] == "run":
                    end = (pos[0] + a[1] * U[h][0], pos[1] + a[1] * U[h][1])
                    out.append(["run", a[1], end, False])
                    pos = end
                elif ((a[1] - h) % 4 != 2 and out and out[-1][0] == "run" and out[-1][1] >= P / 2 + GAP
                      and i + 1 < len(acts) and acts[i + 1][0] == "run"):
                    d = a[1]
                    C = pos
                    E = (C[0] - P / 2 * U[h][0], C[1] - P / 2 * U[h][1])
                    X = (C[0] + P / 2 * U[d][0], C[1] + P / 2 * U[d][1])
                    out[-1][1] -= P / 2
                    out[-1][2], out[-1][3] = E, True
                    out.append(["arc", d, (E[0] + X[0] - C[0], E[1] + X[1] - C[1]),
                                1 if (d - h) % 4 == 1 else -1, X])
                    acts[i + 1] = ["run", acts[i + 1][1] - P / 2]
                    pos, h = X, d
                else:
                    out.append(["turn", a[1]])
                    h = a[1]
            acts = out
            st.update(acts=acts, idx=0, started=-1, need=False, settle=0, edge={}, v=0.0,
                      fleeing=flee is not None)
            print(f"[{st['t']:6.1f}s] targets {self.pellets} {'evade' if flee else 'route'} "
                  f"from {cell}: {len(path)} moves, first obstacle contact "
                  f"{'none' if ct >= HOR else f'{ct:.1f}s'}, avoided {sorted(forbid)}")

        # ---- pose of the AXLE midpoint: [x, y, heading] ----
        if st["pose"] is None:
            st["pose"] = [-0.15 - AB, 0.0, 0.0]
            st["way"] = (-0.15 - AB, 0.0)
        ax, ay, th = st["pose"]
        acts, idx = st["acts"], st["idx"]
        kind, arg = acts[idx][:2] if idx < len(acts) else ("stop", 0.0)
        if st["align"]:
            kind, arg = "align", 0.0
        ds = 0.0 if kind == "turn" else (ENC_ARC if kind == "arc" else ENC_RUN) * (d_left + d_right) / 2
        dth = rate * s["dt"]
        ax += ds * math.cos(th + dth / 2)
        ay += ds * math.sin(th + dth / 2)
        th = wrap(th + dth)

        # ---- correct the position with the ToF sensors against the known map ----
        if kind not in ("turn", "arc"):
            co, si = math.cos(th), math.sin(th)
            ccx, ccy = ax + AB * co, ay + AB * si
            for name, sx_, sy_, ang in SENS[2:]:
                if s[name] >= 1.999:
                    continue
                ox = ccx + sx_ * co - sy_ * si
                oy = ccy + sx_ * si + sy_ * co
                k = int(round((th + ang) / (math.pi / 2))) % 4
                delta = wrap(th + ang - k * math.pi / 2)
                if abs(delta) > 0.3:
                    continue
                col, row = int(round(ox / P)), 14 - int(round(oy / P))
                a_exp = None
                for _ in range(16):
                    if not (0 <= row < 15 and 0 <= col < 15):
                        break
                    if wall[row, col, k]:
                        cx, cy = col * P, (14 - row) * P
                        if k == 0:
                            a_exp = cx + P / 2 - 0.005 - ox
                        elif k == 1:
                            a_exp = cy + P / 2 - 0.005 - oy
                        elif k == 2:
                            a_exp = ox - (cx - P / 2 + 0.005)
                        else:
                            a_exp = oy - (cy - P / 2 + 0.005)
                        break
                    row += STEP[k][0]
                    col += STEP[k][1]
                if a_exp is None:
                    continue
                e = s[name] * math.cos(delta) - a_exp
                if abs(e) > 0.04:
                    continue
                ax -= 0.05 * e * U[k][0]
                ay -= 0.05 * e * U[k][1]
            if kind == "run":
                u, d0 = U[st["dir"]], st["dir"]
                herr = wrap(th - d0 * math.pi / 2)
                wx, wy = st["way"]
                ref = (clamp(14 - int(round(wy / P)), 0, 14), clamp(int(round(wx / P)), 0, 14))
                off = (wx - ref[1] * P) * u[0] + (wy - (14 - ref[0]) * P) * u[1]
                along = (ax - wx) * u[0] + (ay - wy) * u[1] + off
                r_, c_, n_, wall_at = ref[0], ref[1], 0, None
                while 0 <= r_ < 15 and 0 <= c_ < 15:
                    if wall[r_, c_, d0]:
                        wall_at = n_ * P + P / 2 - 0.005
                        break
                    r_, c_, n_ = r_ + STEP[d0][0], c_ + STEP[d0][1], n_ + 1
                if wall_at is not None and abs(herr) < 0.25:
                    est = [wall_at - s[nm] * math.cos(herr) - ((AB + sx_) * math.cos(herr) - sy_ * math.sin(herr))
                           for nm, sx_, sy_, _ in SENS[:2] if s[nm] < 1.999]
                    if len(est) == 2 and abs(est[0] - est[1]) < 0.01:
                        e = along - (est[0] + est[1]) / 2
                        if abs(e) < 0.1:
                            ax -= 0.1 * e * u[0]
                            ay -= 0.1 * e * u[1]
                            along -= 0.1 * e
                        th_m = d0 * math.pi / 2 + math.atan2(s["sl"] - s["sr"], 0.068)
                        th = wrap(th + 0.01 * wrap(th_m - th))
                if s["fl"] < 0.16 and s["fr"] < 0.16 and abs(herr) < 0.15:
                    lv_ = (-u[1], u[0])
                    lat_m = ((s["fr"] - s["fl"]) / 2) * math.cos(herr) - (AB + 0.036) * math.sin(herr)
                    lat_e = (ax - wx) * lv_[0] + (ay - wy) * lv_[1]
                    ax -= 0.2 * (lat_e - lat_m) * lv_[0]
                    ay -= 0.2 * (lat_e - lat_m) * lv_[1]
                for name, sx_, sy_, ang in SENS[2:]:
                    side = (d0 + (1 if ang > 0 else 3)) % 4
                    near = s[name] < 0.16
                    es = st["edge"].get(name)
                    if es is None:
                        st["edge"][name] = [near, near, along, 0.0]
                        continue
                    if near != es[1]:
                        es[1], es[2], es[3] = near, along, 0.0
                        continue
                    es[3] += abs(ds)
                    if es[1] == es[0] or es[3] < 0.015:
                        continue
                    es[0] = es[1]
                    hit = es[2] + AB + sx_ + (-1 if ang > 0 else 1) * 0.1 * math.sin(herr)
                    best = None
                    for j in range(-1, int(arg / P) + 3):
                        a = (ref[0] + j * STEP[d0][0], ref[1] + j * STEP[d0][1])
                        b = (a[0] + STEP[d0][0], a[1] + STEP[d0][1])
                        if not (0 <= min(a + b) and max(a + b) < 15):
                            continue
                        wa, wb = wall[a[0], a[1], side], wall[b[0], b[1], side]
                        if wa and not wb and not near:
                            edge = (j + 0.5) * P + 0.005
                        elif wb and not wa and near:
                            edge = (j + 0.5) * P - 0.005
                        else:
                            continue
                        if best is None or abs(edge - hit) < abs(best - hit):
                            best = edge
                    if best is not None and abs(best - hit) < 0.1:
                        ax += (best - hit) * u[0]
                        ay += (best - hit) * u[1]
                        along += best - hit
        st["pose"] = [ax, ay, th]

        # ---- braking after an obstacle proximity alarm ----
        if st["brake"] > 0:
            st["brake"] -= s["dt"]
            if st["brake"] <= 0:
                rc0 = near_cell(ax, ay)
                u_ = U[st["dir"]]
                ahead_c = (rc0[1] * P - ax) * u_[0] + ((14 - rc0[0]) * P - ay) * u_[1] > 0.02
                g_ahead = any((g["x"] - ax) * u_[0] + (g["y"] - ay) * u_[1] > 0
                              and math.hypot(g["x"] - ax, g["y"] - ay) < 0.6 for g in ghs)
                back = (rc0[0] - STEP[st["dir"]][0], rc0[1] - STEP[st["dir"]][1])
                if ahead_c and g_ahead and 0 <= back[0] < 15 and 0 <= back[1] < 15                         and not wall[back[0], back[1], st["dir"]]:
                    rc0 = back
                st["cell"], st["hd"] = rc0, st["dir"]
                st["way"] = (rc0[1] * P, (14 - rc0[0]) * P)
                st.update(need=True, settle=0, edge={}, v=0.0, align=True, at=0.0)
                print(f"[{st['t']:6.1f}s] proximity alarm: centring on {rc0}, heading {st['dir']}, obstacles {gcells}, "
                      f"nearest {gmin(gcells).get(rc0, 99) if gcells else None} cells")
            return 0.0, 0.0

        # ---- align: drive (forward or reverse) to the cell centre before replanning / pivoting ----
        if st["align"]:
            u = U[st["dir"]]
            lv = (-u[1], u[0])
            Cx, Cy = st["way"]
            e = (Cx - ax) * u[0] + (Cy - ay) * u[1]
            lat = (ax - Cx) * lv[0] + (ay - Cy) * lv[1]
            herr = wrap(th - st["dir"] * math.pi / 2)
            st["at"] += s["dt"]
            if abs(e) < 0.015 or st["at"] > 1.5:
                st.update(align=False, settle=0, edge={}, v=0.0)
                print(f"[{st['t']:6.1f}s] centred, residual {e * 100:+.1f} cm")
                return 0.0, 0.0
            v = math.copysign(clamp(abs(4.0 * e), 0.06, 0.30), e)
            om = clamp(-5.0 * herr - (25.0 * lat if e > 0 else 0.0), -1.5, 1.5)
            return wheels(v / WHEEL_RADIUS, om * HALF_TRACK / WHEEL_RADIUS)

        if idx >= len(acts):
            if st["fleeing"]:
                rc1 = near_cell(ax, ay)
                st.update(cell=rc1, hd=st["dir"], need=True, fleeing=False, align=True, at=0.0,
                          way=(rc1[1] * P, (14 - rc1[0]) * P))
            return 0.0, 0.0

        if st["npel"] is None:
            st["npel"] = len(self.pellets)
        if len(self.pellets) != st["npel"]:
            st["npel"] = len(self.pellets)
            st["repl"] = True
        if st["repl"] and kind in ("run", "turn"):
            st["repl"] = False
            rc_ = near_cell(ax, ay)
            if gcells and gmin(gcells).get(rc_, 99) <= 10:
                print(f"[{st['t']:6.1f}s] target collected with obstacle {gmin(gcells).get(rc_, 99)} cells away: replanning")
                st["brake"], st["cool"] = 0.05, 1.0
                return 0.0, 0.0

        # ---- obstacle proximity monitor ----
        if st["cool"] > 0:
            st["cool"] -= s["dt"]
        if (ghs and st["route"] and st["cool"] <= 0
                and (kind == "run" or (kind == "turn" and st["started"] != idx))):
            rt = st["route"]
            rc = near_cell(ax, ay)
            for j in range(st["ri"], min(st["ri"] + 3, len(rt))):
                if rt[j] == rc:
                    st["ri"] = j
                    break
            ahead = False
            if st["t"] - st["lastchk"] >= 0.2:
                st["lastchk"] = st["t"]
                cl, tl = timeline(rt[st["ri"]], st["dir"], st["rdirs"][st["ri"]:], 0.0, kind == "run")
                ahead = chase(cl, tl, gstate(), TRIG + 0.5, False, 1) < TRIG
            dgm = gmin(gcells)
            u0 = U[arg] if kind == "turn" else U[st["dir"]]
            close = False
            if dgm.get(rc, 99) <= 2:
                for g in ghs:
                    gx0 = g["x"] - (ax + AB * math.cos(th))
                    gy0 = g["y"] - (ay + AB * math.sin(th))
                    if math.hypot(gx0, gy0) < 0.5 and (gx0 * u0[0] + gy0 * u0[1]) > -0.1:
                        close = True
            if ahead or close:
                st["brake"] = 0.1
                st["cool"] = 1.0
                return 0.0, 0.0

        # ---- start of a new motion primitive ----
        if st["started"] != idx:
            wx, wy = st["way"]
            if idx == len(acts) - 1 and self.pellets and kind == "run" and not st["fleeing"]:
                st["cell"] = (clamp(14 - int(round(wy / P)), 0, 14), clamp(int(round(wx / P)), 0, 14))
                st["hd"] = st["dir"]
                st["need"] = True
                return 0.0, 0.0
            if kind == "turn":
                uu = U[st["dir"]]
                if abs((wx - ax) * uu[0] + (wy - ay) * uu[1]) > 0.03:
                    st.update(align=True, at=0.0)
                    return 0.0, 0.0
            st["wait"] = 0.0
            st["started"] = idx

        # ---- arc cornering ----
        if kind == "arc":
            K, sgn, X = acts[idx][2], acts[idx][3], acts[idx][4]
            if sgn * wrap(th - arg * math.pi / 2) >= -0.03:
                print(f"[{st['t']:6.1f}s] arc {arg} done, off {(ax - X[0]) * 100:+.1f},{(ay - X[1]) * 100:+.1f} cm")
                st["dir"] = arg
                st["way"] = X
                st["idx"] += 1
                st.update(settle=99, edge={}, v=VARC)
                return wheels(VARC / WHEEL_RADIUS, 0.0)
            rad = P / 2
            lat = sgn * (rad - math.hypot(ax - K[0], ay - K[1]))
            herr = wrap(th - (math.atan2(ay - K[1], ax - K[0]) + sgn * math.pi / 2))
            om = sgn * VARC / rad + clamp(-5.0 * herr - 25.0 * lat, -2.5, 2.5)
            return wheels(VARC / WHEEL_RADIUS, om * HALF_TRACK / WHEEL_RADIUS)

        # ---- turn: pivot on the spot ----
        if kind == "turn":
            err = wrap(arg * math.pi / 2 - th)
            if abs(err) < 0.03 and abs(rate) < 0.6:
                print(f"[{st['t']:6.1f}s] turn {arg} done, off centre "
                      f"{(ax - st['way'][0]) * 100:+.1f},{(ay - st['way'][1]) * 100:+.1f} cm")
                st["dir"] = arg
                st["idx"] += 1
                st.update(settle=0, edge={}, v=0.0)
                return 0.0, 0.0
            want = math.copysign(clamp(math.sqrt(2.0 * 14.0 * max(abs(err) - 0.01, 0.0)), 0.6, 6.5), err)
            om = clamp(want + 1.0 * (want - rate), -8.0, 8.0)
            return wheels(0.0, om * HALF_TRACK / WHEEL_RADIUS)

        # ---- run: straight to the target cell centre ----
        u = U[st["dir"]]
        lv = (-u[1], u[0])
        rx, ry = ax - st["way"][0], ay - st["way"][1]
        along = rx * u[0] + ry * u[1]
        lat = rx * lv[0] + ry * lv[1]
        herr = wrap(th - st["dir"] * math.pi / 2)
        if st["settle"] < 2:
            st["settle"] += 1
            return 0.0, 0.0
        remaining = arg - along
        fly = acts[idx][3]
        if fly and remaining <= 0.0:
            print(f"[{st['t']:6.1f}s] run {arg / P:.1f} cells done (into a corner), lat {lat * 100:+.1f} cm")
            st["way"] = acts[idx][2]
            st["idx"] += 1
            st["edge"] = {}
            return wheels(st["v"] / WHEEL_RADIUS, 0.0)
        if not fly and remaining <= 0.004:
            tx, ty = st["way"][0] + arg * u[0], st["way"][1] + arg * u[1]
            tgt = (14 - int(round(ty / P)), int(round(tx / P)))
            if (0 <= min(tgt) and max(tgt) < 15 and abs(tx - round(tx / P) * P) + abs(ty - round(ty / P) * P) < 0.01
                    and -remaining < 0.15 and any(
                        (s[nm] < 0.16) != wall[tgt[0], tgt[1], (st["dir"] + (1 if ang > 0 else 3)) % 4]
                        for nm, _, _, ang in SENS[2:])):
                return wheels(0.05 / WHEEL_RADIUS, clamp(-5.0 * herr - 25.0 * lat, -2.5, 2.5) * HALF_TRACK / WHEEL_RADIUS)
            nx = round((st["way"][0] + arg * u[0]) / P) * P
            ny = round((st["way"][1] + arg * u[1]) / P) * P
            print(f"[{st['t']:6.1f}s] run {arg / P:.1f} cells done, lat {lat * 100:+.1f} cm")
            st["way"] = (nx, ny)
            st["idx"] += 1
            st.update(settle=0, edge={}, v=0.0)
            return 0.0, 0.0
        for g in ghs:
            gx = g["x"] - (ax + AB * math.cos(th))
            gy = g["y"] - (ay + AB * math.sin(th))
            g_al = gx * u[0] + gy * u[1]
            g_lat = gx * lv[0] + gy * lv[1]
            if 0.0 < g_al < 0.40 and abs(g_lat) < 0.14 and g_al < min(s["sl"], s["sr"]) + 0.05:
                st["brake"], st["cool"] = 0.05, 1.0
                return 0.0, 0.0
        om = clamp(-5.0 * herr - 25.0 * lat, -2.5, 2.5)
        v_end = VARC if fly else 0.0
        v = clamp(math.sqrt(v_end ** 2 + 2.0 * 2.0 * max(remaining - 0.003, 0.0)), 0.04, VMAX) * clamp(1.0 - abs(herr) / 0.5, 0.3, 1.0)
        v = st["v"] = min(v, st["v"] + 2.5 * s["dt"])
        return wheels(v / WHEEL_RADIUS, om * HALF_TRACK / WHEEL_RADIUS)


def main():
    maze = Maze.official()
    print(f"grid environment {grid_env.MAZE_ID}: {maze.grid}x{maze.grid}, "
          f"spawn row {maze.spawn_row}, exits {maze.exits}")

    controller = Controller(maze)
    client = _mqtt_client()

    def on_message(cli, userdata, msg):
        payload = json.loads(msg.payload.decode())
        if msg.topic == TOPIC_PELLETS:
            controller.on_pellets(payload)
            return
        if msg.topic == TOPIC_GHOST:
            controller.ghost = payload
            return
        if msg.topic == TOPIC_GHOST2:
            controller.ghost2 = payload
            return
        if msg.topic == TOPIC_RESULT:
            print(f"run over: solved={payload['solved']}, "
                  f"collisions={payload['collisions']}")
            return
        left, right = controller.decide(payload)
        left = max(-WHEEL_LIMIT, min(WHEEL_LIMIT, float(left)))
        right = max(-WHEEL_LIMIT, min(WHEEL_LIMIT, float(right)))
        cli.publish(TOPIC_WHEEL_VEL,
                    json.dumps({"left": left, "right": right}))

    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT)
    client.subscribe(TOPIC_PELLETS)
    client.subscribe(TOPIC_GHOST)
    client.subscribe(TOPIC_GHOST2)
    client.subscribe(TOPIC_RESULT)
    client.subscribe(TOPIC_SENSORS)
    print("connected; waiting for the simulator")
    client.loop_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
