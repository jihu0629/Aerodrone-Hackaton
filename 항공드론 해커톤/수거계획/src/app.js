            (function() {
                'use strict';
                const D = window.PLAN;
                Object.assign(D.control_defaults, {
                    'c-buoy': true, 'c-buoy-area': 0.5, 'c-buoy-fp': 0.5,
                    'c-buoy-min': 1.0, 'c-buoy-typ': 2.0, 'c-buoy-max': 5,
                    'c-dup': 1.5, 'c-exdup': false, 'c-extide': false
                });
                const $ = (s) => document.querySelector(s);
                const esc = (s) => String(s).replace(/[&<>"']/g, c => ({
                    '&': '&amp;',
                    '<': '&lt;',
                    '>': '&gt;',
                    '"': '&quot;',
                    "'": '&#39;'
                }[c]));
                const DAYC = D.day_colors;
                const TEAMC = ["#2a78d6", "#eb6834", "#1baf7a", "#8e44ad", "#eda100", "#e34948"];
                const TEAMN = ["A", "B", "C", "D", "E", "F"];
                const COMPASS = ["북", "북동", "동", "남동", "남", "남서", "서", "북서"];
                const CIRC = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳";
                const NIOSH = 23;

                // ───────── 저장 (이 브라우저에만): 완료 체크, 실측 보정 계수 ─────────
                const LS_KEY = 'shoresweep:' + (D.site || 'site');
                // 완료·만조 표시는 물체별 "마지막 변경 시각" 으로 기록한다.
                // 여러 사람이 동시에 바꿔도 항목별 최신 값이 남아 서로의 변경을 지우지 않는다.
                const EV_KEYS = ['done', 'undone', 'tide', 'untide'];
                let ST = {
                    done: new Set(),
                    tide: new Set(),
                    calib: {},
                    calibAt: {},
                    depotAt: 0,
                    ev: { done: {}, undone: {}, tide: {}, untide: {} }
                };
                function deriveSets() {
                    ST.done = new Set(Object.keys(ST.ev.done).filter(id => (ST.ev.done[id] || 0) > (ST.ev.undone[id] || 0)));
                    ST.tide = new Set(Object.keys(ST.ev.tide).filter(id => (ST.ev.tide[id] || 0) > (ST.ev.untide[id] || 0)));
                }
                function markEv(kind, ids, ts) {
                    ts = ts || Date.now();
                    for (const id of ids)
                        ST.ev[kind][id] = Math.max(ST.ev[kind][id] || 0, ts);
                    deriveSets();
                }
                try {
                    const o = JSON.parse(localStorage.getItem(LS_KEY) || '{}');
                    if (o.ev) {
                        for (const k of EV_KEYS)
                            ST.ev[k] = Object.assign({}, o.ev[k] || {});
                    } else {
                        for (const id of o.done || [])
                            ST.ev.done[id] = 1;
                        // 예전 저장 형식: 시각이 없어 가장 오래된 것으로 취급
                    }
                    ST.calib = o.calib || {};
                    ST.calibAt = o.calibAt || {};
                    ST.depotAt = o.depotAt || 0;
                    deriveSets();
                } catch (e) {}
                const persist = () => {
                    try {
                        localStorage.setItem(LS_KEY, JSON.stringify({
                            done: [...ST.done],
                            tide: [...ST.tide],
                            calib: ST.calib,
                            calibAt: ST.calibAt,
                            depotAt: ST.depotAt,
                            ev: ST.ev
                        }));
                    } catch (e) {}
                }
                ;
                // 중복 의심: 같은 종류가 dupM 미터 안에 또 있으면 서로를 가리킨다 (뒤쪽 id 가 second)
                function dupInfo(dupM) {
                    const near = {};
                    const O = D.objects;
                    if (!(dupM > 0))
                        return near;
                    for (let i = 0; i < O.length; i++)
                        for (let j = i + 1; j < O.length; j++) {
                            if (O[i].code !== O[j].code)
                                continue;
                            const d = Math.hypot(O[i].x - O[j].x, O[i].y - O[j].y);
                            if (d > dupM)
                                continue;
                            if (!near[O[i].id] || d < near[O[i].id].d)
                                near[O[i].id] = { id: O[j].id, d, second: false };
                            if (!near[O[j].id] || d < near[O[j].id].d)
                                near[O[j].id] = { id: O[i].id, d, second: true };
                        }
                    return near;
                }

                // ───────── 좌표: 위경도 ↔ 투영 m (아핀 근사) ─────────
                const AF = D.affine.fwd
                  , AI = D.affine.inv;
                const ll2xy = (lon, lat) => [AF[0] * lon + AF[1] * lat + AF[2], AF[3] * lon + AF[4] * lat + AF[5]];
                const xy2ll = (x, y) => {
                    const u = x - AF[2]
                      , v = y - AF[5];
                    return [AI[0] * u + AI[1] * v, AI[2] * u + AI[3] * v];
                }
                ;
                const toLL = (p) => {
                    const ll = xy2ll(p[0], p[1]);
                    return [ll[1], ll[0]];
                }
                ;

                // ───────── 지형 격자 + 다익스트라 ─────────
                const T = D.terrain;
                let G = null
                  , NR = 0
                  , NC = 0
                  , CELL = 10
                  , GX0 = 0
                  , GY0 = 0;
                if (T) {
                    const bin = atob(T.b64);
                    G = new Uint8Array(bin.length);
                    for (let i = 0; i < bin.length; i++)
                        G[i] = bin.charCodeAt(i);
                    NR = T.nr;
                    NC = T.nc;
                    CELL = T.cell;
                    GX0 = T.x0;
                    GY0 = T.y0;
                }
                const cellOf = (x, y) => {
                    let c = Math.floor((x - GX0) / CELL)
                      , r = Math.floor((GY0 - y) / CELL);
                    r = Math.min(Math.max(r, 0), NR - 1);
                    c = Math.min(Math.max(c, 0), NC - 1);
                    return r * NC + c;
                }
                ;
                const xyOfCell = (k) => [GX0 + ((k % NC) + 0.5) * CELL, GY0 - (Math.floor(k / NC) + 0.5) * CELL];
                let runCache = {};
                function forcePassable(xy) {
                    if (!G)
                        return;
                    const k = cellOf(xy[0], xy[1]);
                    if (G[k] === 0 || G[k] === 1) {
                        G[k] = 3;
                        runCache = {};
                    }
                }
                class Heap {
                    constructor() {
                        this.k = [];
                        this.v = [];
                    }
                    push(key, val) {
                        const k = this.k
                          , v = this.v;
                        k.push(key);
                        v.push(val);
                        let i = k.length - 1;
                        while (i > 0) {
                            const p = (i - 1) >> 1;
                            if (k[p] <= k[i])
                                break;
                            [k[p],k[i]] = [k[i], k[p]];
                            [v[p],v[i]] = [v[i], v[p]];
                            i = p;
                        }
                    }
                    pop() {
                        const k = this.k
                          , v = this.v;
                        const rk = k[0]
                          , rv = v[0];
                        const lk = k.pop()
                          , lv = v.pop();
                        if (k.length) {
                            k[0] = lk;
                            v[0] = lv;
                            let i = 0;
                            const n = k.length;
                            while (true) {
                                const l = 2 * i + 1
                                  , r = l + 1;
                                let m = i;
                                if (l < n && k[l] < k[m])
                                    m = l;
                                if (r < n && k[r] < k[m])
                                    m = r;
                                if (m === i)
                                    break;
                                [k[m],k[i]] = [k[i], k[m]];
                                [v[m],v[i]] = [v[i], v[m]];
                                i = m;
                            }
                        }
                        return [rk, rv];
                    }
                    get size() {
                        return this.k.length;
                    }
                }
                function costKey(mode) {
                    return mode + '|' + JSON.stringify(T.cost[mode]);
                }
                function dijkstra(mode, src) {
                    const key = costKey(mode) + ':' + src;
                    if (runCache[key])
                        return runCache[key];
                    const cost = T.cost[mode];
                    const mult = new Float32Array(4);
                    const pass = [false, false, false, false];
                    for (const k of [0, 1, 2, 3]) {
                        const v = cost[String(k)];
                        if (v !== null && v !== undefined) {
                            mult[k] = v;
                            pass[k] = true;
                        }
                    }
                    const N = NR * NC;
                    const dist = new Float64Array(N).fill(Infinity);
                    const prev = new Int32Array(N).fill(-1);
                    const done = new Uint8Array(N);
                    const h = new Heap();
                    dist[src] = 0;
                    h.push(0, src);
                    const dirs = [[0, 1, 1], [0, -1, 1], [1, 0, 1], [-1, 0, 1], [1, 1, Math.SQRT2], [1, -1, Math.SQRT2], [-1, 1, Math.SQRT2], [-1, -1, Math.SQRT2]];
                    while (h.size) {
                        const [d,u] = h.pop();
                        if (done[u])
                            continue;
                        done[u] = 1;
                        const gu = G[u];
                        if (!pass[gu])
                            continue;
                        const r = Math.floor(u / NC)
                          , c = u % NC;
                        const wu = gu === 1;
                        for (const [dr,dc,dl] of dirs) {
                            const rr = r + dr
                              , cc = c + dc;
                            if (rr < 0 || rr >= NR || cc < 0 || cc >= NC)
                                continue;
                            const v = rr * NC + cc;
                            const gv = G[v];
                            if (!pass[gv] || done[v])
                                continue;
                            let w = CELL * dl * (mult[gu] + mult[gv]) / 2;
                            if ((gv === 1) !== wu)
                                w += T.transition_m;
                            const nd = d + w;
                            if (nd < dist[v]) {
                                dist[v] = nd;
                                prev[v] = u;
                                h.push(nd, v);
                            }
                        }
                    }
                    return runCache[key] = {
                        dist,
                        prev
                    };
                }
                function pathCells(mode, aCell, bCell) {
                    const {dist, prev} = dijkstra(mode, aCell);
                    if (!isFinite(dist[bCell]))
                        return null;
                    const out = [];
                    let j = bCell;
                    while (j >= 0 && j !== aCell) {
                        out.push(j);
                        j = prev[j];
                    }
                    out.push(aCell);
                    out.reverse();
                    return out;
                }

                // ───────── 파라미터 ─────────
                const DEF = D.defaults;
                let P = JSON.parse(JSON.stringify(DEF));
                let depot = {
                    lon: D.depot.lon,
                    lat: D.depot.lat,
                    name: D.depot.name
                };
                const defaultDepot = {
                    lon: D.depot.lon,
                    lat: D.depot.lat,
                    name: D.depot.name
                };
                function readControls() {
                    const num = (id, dflt) => {
                        const el = $('#' + id);
                        if (!el)
                            return dflt;
                        const v = parseFloat(el.value);
                        return isFinite(v) ? v : dflt;
                    }
                    ;
                    P.workers = Math.max(1, Math.round(num('c-workers', 2)));
                    P.hours_per_day = Math.max(0.5, num('c-hours', 4));
                    P.teams = Math.max(1, Math.round(num('c-teams', 1)));
                    P.link_m = num('c-link', 150);
                    P.walk_kmh = Math.max(0.5, num('c-walk', 3));
                    P.item_min = num('c-item', 2);
                    P.min_per_m2 = num('c-m2', 1.5);
                    P.carry_kg_per_person = Math.max(1, num('c-ckg', 15));
                    P.carry_bags_per_person = Math.max(1, Math.round(num('c-cbags', 2)));
                    P.min_kg = num('c-minkg', 0);
                    P.bag_kg = Math.max(1, num('c-bagkg', 15));
                    P.bag_l = Math.max(5, num('c-bagl', 80));
                    P.detour = num('c-detour', 1.4);
                    P.veg_cost = Math.max(1, num('c-veg', 3));
                    P.boat_cost = Math.max(0.05, num('c-boat', 0.5));
                    P.round_trip = $('#c-round').checked;
                    P.include_codes = [...document.querySelectorAll('.codes input:checked')].map(e => e.value);
                    P.exclude_done = $('#c-exdone') ? $('#c-exdone').checked : true;
                    P.exclude_dup = $('#c-exdup') ? $('#c-exdup').checked : false;
                    P.exclude_tide = $('#c-extide') ? $('#c-extide').checked : false;
                    P.buoy_mode = $('#c-buoy') ? $('#c-buoy').checked : false;
                    P.buoy_area_min = Math.max(0.05, num('c-buoy-area', 0.5));
                    P.buoy_fp = Math.max(0.05, num('c-buoy-fp', 0.5));
                    P.buoy_kg = [Math.max(0.01, num('c-buoy-min', 1.0)), Math.max(0.01, num('c-buoy-typ', 2.0)), Math.max(0.01, num('c-buoy-max', 5))].sort( (a, b) => a - b);
                    P.dup_m = Math.max(0, num('c-dup', 1.5));
                    for (const c of Object.keys(D.mats)) {
                        const el = $('#c-cal-' + c);
                        if (el) {
                            const v = parseFloat(el.value);
                            const nv = isFinite(v) && v > 0 ? v : 1;
                            if ((ST.calib[c] || 1) !== nv)
                                ST.calibAt[c] = Date.now();
                            ST.calib[c] = nv;
                        }
                    }
                    persist();
                    if (T) {
                        T.cost.walk['2'] = P.veg_cost;
                        T.cost.boat['2'] = P.veg_cost;
                        T.cost.boat['1'] = P.boat_cost;
                    }
                }
                function segVal(name) {
                    const b = document.querySelector(`.seg[data-name="${name}"] button.on`);
                    return b ? b.dataset.v : null;
                }
                function setSeg(name, v) {
                    document.querySelectorAll(`.seg[data-name="${name}"] button`).forEach(b => b.classList.toggle('on', b.dataset.v === v));
                }
                function opts(over) {
                    return Object.assign({
                        travel: segVal('travel'),
                        carry: segVal('carry'),
                        objective: segVal('objective'),
                        wsrc: segVal('wsrc'),
                        wstat: segVal('wstat'),
                        workers: P.workers,
                        hours: P.hours_per_day,
                        teams: P.teams,
                        scale: 1,
                        exclude_done: P.exclude_done
                    }, over || {});
                }

                // ───────── 모델 (collect.py 와 같은 규칙) ─────────
                function bagsOf(kg, m3) {
                    const byW = kg / P.bag_kg
                      , byV = m3 * D.bag.bulk / (P.bag_l / 1000);
                    if (byV >= byW)
                        return [Math.max(Math.ceil(byV), (m3 > 0 || kg > 0) ? 1 : 0), 'volume'];
                    return [Math.max(Math.ceil(byW), 1), 'weight'];
                }
                function compassName(dx, dy) {
                    const ang = (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360;
                    return COMPASS[Math.floor((ang + 22.5) / 45) % 8];
                }
                function tourOrder(Dm, start, nodes, roundTrip) {
                    const n = nodes.length;
                    if (!n)
                        return [];
                    const L = (order) => {
                        let s = 0
                          , pos = start;
                        for (const i of order) {
                            s += Dm[pos][i];
                            pos = i;
                        }
                        return s + (roundTrip ? Dm[pos][start] : 0);
                    }
                    ;
                    if (n <= 8) {
                        let best = Infinity
                          , bo = nodes.slice();
                        const perm = (arr, m) => {
                            if (!arr.length) {
                                if (roundTrip && n > 1 && m[0] > m[n - 1])
                                    return;
                                const l = L(m);
                                if (l < best) {
                                    best = l;
                                    bo = m.slice();
                                }
                                return;
                            }
                            for (let i = 0; i < arr.length; i++) {
                                const rest = arr.slice();
                                const v = rest.splice(i, 1)[0];
                                perm(rest, m.concat([v]));
                            }
                        }
                        ;
                        perm(nodes, []);
                        return bo;
                    }
                    let rem = new Set(nodes)
                      , pos = start
                      , order = [];
                    while (rem.size) {
                        let best = null
                          , bd = Infinity;
                        for (const i of rem) {
                            if (Dm[pos][i] < bd) {
                                bd = Dm[pos][i];
                                best = i;
                            }
                        }
                        order.push(best);
                        rem.delete(best);
                        pos = best;
                    }
                    let best = L(order)
                      , improved = true;
                    while (improved) {
                        improved = false;
                        for (let i = 0; i < n - 1; i++)
                            for (let j = i + 1; j < n; j++) {
                                const cand = order.slice(0, i).concat(order.slice(i, j + 1).reverse(), order.slice(j + 1));
                                const l = L(cand);
                                if (l < best - 1e-9) {
                                    order = cand;
                                    best = l;
                                    improved = true;
                                }
                            }
                        for (let i = 0; i < n; i++)
                            for (let j = 0; j < n; j++) {
                                if (i === j)
                                    continue;
                                const cand = order.slice();
                                const v = cand.splice(i, 1)[0];
                                cand.splice(j, 0, v);
                                const l = L(cand);
                                if (l < best - 1e-9) {
                                    order = cand;
                                    best = l;
                                    improved = true;
                                }
                            }
                    }
                    return order;
                }
                function computePlan(o) {
                    const t0 = performance.now();
                    const mode = (T && o.travel === 'boat') ? 'boat' : 'walk';
                    const dup = dupInfo(P.dup_m);
                    const objs = D.objects.map(ob => {
                        // 부표 모드: 큰 스티로폼 라벨은 부표 묶음으로 보고 개수 × 개당 kg 로 추정 (면적×두께 방식보다 범위가 좁음)
                        let kmin = ob.kmin, ktyp = ob.ktyp, kmax = ob.kmax, buoys = 0;
                        if (P.buoy_mode && ob.code === 'STY' && ob.area >= P.buoy_area_min) {
                            buoys = Math.max(1, Math.round(ob.area / P.buoy_fp));
                            kmin = buoys * P.buoy_kg[0];
                            ktyp = buoys * P.buoy_kg[1];
                            kmax = buoys * P.buoy_kg[2];
                        }
                        const cal = (o.wsrc === 'company') ? 1 : (ST.calib[ob.code] || 1);
                        const kg = ((o.wsrc === 'company' && ob.ckg != null) ? ob.ckg : (o.wstat === 'min' ? kmin : o.wstat === 'max' ? kmax : ktyp) * cal) * o.scale;
                        const heavy = kg > NIOSH || (ob.area >= 3 && kmax > NIOSH);
                        const done = ST.done.has(ob.id);
                        const tide = ST.tide.has(ob.id);
                        const dp = dup[ob.id] || null;
                        const inc = P.include_codes.includes(ob.code) && kg >= P.min_kg && !(o.exclude_done && done) && !(P.exclude_dup && dp && dp.second) && !(P.exclude_tide && tide);
                        return Object.assign({}, ob, {
                            kmin,
                            ktyp,
                            kmax,
                            buoys,
                            kg,
                            heavy,
                            inc,
                            done,
                            tide,
                            dup: dp,
                            zone: -1,
                            order: 0
                        });
                    }
                    );
                    const inc = objs.filter(ob => ob.inc);
                    const n = inc.length;
                    const dxy = ll2xy(depot.lon, depot.lat);
                    const nodes = [{
                        x: dxy[0],
                        y: dxy[1]
                    }].concat(inc.map(ob => ({
                        x: ob.x,
                        y: ob.y
                    })));
                    let Dm, cells = null, unreachable = 0;
                    if (T && n) {
                        nodes.forEach(nd => forcePassable([nd.x, nd.y]));
                        cells = nodes.map(nd => cellOf(nd.x, nd.y));
                        Dm = nodes.map( (a, i) => {
                            const {dist} = dijkstra(mode, cells[i]);
                            return cells.map( (c, j) => i === j ? 0 : dist[c]);
                        }
                        );
                        for (let i = 0; i <= n; i++)
                            for (let j = 0; j <= n; j++)
                                if (!isFinite(Dm[i][j])) {
                                    unreachable++;
                                    Dm[i][j] = Math.hypot(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y) * P.detour * 3;
                                }
                    } else {
                        Dm = nodes.map(a => nodes.map(b => Math.hypot(a.x - b.x, a.y - b.y) * P.detour));
                    }
                    const seg = (i, j) => {
                        if (T && cells) {
                            const pc = pathCells(mode, cells[i], cells[j]);
                            if (pc) {
                                const pts = pc.map(xyOfCell);
                                return [[nodes[i].x, nodes[i].y]].concat(pts.slice(1, -1), [[nodes[j].x, nodes[j].y]]);
                            }
                        }
                        return [[nodes[i].x, nodes[i].y], [nodes[j].x, nodes[j].y]];
                    }
                    ;
                    const parent = Array.from({
                        length: n
                    }, (_, i) => i);
                    const find = (i) => {
                        while (parent[i] !== i) {
                            parent[i] = parent[parent[i]];
                            i = parent[i];
                        }
                        return i;
                    }
                    ;
                    for (let i = 0; i < n; i++)
                        for (let j = i + 1; j < n; j++)
                            if (Dm[i + 1][j + 1] <= P.link_m) {
                                const a = find(i)
                                  , b = find(j);
                                if (a !== b)
                                    parent[a] = b;
                            }
                    const remap = {};
                    const groups = {};
                    inc.forEach( (ob, i) => {
                        const r = find(i);
                        if (!(r in remap))
                            remap[r] = Object.keys(remap).length;
                        ob.zone = remap[r];
                        (groups[ob.zone] = groups[ob.zone] || []).push(i + 1);
                    }
                    );
                    const zoneIds = Object.keys(groups).map(Number).sort( (a, b) => a - b);
                    const Z = zoneIds.length;
                    const zkg = {}
                      , zwork = {};
                    for (const z of zoneIds) {
                        zkg[z] = groups[z].reduce( (s, i) => s + inc[i - 1].kg, 0);
                        zwork[z] = groups[z].reduce( (s, i) => s + P.item_min + P.min_per_m2 * inc[i - 1].area, 0) / o.workers + DEF.heavy_extra_min * groups[z].filter(i => inc[i - 1].heavy).length;
                    }
                    const ZD = Array.from({
                        length: Z + 1
                    }, () => new Array(Z + 1).fill(0));
                    zoneIds.forEach( (za, ai) => {
                        const a = ai + 1;
                        ZD[0][a] = ZD[a][0] = Math.min(...groups[za].map(i => Dm[0][i]));
                        zoneIds.forEach( (zb, bi) => {
                            const b = bi + 1;
                            if (a !== b) {
                                let m = Infinity;
                                for (const i of groups[za])
                                    for (const j of groups[zb])
                                        if (Dm[i][j] < m)
                                            m = Dm[i][j];
                                ZD[a][b] = m;
                            }
                        }
                        );
                    }
                    );
                    const wmpm = 60 / (P.walk_kmh * 1000);
                    let order;
                    if (o.objective === 'weight' && Z) {
                        let rem = zoneIds.map( (_, i) => i + 1)
                          , pos = 0;
                        order = [];
                        while (rem.length) {
                            let best = null
                              , bv = -1;
                            for (const a of rem) {
                                const z = zoneIds[a - 1];
                                const v = zkg[z] / Math.max(ZD[pos][a] * wmpm + zwork[z], 1e-6);
                                if (v > bv) {
                                    bv = v;
                                    best = a;
                                }
                            }
                            order.push(best);
                            rem = rem.filter(x => x !== best);
                            pos = best;
                        }
                    } else
                        order = tourOrder(ZD, 0, zoneIds.map( (_, i) => i + 1), P.round_trip);
                    const ordered = order.map(a => zoneIds[a - 1]);
                    const cents = {};
                    for (const z of zoneIds) {
                        const L = groups[z].map(i => inc[i - 1]);
                        cents[z] = [L.reduce( (s, ob) => s + ob.x, 0) / L.length, L.reduce( (s, ob) => s + ob.y, 0) / L.length];
                    }
                    const dirCount = {}
                      , seen = {}
                      , names = {};
                    for (const z of ordered) {
                        const d = compassName(cents[z][0] - D.center.x, cents[z][1] - D.center.y);
                        dirCount[d] = (dirCount[d] || 0) + 1;
                    }
                    for (const z of ordered) {
                        const d = compassName(cents[z][0] - D.center.x, cents[z][1] - D.center.y);
                        seen[d] = (seen[d] || 0) + 1;
                        names[z] = d + '쪽 해안' + (dirCount[d] > 1 && seen[d] <= CIRC.length ? ' ' + CIRC[seen[d] - 1] : '');
                    }
                    const capKg = o.workers * P.carry_kg_per_person
                      , capBags = o.workers * P.carry_bags_per_person;
                    // 한 팀이 seq 순서로 도는 함수
                    function traverse(seq, startStep, team) {
                        const zones = [];
                        let pos = 0
                          , cumMin = 0
                          , cumBags = 0
                          , loadKg = 0
                          , loadBags = 0;
                        const speedF = () => 1 - Math.min(0.4, DEF.load_slow * loadKg / o.workers);
                        seq.forEach( (z, si) => {
                            const step = startStep + si;
                            let members = groups[z].slice();
                            const inner = [];
                            let cur = pos;
                            while (members.length) {
                                let best = null
                                  , bd = Infinity;
                                for (const i of members) {
                                    if (Dm[cur][i] < bd) {
                                        bd = Dm[cur][i];
                                        best = i;
                                    }
                                }
                                inner.push(best);
                                members = members.filter(x => x !== best);
                                cur = best;
                            }
                            inner.forEach( (i, k) => {
                                inc[i - 1].order = k + 1;
                            }
                            );
                            const dIn = Dm[pos][inner[0]];
                            const segs = [];
                            let walkMin = 0
                              , distTot = 0
                              , returns = 0;
                            cur = pos;
                            let approach = null;
                            for (const i of inner) {
                                const ob = inc[i - 1];
                                const b = bagsOf(ob.kg, ob.vol)[0];
                                if (o.carry === 'carry' && (loadKg > 0 || loadBags > 0) && (loadKg + ob.kg > capKg || loadBags + b > capBags)) {
                                    const back = Dm[cur][0]
                                      , out = Dm[0][i];
                                    walkMin += back * wmpm / speedF();
                                    loadKg = 0;
                                    loadBags = 0;
                                    walkMin += out * wmpm / speedF();
                                    distTot += back + out;
                                    segs.push(seg(cur, 0));
                                    segs.push(seg(0, i));
                                    returns++;
                                } else {
                                    const d = Dm[cur][i];
                                    walkMin += d * wmpm / speedF();
                                    distTot += d;
                                    segs.push(seg(cur, i));
                                }
                                if (approach === null)
                                    approach = segs.length === 1 ? segs[0] : segs[0].concat(segs[1].slice(1));
                                if (o.carry === 'carry') {
                                    loadKg += ob.kg;
                                    loadBags += b;
                                }
                                cur = i;
                            }
                            pos = cur;
                            const L = inner.map(i => inc[i - 1]);
                            const kg = L.reduce( (s, ob) => s + ob.kg, 0)
                              , m3 = L.reduce( (s, ob) => s + ob.vol, 0);
                            const [bags] = bagsOf(kg, m3);
                            const heavy = L.filter(ob => ob.heavy);
                            const workMin = zwork[z];
                            cumMin += walkMin + workMin;
                            cumBags += bags;
                            const byCode = {};
                            for (const ob of L)
                                byCode[ob.code] = (byCode[ob.code] || 0) + 1;
                            const tools = [...new Set(Object.keys(byCode).map(c => D.mats[c] && D.mats[c].tool).filter(Boolean))].sort();
                            const notes = [];
                            if (heavy.length)
                                notes.push(`무거운 물체 ${heavy.length}개 → 2인 이상 또는 장비 (NIOSH 23 kg 초과 가능)`);
                            const big = L.filter(ob => ob.area >= 3);
                            if (big.length)
                                notes.push(`면적 3 m² 이상 큰 물체 ${big.length}개 (${big.map(ob => `${ob.ko} ${ob.area.toFixed(1)} m²`).join(', ')})`);
                            if (returns)
                                notes.push(`적재량 초과로 출발지 복귀 ${returns}회 포함`);
                            const dups = L.filter(ob => ob.dup);
                            if (dups.length)
                                notes.push(`중복 의심 ${dups.length}개 (같은 종류가 ${P.dup_m} m 안에 또 있음) → 현장에서 같은 물체인지 확인`);
                            const tides = L.filter(ob => ob.tide);
                            if (tides.length)
                                notes.push('🌊 만조 주의 구역 → 물때표 확인, 간조 전후로 배정');
                            const buoyN = L.reduce( (s, ob) => s + (ob.buoys || 0), 0);
                            if (buoyN)
                                notes.push(`스티로폼 부표 추정 약 ${buoyN}개 (개수 × 개당 kg 로 무게 추정)`);
                            const path = [];
                            segs.forEach( (sg, si2) => {
                                sg.forEach( (q, k) => {
                                    if (!(si2 > 0 && k === 0))
                                        path.push(q);
                                }
                                );
                            }
                            );
                            const cll = xy2ll(cents[z][0], cents[z][1]);
                            zones.push({
                                zone: z,
                                step,
                                team,
                                name: names[z],
                                cx: cents[z][0],
                                cy: cents[z][1],
                                lon: cll[0],
                                lat: cll[1],
                                objects: L,
                                n: L.length,
                                byCode,
                                kg,
                                kmin: L.reduce( (s, ob) => s + ob.kmin, 0),
                                kmax: L.reduce( (s, ob) => s + ob.kmax, 0),
                                ckg: L.reduce( (s, ob) => s + (ob.ckg || 0), 0),
                                m3,
                                bags,
                                heavy,
                                tools,
                                notes,
                                path,
                                approach: approach || [],
                                dIn,
                                distTot,
                                walkMin,
                                workMin,
                                cumMin,
                                cumBags,
                                returns,
                                tide: tides.length > 0,
                                buoys: buoyN,
                                day: 1,
                                firstCell: cells ? cells[inner[0]] : null
                            });
                        }
                        );
                        let backM = 0
                          , backMin = 0
                          , backPath = [];
                        if (P.round_trip && zones.length) {
                            backM = Dm[pos][0];
                            backMin = backM * wmpm / speedF();
                            backPath = seg(pos, 0);
                        }
                        return {
                            zones,
                            backM,
                            backMin,
                            backPath
                        };
                    }
                    // 팀 분할: 한 팀 순회를 시간 균형(전체 누적 기준)으로 연속 구간으로
                    const nTeams = Math.max(1, o.teams);
                    let segments;
                    if (nTeams > 1 && ordered.length > 1) {
                        const single = traverse(ordered, 1, 1);
                        const tot = single.zones.reduce( (s, z) => s + z.walkMin + z.workMin, 0);
                        segments = [];
                        let cur = []
                          , cum = 0;
                        single.zones.forEach( (z, k) => {
                            cur.push(ordered[k]);
                            cum += z.walkMin + z.workMin;
                            const remaining = single.zones.length - k - 1;
                            const teamsLeft = nTeams - segments.length - 1;
                            if (segments.length < nTeams - 1 && remaining >= teamsLeft && (cum >= (segments.length + 1) * tot / nTeams || remaining === teamsLeft)) {
                                segments.push(cur);
                                cur = [];
                            }
                        }
                        );
                        if (cur.length)
                            segments.push(cur);
                    } else
                        segments = ordered.length ? [ordered] : [];
                    const zones = []
                      , teams = []
                      , days = [];
                    let routeLen = 0
                      , totalWalk = 0
                      , totalWork = 0
                      , step0 = 1;
                    const dayCap = o.hours * 60;
                    segments.forEach( (seq, ti) => {
                        const t = ti + 1;
                        const r = traverse(seq, step0, t);
                        step0 += r.zones.length;
                        const walk = r.zones.reduce( (s, z) => s + z.walkMin, 0) + r.backMin
                          , work = r.zones.reduce( (s, z) => s + z.workMin, 0);
                        routeLen += r.zones.reduce( (s, z) => s + z.distTot, 0) + r.backM;
                        totalWalk += walk;
                        totalWork += work;
                        let day = 1
                          , acc = 0;
                        for (const z of r.zones) {
                            if (acc > 0 && acc + z.walkMin + z.workMin > dayCap) {
                                days.push({
                                    team: t,
                                    day,
                                    steps: r.zones.filter(q => q.day === day).map(q => q.step),
                                    minutes: acc
                                });
                                day++;
                                acc = 0;
                            }
                            z.day = day;
                            acc += z.walkMin + z.workMin;
                        }
                        if (r.zones.length)
                            days.push({
                                team: t,
                                day,
                                steps: r.zones.filter(q => q.day === day).map(q => q.step),
                                minutes: acc + r.backMin
                            });
                        teams.push({
                            team: t,
                            zones: r.zones.map(z => z.step),
                            n: r.zones.reduce( (s, z) => s + z.n, 0),
                            kg: r.zones.reduce( (s, z) => s + z.kg, 0),
                            bags: r.zones.reduce( (s, z) => s + z.bags, 0),
                            walkMin: walk,
                            workMin: work,
                            minutes: walk + work,
                            days: r.zones.length ? day : 0,
                            backPath: r.backPath,
                            routeM: r.zones.reduce( (s, z) => s + z.distTot, 0) + r.backM
                        });
                        zones.push(...r.zones);
                    }
                    );
                    const totalMin = teams.length ? Math.max(...teams.map(t => t.minutes)) : 0;
                    const kgPlan = inc.reduce( (s, ob) => s + ob.kg, 0)
                      , vol = inc.reduce( (s, ob) => s + ob.vol, 0);
                    const doneObjs = objs.filter(ob => ob.done);
                    const allSel = objs.filter(ob => P.include_codes.includes(ob.code) && ob.kg >= P.min_kg);
                    const totals = {
                        kg: kgPlan,
                        kmin: inc.reduce( (s, ob) => s + ob.kmin, 0),
                        kmax: inc.reduce( (s, ob) => s + ob.kmax, 0),
                        ckg: inc.reduce( (s, ob) => s + (ob.ckg || 0), 0),
                        vol,
                        bags: zones.reduce( (s, z) => s + z.bags, 0),
                        heavy: zones.reduce( (s, z) => s + z.heavy.length, 0),
                        zones: zones.length,
                        returns: zones.reduce( (s, z) => s + z.returns, 0),
                        tonbags: inc.length ? Math.ceil(Math.max(kgPlan / D.bag.tonbag_kg, vol * D.bag.bulk / D.bag.tonbag_m3)) : 0,
                        done: doneObjs.length,
                        doneKg: doneObjs.reduce( (s, ob) => s + ob.kg, 0),
                        all: allSel.length,
                        allKg: allSel.reduce( (s, ob) => s + ob.kg, 0)
                    };
                    const byCode = {};
                    for (const ob of inc) {
                        const d = byCode[ob.code] = byCode[ob.code] || {
                            ko: ob.ko,
                            color: ob.color,
                            count: 0,
                            area: 0,
                            kg: 0,
                            kmin: 0,
                            kmax: 0,
                            ckg: 0,
                            vol: 0
                        };
                        d.count++;
                        d.area += ob.area;
                        d.kg += ob.kg;
                        d.kmin += ob.kmin;
                        d.kmax += ob.kmax;
                        d.ckg += ob.ckg || 0;
                        d.vol += ob.vol;
                    }
                    const equipment = [`마대 ${Math.ceil(totals.bags * 1.2)}장 (계산 ${totals.bags}장 + 여유 20 %)`, `장갑·집게 ${o.workers * nTeams}명분`];
                    const tools = [...new Set(zones.flatMap(z => z.tools))].sort();
                    if (tools.length)
                        equipment.push(tools.join(' / ') + ' (로프·그물 자르기)');
                    if (totals.tonbags >= 1 && vol * D.bag.bulk > 0.5)
                        equipment.push(`톤백 ${totals.tonbags}개 또는 집결지 적재 공간 ${(vol * D.bag.bulk).toFixed(1)} m³`);
                    if (totals.heavy)
                        equipment.push(`무거운 물체 ${totals.heavy}개 → 2인 운반 또는 손수레`);
                    if (mode === 'boat')
                        equipment.push('보트·구명조끼, 승·하선 지점 사전 확인');
                    if (nTeams > 1)
                        equipment.push(`팀 ${nTeams}개 → 팀별 무전기·연락 수단, 집결 시각 약속`);
                    equipment.push('식수·구급약, 물때표 확인 (갯바위 구간)');
                    return {
                        o,
                        objs,
                        inc,
                        zones,
                        teams,
                        days,
                        totals,
                        byCode,
                        equipment,
                        routeLen,
                        totalWalk,
                        totalWork,
                        totalMin,
                        mode,
                        unreachable,
                        ms: performance.now() - t0,
                        skipped: allSel.length - inc.length,
                        nTeams
                    };
                }

                // ───────── 표시 ─────────
                const fmtKg = (kg) => kg >= 100 ? kg.toLocaleString('ko', {
                    maximumFractionDigits: 0
                }) : kg >= 1 ? kg.toFixed(1) : kg.toFixed(2);
                const fmtMin = (m) => m >= 60 ? `${Math.floor(m / 60)}시간 ${Math.round(m % 60)}분` : `${Math.round(m)}분`;
                const colorOf = (z, R) => (R.nTeams > 1 ? TEAMC[(z.team - 1) % TEAMC.length] : DAYC[(z.day - 1) % DAYC.length]);
                const teamName = (t) => TEAMN[t - 1] || String(t);
                const navLinks = (name, lat, lon) => `<a href="https://map.kakao.com/link/to/${encodeURIComponent(name)},${lat.toFixed(6)},${lon.toFixed(6)}" target="_blank" rel="noopener">카카오맵 길찾기</a> · <a href="https://www.google.com/maps/dir/?api=1&destination=${lat.toFixed(6)},${lon.toFixed(6)}&travelmode=walking" target="_blank" rel="noopener">구글 지도</a>`;
                let map = null
                  , layers = null
                  , zoneMarkers = {}
                  , depotMarker = null
                  , meMarker = null
                  , allBounds = null
                  , fitted = false;
                // 출발지 제한: 정사영상 범위 안 + 땅(맨땅·숲) 또는 해안 2칸(20 m) 이내의 물(부두·선착장). 벗어나면 이유 문자열 반환
                function depotProblem(lon, lat) {
                    if (D.basemap) {
                        const b = D.basemap.bounds;
                        if (lat < b[0] || lat > b[2] || lon < b[1] || lon > b[3])
                            return '정사영상 범위 밖';
                    }
                    if (!T)
                        return null;
                    const [x,y] = ll2xy(lon, lat);
                    const c = Math.floor((x - GX0) / CELL)
                      , r = Math.floor((GY0 - y) / CELL);
                    if (r < 0 || r >= NR || c < 0 || c >= NC)
                        return '지형 격자 밖';
                    const g = G[r * NC + c];
                    if (g === 2 || g === 3)
                        return null;
                    if (g === 0)
                        return '촬영되지 않은 영역';
                    for (let dr = -2; dr <= 2; dr++)
                        for (let dc = -2; dc <= 2; dc++) {
                            const rr = r + dr
                              , cc = c + dc;
                            if (rr < 0 || rr >= NR || cc < 0 || cc >= NC)
                                continue;
                            const gg = G[rr * NC + cc];
                            if (gg === 2 || gg === 3)
                                return null;
                        }
                    return '해안에서 20 m 넘게 떨어진 바다';
                }
                function initMap() {
                    const fb = $('#fallback');
                    if (typeof L === 'undefined') {
                        fb.style.display = 'block';
                        $('#maphint').textContent = '인터넷 연결이 없어 인쇄용 지도(기본 설정)를 보여줍니다. 숫자·카드는 아래에서 계속 다시 계산됩니다. 출발 전에 인터넷이 되는 곳에서 이 페이지를 한 번 열어 두면 지도 라이브러리가 브라우저에 저장됩니다.';
                        return;
                    }
                    map = L.map('map', {
                        zoomControl: true,
                        preferCanvas: true,
                        scrollWheelZoom: false
                    });
                    map.on('click focus', () => map.scrollWheelZoom.enable());
                    map.on('mouseout', () => map.scrollWheelZoom.disable());
                    const base = {};
                    const ov = {};
                    if (!D.no_tiles) {
                        const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                            maxZoom: 21,
                            maxNativeZoom: 19,
                            attribution: 'Esri World Imagery'
                        });
                        const osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
                            maxZoom: 19,
                            attribution: '© OpenStreetMap'
                        });
                        sat.addTo(map);
                        let ok = false;
                        sat.on('tileload', () => {
                            ok = true;
                        }
                        );
                        setTimeout( () => {
                            if (!ok && !D.basemap) {
                                fb.style.display = 'block';
                            }
                        }
                        , 6000);
                        base['위성 지도'] = sat;
                        base['일반 지도'] = osm;
                    } else if (!D.basemap) {
                        fb.style.display = 'block';
                    }
                    if (D.basemap) {
                        const o = L.imageOverlay(D.basemap.url, [[D.basemap.bounds[0], D.basemap.bounds[1]], [D.basemap.bounds[2], D.basemap.bounds[3]]], {
                            opacity: 1,
                            zIndex: 5
                        });
                        o.addTo(map);
                        ov['드론 정사영상 (조사일)'] = o;
                    }
                    if (D.terrain_png) {
                        const tb = D.terrain_png.bounds;
                        const o = L.imageOverlay(D.terrain_png.url, [[tb[0], tb[1]], [tb[2], tb[3]]], {
                            opacity: .45,
                            zIndex: 6
                        });
                        ov['지형 분류 (파랑 물·초록 숲·흰색 맨땅)'] = o;
                    }
                    layers = {
                        route: L.layerGroup().addTo(map),
                        obj: L.layerGroup().addTo(map),
                        zone: L.layerGroup().addTo(map)
                    };
                    ov['이동 경로'] = layers.route;
                    ov['쓰레기 위치'] = layers.obj;
                    ov['구역 번호'] = layers.zone;
                    L.control.layers(Object.keys(base).length ? base : null, ov, {
                        collapsed: true,
                        position: 'topright'
                    }).addTo(map);
                    L.control.scale({
                        imperial: false
                    }).addTo(map);
                    if (D.no_tiles) {
                        map.options.maxZoom = 20;
                        map.setMaxZoom(20);
                    }
                    depotMarker = L.marker([depot.lat, depot.lon], {
                        icon: L.divIcon({
                            className: '',
                            html: '<div class="dp" title="끌어서 출발지 변경">★</div>',
                            iconSize: [30, 30],
                            iconAnchor: [15, 15]
                        }),
                        zIndexOffset: 2000,
                        draggable: true
                    }).addTo(map);
                    depotMarker.bindTooltip('출발·집결지 (끌어서 옮길 수 있음)', {
                        direction: 'top',
                        offset: [0, -12]
                    });
                    depotMarker.on('dragend', () => {
                        const ll = depotMarker.getLatLng();
                        const why = depotProblem(ll.lng, ll.lat);
                        if (why) {
                            depotMarker.setLatLng([depot.lat, depot.lon]);
                            $('#status').textContent = '출발지를 옮길 수 없습니다: ' + why + ' — 정사영상 안의 땅이나 해안 20 m 이내(부두 포함)에만 둘 수 있습니다';
                            $('#status').style.color = '#c0392b';
                            setTimeout( () => {
                                $('#status').style.color = '';
                            }
                            , 4000);
                            return;
                        }
                        depot = {
                            lon: ll.lng,
                            lat: ll.lat,
                            name: '지정 출발지'
                        };
                        ST.depotAt = Date.now();
                        persist();
                        recompute();
                    }
                    );
                    allBounds = L.latLngBounds(D.objects.map(ob => [ob.lat, ob.lon]).concat([[depot.lat, depot.lon]]));
                    const refit = () => {
                        const el = map.getContainer();
                        if (el.clientWidth > 0 && el.clientHeight > 0) {
                            map.invalidateSize();
                            if (!fitted) {
                                fitted = true;
                                map.fitBounds(allBounds.pad(0.08), {
                                    animate: false
                                });
                            }
                        }
                    }
                    ;
                    refit();
                    map.whenReady(refit);
                    window.addEventListener('load', refit);
                    setTimeout(refit, 300);
                    setTimeout(refit, 1500);
                    if (window.ResizeObserver)
                        new ResizeObserver(refit).observe(map.getContainer());
                    const iv = setInterval( () => {
                        refit();
                        if (fitted)
                            clearInterval(iv);
                    }
                    , 500);
                    setTimeout( () => clearInterval(iv), 20000);
                    document.addEventListener('visibilitychange', refit);
                    ['pointermove', 'touchstart', 'scroll', 'focus'].forEach(ev => window.addEventListener(ev, refit, {
                        passive: true
                    }));
                    window.fitAll = () => {
                        map.invalidateSize();
                        map.fitBounds(allBounds.pad(0.08));
                    }
                    ;
                    window._map = map;
                }
                function renderMap(R) {
                    if (!map)
                        return;
                    layers.route.clearLayers();
                    layers.obj.clearLayers();
                    layers.zone.clearLayers();
                    zoneMarkers = {};
                    for (const z of R.zones) {
                        const col = colorOf(z, R);
                        const ap = z.approach.map(toLL);
                        const rest = z.path.slice(Math.max(z.approach.length - 1, 0)).map(toLL);
                        if (ap.length > 1) {
                            L.polyline(ap, {
                                color: '#fff',
                                weight: 7,
                                opacity: .85
                            }).addTo(layers.route);
                            L.polyline(ap, {
                                color: col,
                                weight: 3.5,
                                dashArray: '10 8'
                            }).addTo(layers.route);
                        }
                        if (rest.length > 1) {
                            L.polyline(rest, {
                                color: '#fff',
                                weight: 7,
                                opacity: .85
                            }).addTo(layers.route);
                            L.polyline(rest, {
                                color: col,
                                weight: 4
                            }).addTo(layers.route);
                        }
                    }
                    for (const t of R.teams) {
                        if (t.backPath.length > 1) {
                            const bp = t.backPath.map(toLL);
                            const last = R.zones.filter(z => z.team === t.team).slice(-1)[0];
                            const col = last ? colorOf(last, R) : '#555';
                            L.polyline(bp, {
                                color: '#fff',
                                weight: 7,
                                opacity: .85
                            }).addTo(layers.route);
                            L.polyline(bp, {
                                color: col,
                                weight: 3.5,
                                dashArray: '4 8'
                            }).addTo(layers.route);
                        }
                    }
                    const stepOf = {};
                    R.zones.forEach(z => z.objects.forEach(ob => stepOf[ob.id] = z.step));
                    for (const ob of R.objs) {
                        const m = L.circleMarker([ob.lat, ob.lon], {
                            radius: ob.inc ? (ob.heavy ? 9 : 6) : 4,
                            color: ob.done ? '#1baf7a' : (ob.inc ? '#111' : '#777'),
                            weight: ob.done ? 2 : 1.2,
                            fillColor: ob.inc ? ob.color : (ob.done ? '#b9f0d8' : '#bbb'),
                            fillOpacity: .95
                        }).addTo(layers.obj);
                        const state = ob.done ? '✅ 완료' : (ob.inc ? `${stepOf[ob.id]}-${ob.order}` : '(제외)');
                        m.bindPopup(`<b>${state} · ${esc(ob.ko)}</b><br>크기 ${ob.w.toFixed(1)}×${ob.h.toFixed(1)} m (${ob.area.toFixed(1)} m²)<br>무게 ${fmtKg(ob.kg)} kg (범위 ${fmtKg(ob.kmin)}–${fmtKg(ob.kmax)})<br><span style="color:#888">기업 지수 ${ob.ckg != null ? ob.ckg.toFixed(3) : '-'} (참고, kg 아님)</span>${ob.buoys ? `<br>부표 추정 ${ob.buoys}개` : ''}${ob.dup ? `<br><span style="color:#c0392b">중복 의심: ${esc(ob.dup.id)} 와 ${ob.dup.d.toFixed(1)} m</span>` : ''}${ob.tide ? '<br>🌊 만조 주의' : ''}${ob.heavy ? '<br><b style="color:#c0392b">⚠ 2인 운반</b>' : ''}${ob.img ? `<img src="${ob.img}" alt="">` : ''}<br><a href="#" onclick="toggleDone(['${ob.id}']);return false;">${ob.done ? '완료 취소' : '이 물체 완료'}</a>${ob.inc ? ` · <a href="#" onclick="openZone(${stepOf[ob.id]},false);return false;">구역 목록 보기</a>` : ''}`);
                    }
                    for (const z of R.zones) {
                        const col = colorOf(z, R);
                        const lab = (R.nTeams > 1 ? teamName(z.team) : '') + z.step;
                        const ic = L.divIcon({
                            className: '',
                            html: `<div class="zn" style="border-color:${col}">${lab}</div>`,
                            iconSize: [34, 34],
                            iconAnchor: [17, 17]
                        });
                        const m = L.marker([z.lat, z.lon], {
                            icon: ic,
                            zIndexOffset: 1000
                        }).addTo(layers.zone);
                        zoneMarkers[z.step] = m;
                        const comp = Object.entries(z.byCode).sort( (a, b) => b[1] - a[1]).map( ([c,n]) => `${esc(D.mats[c] ? D.mats[c].ko : c)} ${n}개`).join(', ');
                        m.bindPopup(`<b>${z.step}. ${esc(z.name)}</b>${R.nTeams > 1 ? ` · ${teamName(z.team)}팀` : ''}<br>${comp}<br>무게 ${fmtKg(z.kg)} kg · 마대 ${z.bags}장<br>접근 ${Math.round(z.dIn)} m · 이동 ${Math.round(z.walkMin)}분 · 작업 ${Math.round(z.workMin)}분 ${z.heavy.length ? `<br><b style="color:#c0392b">⚠ 2인 운반 물체 ${z.heavy.length}개</b>` : ''}<br>${navLinks(z.name, z.lat, z.lon)}<br><a href="#" onclick="doneZone(${z.step});return false;">이 구역 완료</a>`);
                        m.unbindPopup();
                        m.on('click', () => {
                            highlight(z.step, false);
                            openZone(z.step);
                        }
                        );
                    }
                    if (depotMarker)
                        depotMarker.setLatLng([depot.lat, depot.lon]);
                    if (PANEL_IDS) {
                        const z = R.zones.find(z => z.objects.some(ob => PANEL_IDS.includes(ob.id)));
                        if (z)
                            openZone(z.step, false);
                        else
                            closeZone();
                    }
                }
                window.highlight = (step, fly=true) => {
                    document.querySelectorAll('.step').forEach(e => e.classList.toggle('active', +e.dataset.step === step));
                    if (!CUR)
                        return;
                    const z = CUR.zones.find(z => z.step === step);
                    if (!z)
                        return;
                    if (fly) {
                        $('#map').scrollIntoView({
                            behavior: 'smooth',
                            block: 'center'
                        });
                        openZone(step, true);
                    }
                }
                ;
                // ───────── 구역 상세 패널 (waypoint 를 누르면 수거할 물체 목록) ─────────
                let PANEL_IDS = null;
                window.closeZone = () => {
                    PANEL_IDS = null;
                    const el = $('#zpanel');
                    el.classList.remove('open');
                    el.innerHTML = '';
                }
                ;
                window.zoomZone = (step) => {
                    if (!map || !CUR)
                        return;
                    const z = CUR.zones.find(z => z.step === step);
                    if (!z)
                        return;
                    const b = L.latLngBounds(z.path.map(toLL));
                    map.flyToBounds(b.pad(0.4), {
                        maxZoom: 19,
                        paddingBottomRight: [window.innerWidth > 600 ? 380 : 0, window.innerWidth > 600 ? 0 : Math.round(map.getSize().y * 0.6)]
                    });
                }
                ;
                window.flyObj = (id) => {
                    if (!map || !CUR)
                        return;
                    const ob = CUR.objs.find(o => o.id === id);
                    if (!ob)
                        return;
                    map.flyTo([ob.lat, ob.lon], 19);
                }
                ;
                window.openZone = (step, fly=true) => {
                    if (!CUR)
                        return;
                    const R = CUR;
                    const z = R.zones.find(z => z.step === step);
                    if (!z) {
                        closeZone();
                        return;
                    }
                    PANEL_IDS = z.objects.map(ob => ob.id);
                    const col = colorOf(z, R);
                    const chips = Object.entries(z.byCode).sort( (a, b) => b[1] - a[1]).map( ([c,n]) => `<span class="chip" style="background:${D.mats[c] ? D.mats[c].color : '#ccc'}">${esc(D.mats[c] ? D.mats[c].ko : c)} ${n}</span>`).join('');
                    const rows = z.objects.map( (ob, i) => `<li class="${ob.done ? 'done' : ''}" onclick="flyObj('${ob.id}')">${ob.img ? `<img src="${ob.img}" alt="">` : '<div class="noimg"></div>'}<div class="ot"><b>${z.step}-${i + 1} ${esc(ob.ko)}</b>${ob.heavy ? ' <span style="color:#c0392b;font-weight:700">⚠ 2인</span>' : ''}${ob.dup ? ' <span style="color:#c0392b;font-weight:700" title="같은 종류가 가까이 또 있음">중복?</span>' : ''}${ob.buoys ? ` <span style="color:var(--muted)">부표 ${ob.buoys}개</span>` : ''}<br>${ob.w.toFixed(1)}×${ob.h.toFixed(1)} m · ${ob.area.toFixed(1)} m²<br><b>${fmtKg(ob.kg)} kg</b> <span style="color:var(--muted)">(${fmtKg(ob.kmin)}–${fmtKg(ob.kmax)})</span></div><input type="checkbox" title="완료" ${ob.done ? 'checked' : ''} onclick="event.stopPropagation();toggleDone(['${ob.id}'])"></li>`).join('');
                    const notes = z.notes.map(n => `<div class="warn">⚠ ${esc(n)}</div>`).join('');
                    const tools = z.tools.length ? `<p class="sub" style="margin:6px 0 0">도구: ${esc(z.tools.join(', '))}</p>` : '';
                    $('#zpanel').innerHTML = `<div class="zh"><div class="stepnum" style="background:${col}">${z.step}</div><h3>${esc(z.name)}${R.nTeams > 1 ? ` · ${teamName(z.team)}팀` : ''} · ${z.day}일차</h3><button class="x" onclick="closeZone()" title="닫기">✕</button></div><div class="chips">${chips}</div><div class="facts"><div class="fact"><b>${z.n}</b><span>개</span></div><div class="fact"><b>${fmtKg(z.kg)}</b><span>kg</span></div><div class="fact"><b>${z.bags}</b><span>마대</span></div><div class="fact"><b>${Math.round(z.walkMin + z.workMin)}</b><span>분</span></div></div><p class="sub" style="margin:4px 0">이전 지점에서 ${Math.round(z.dIn)} m · 이동 ${Math.round(z.walkMin)}분 · 작업 ${Math.round(z.workMin)}분</p>${notes}${tools}<p class="sub" style="margin:8px 0 2px;font-weight:700;color:var(--ink)">수거할 물체 (체크 = 완료, 누르면 지도 이동)</p><ul class="olist">${rows}</ul><div class="links">${navLinks(z.name, z.lat, z.lon)}</div><div class="btns"><button class="primary" onclick="doneZone(${z.step})">구역 전체 완료 ✓</button><button onclick="zoomZone(${z.step})">구역 확대</button><button class="tidebtn${z.tide ? ' on' : ''}" onclick="toggleTide(${z.step})">🌊 만조 주의 ${z.tide ? '해제' : '표시'}</button></div>`;
                    $('#zpanel').classList.add('open');
                    if (fly)
                        zoomZone(step);
                }
                ;
                window.showDay = (team, d) => {
                    if (!map || !CUR)
                        return;
                    const zs = CUR.zones.filter(z => z.team === team && z.day === d);
                    if (!zs.length)
                        return;
                    const pts = [].concat(...zs.map(z => z.path.map(toLL)));
                    map.flyToBounds(L.latLngBounds(pts).pad(0.15));
                }
                ;
                function renderTiles(R) {
                    const t = R.totals;
                    const tiles = [['수거할 쓰레기', `${R.inc.length}개`, `구역 ${t.zones}곳${R.skipped ? ` · 제외 ${R.skipped}개` : ''}`], ['예상 무게', `${fmtKg(t.kg)} kg`, `범위 ${fmtKg(t.kmin)}–${fmtKg(t.kmax)} kg${P.buoy_mode ? ' · 부표 모드' : ''}${R.o.wsrc === 'company' ? ' · ⚠ 기업 지수' : ''}`], ['마대', `${t.bags}장`, `부피 약 ${(t.vol * D.bag.bulk).toFixed(1)} m³`], [R.nTeams > 1 ? '가장 오래 걸리는 팀' : '총 작업 시간', fmtMin(R.totalMin), R.nTeams > 1 ? `${R.nTeams}팀 × ${R.o.workers}명 · 전 팀 합 ${fmtMin(R.totalWalk + R.totalWork)}` : `이동 ${fmtMin(R.totalWalk)} + 작업 ${fmtMin(R.totalWork)}`], ['일정', `${Math.max(0, ...R.teams.map(x => x.days))}일`, `팀당 ${R.o.workers}명 · 하루 ${R.o.hours}시간`], ['이동 거리', `${(R.routeLen / 1000).toFixed(1)} km`, (T ? '지형 최단경로(걷기 환산)' : '직선×우회') + (t.returns ? ` · 복귀 ${t.returns}회` : '') + (R.nTeams > 1 ? ' · 전 팀 합' : '')], ['2인 운반 물체', `${t.heavy}개`, '23 kg 넘을 수 있음'], ['진행', `${t.all ? Math.round(t.done / t.all * 100) : 0} %`, `완료 ${t.done}/${t.all}개 · ${fmtKg(t.doneKg)} kg`]];
                    $('#tiles').innerHTML = tiles.map( ([l,v,s]) => `<div class="tile"><div class="lab">${l}</div><div class="val">${v}</div><div class="sub">${s}</div></div>`).join('');
                    $('#daybtns').innerHTML = '<button onclick="fitAll()">전체 보기</button>' + R.days.map(d => `<button onclick="showDay(${d.team},${d.day})" style="border-color:${R.nTeams > 1 ? TEAMC[(d.team - 1) % TEAMC.length] : DAYC[(d.day - 1) % DAYC.length]}">${R.nTeams > 1 ? teamName(d.team) + '팀 ' : ''}${d.day}일차</button>`).join('') + '<button onclick="locateMe()">📍 내 위치</button>';
                    $('#legend').innerHTML = Object.entries(R.byCode).sort( (a, b) => b[1].count - a[1].count).map( ([c,d]) => `<span><i class="dot" style="background:${d.color}"></i>${esc(d.ko)} ${d.count}개</span>`).join('') + '<span><i class="dot" style="background:#b9f0d8;border-color:#1baf7a"></i>완료</span><span><i class="dot" style="background:#bbb;border-color:#777"></i>제외</span><span><i class="dia"></i>2인 운반(무거움)</span><span>★ 출발·집결지 (끌어서 변경)</span><span>점선 = 구역 접근 / 복귀</span>' + (R.nTeams > 1 ? R.teams.map(x => `<span><i style="display:inline-block;width:22px;height:4px;background:${TEAMC[(x.team - 1) % TEAMC.length]};vertical-align:middle;margin-right:4px"></i>${teamName(x.team)}팀 경로</span>`).join('') : R.days.map(d => `<span><i style="display:inline-block;width:22px;height:4px;background:${DAYC[(d.day - 1) % DAYC.length]};vertical-align:middle;margin-right:4px"></i>${d.day}일차 경로</span>`).join(''));
                    $('#status').textContent = `${Math.round(R.ms)} ms 에 다시 계산 · ${R.mode === 'boat' ? '보트 지원' : '도보'} · ${R.o.carry === 'carry' ? '들고 이동' : '현장 적치'} · ${R.o.objective === 'weight' ? '무게 우선' : '최단 이동'}${R.nTeams > 1 ? ` · ${R.nTeams}팀` : ''}${R.unreachable ? ` · 경로 없음 ${R.unreachable}쌍(직선 대체)` : ''}${R.o.wsrc === 'company' ? ' · ⚠ 기업 지수 기준: 실측 무게가 아니라 계획에 쓰면 안 됩니다 (합계 약 1.3 kg)' : ''}`;
                    $('#status').style.color = R.o.wsrc === 'company' ? '#c0392b' : '';
                }
                function renderSteps(R) {
                    let h = '';
                    for (const tm of R.teams) {
                        if (R.nTeams > 1) {
                            const col = TEAMC[(tm.team - 1) % TEAMC.length];
                            h += `<div class="team"><span class="pill" style="background:${col}">${teamName(tm.team)}팀</span><h3>${tm.zones.length}개 구역 · ${tm.n}개 · ${fmtKg(tm.kg)} kg · 마대 ${tm.bags}장 · ${fmtMin(tm.minutes)} · ${tm.days}일</h3></div>`;
                        }
                        for (const d of R.days.filter(d => d.team === tm.team)) {
                            const col = R.nTeams > 1 ? TEAMC[(d.team - 1) % TEAMC.length] : DAYC[(d.day - 1) % DAYC.length];
                            h += `<div class="day"><span class="pill" style="background:${col}">${d.day}일차</span><h3>${d.steps.length}개 구역 · 약 ${fmtMin(d.minutes)}</h3></div><div class="steps">`;
                            for (const z of R.zones.filter(z => z.team === tm.team && z.day === d.day)) {
                                const chips = Object.entries(z.byCode).sort( (a, b) => b[1] - a[1]).map( ([c,n]) => `<span class="chip" style="background:${D.mats[c] ? D.mats[c].color : '#ccc'}">${esc(D.mats[c] ? D.mats[c].ko : c)} ${n}</span>`).join('');
                                const warn = z.notes.map(n => `<div class="warn">⚠ ${esc(n)}</div>`).join('');
                                const tools = z.tools.length ? ` · 도구: ${esc(z.tools.join(', '))}` : '';
                                const phs = z.objects.map( (ob, i) => ob.img ? `<div class="ph"><img src="${ob.img}" alt="" title="${esc(ob.id)}"><small>${z.step}-${i + 1}</small></div>` : '').join('');
                                h += `<div class="step" data-step="${z.step}" style="border-left-color:${col}" onclick="highlight(${z.step})"><div class="head"><div class="stepnum" style="background:${col}">${z.step}</div><div style="flex:1;min-width:0"><div class="name">${esc(z.name)}</div><div class="meta">이전 지점에서 ${Math.round(z.dIn)} m · 이 구역 이동 ${Math.round(z.walkMin)}분${tools}</div></div><button class="donebtn" onclick="event.stopPropagation();doneZone(${z.step})">완료 ✓</button><button class="tidebtn${z.tide ? ' on' : ''}" title="만조 시 접근이 어려운 구역 표시 (물때표 확인)" onclick="event.stopPropagation();toggleTide(${z.step})">🌊${z.tide ? ' 만조 주의' : ''}</button></div><div class="chips">${chips}</div><div class="facts"><div class="fact"><b>${z.n}</b><span>개</span></div><div class="fact"><b>${fmtKg(z.kg)}</b><span>kg (${fmtKg(z.kmin)}–${fmtKg(z.kmax)})</span></div><div class="fact"><b>${z.bags}</b><span>마대</span></div><div class="fact"><b>${Math.round(z.workMin)}</b><span>분 작업</span></div></div>${warn}<div class="photos">${phs}</div><div class="links" onclick="event.stopPropagation()">${navLinks(z.name, z.lat, z.lon)}</div></div>`;
                            }
                            h += '</div>';
                        }
                    }
                    if (!R.zones.length)
                        h = '<p class="sub">' + (R.totals.all && R.totals.done >= R.totals.all ? '선택한 쓰레기를 모두 수거했습니다 🎉' : '선택한 조건에 맞는 쓰레기가 없습니다. 종류·최소 무게를 확인하세요.') + '</p>';
                    $('#steps').innerHTML = h;
                    $('#equip').innerHTML = R.equipment.map(e => `<li>${esc(e)}</li>`).join('');
                    $('#bycode').innerHTML = Object.entries(R.byCode).sort( (a, b) => b[1].kg - a[1].kg).map( ([c,d]) => {
                        const m = D.mats[c] || {};
                        return `<tr><td><i class="dot" style="background:${d.color}"></i>${esc(d.ko)}</td><td class="num">${d.count}</td><td class="num">${d.area.toFixed(1)}</td><td class="num"><b>${fmtKg(d.kg)}</b></td><td class="num">${fmtKg(d.kmin)}–${fmtKg(d.kmax)}</td><td class="num">${d.ckg.toFixed(3)}</td><td>${esc(m.handling || '-')}${m.tool ? ' · 도구: ' + esc(m.tool) : ''}</td></tr>`;
                    }
                    ).join('');
                    const stepOf = {};
                    R.zones.forEach(z => z.objects.forEach(ob => stepOf[ob.id] = z.step));
                    const rows = R.objs.slice().sort( (a, b) => (a.inc ? 0 : 1) - (b.inc ? 0 : 1) || (stepOf[a.id] || 0) - (stepOf[b.id] || 0) || a.order - b.order);
                    $('#objtab').innerHTML = rows.map(ob => `<tr class="${ob.inc ? '' : 'off'}"><td>${ob.done ? '✅' : (ob.inc ? `${stepOf[ob.id]}-${ob.order}` : '제외')}</td><td>${esc(ob.id)}</td><td><i class="dot" style="background:${ob.color}"></i>${esc(ob.ko)}</td><td class="num">${ob.w.toFixed(1)}×${ob.h.toFixed(1)}</td><td class="num">${ob.area.toFixed(2)}</td><td class="num"><b>${fmtKg(ob.kg)}</b></td><td class="num">${fmtKg(ob.kmin)}–${fmtKg(ob.kmax)}</td><td class="num">${ob.ckg != null ? ob.ckg : '-'}</td><td>${[ob.heavy ? '⚠ 2인 운반' : '', ob.buoys ? `부표 추정 ${ob.buoys}개` : '', ob.dup ? `중복 의심 (${esc(ob.dup.id)} · ${ob.dup.d.toFixed(1)} m)` : '', ob.tide ? '🌊 만조 주의' : ''].filter(Boolean).join(' · ')}</td></tr>`).join('');
                    // 진행 현황
                    const t = R.totals;
                    const pct = t.all ? Math.round(t.done / t.all * 100) : 0;
                    $('#progress').innerHTML = `<div class="bar"><div style="width:${pct}%"></div></div><p class="sub">완료 ${t.done}개 / 선택 ${t.all}개 (${pct} %) · 완료 무게 ${fmtKg(t.doneKg)} kg · 남은 ${R.inc.length}개 ${fmtKg(t.kg)} kg · 남은 시간 ${fmtMin(R.totalMin)}</p>` + (t.done ? `<div class="btns"><button onclick="resetDone()">완료 전부 취소</button></div>` : '');
                    // 실측 보정: 구역 선택 목록
                    const sel = $('#cal-zone');
                    if (sel) {
                        const curv = sel.value;
                        sel.innerHTML = R.zones.map(z => `<option value="${z.step}">${z.step}. ${esc(z.name)} (예상 ${fmtKg(z.kg)} kg)</option>`).join('');
                        if ([...sel.options].some(o => o.value === curv))
                            sel.value = curv;
                    }
                    $('#assume-params').textContent = `구역 묶기 ${P.link_m} m · 걷기 ${P.walk_kmh} km/h · 물체당 ${P.item_min}분 + ${P.min_per_m2}분/m² · 마대 ${P.bag_kg} kg / ${P.bag_l} L · 1인 운반 ${P.carry_kg_per_person} kg·${P.carry_bags_per_person}마대 · 숲 통과 ×${P.veg_cost} · 보트 ×${P.boat_cost}${T ? '' : ' · 우회 ×' + P.detour}${P.buoy_mode ? ` · 부표 모드 (≥${P.buoy_area_min} m², 1개 ${P.buoy_fp} m², 개당 ${P.buoy_kg.join('/')} kg)` : ''}${P.dup_m ? ` · 중복 의심 ${P.dup_m} m` : ''}` + (Object.values(ST.calib).some(v => v !== 1) ? ` · 실측 보정 ${Object.entries(ST.calib).filter( ([c,v]) => v !== 1).map( ([c,v]) => `${D.mats[c] ? D.mats[c].ko : c} ×${v.toFixed(2)}`).join(', ')}` : '') + ' (가정값)';
                }
                let CUR = null
                  , timer = null;
                function recompute() {
                    const busy = $('#busy');
                    busy.style.display = 'flex';
                    setTimeout( () => {
                        try {
                            readControls();
                            CUR = computePlan(opts());
                            renderTiles(CUR);
                            renderMap(CUR);
                            renderSteps(CUR);
                            if (typeof maybePush === 'function')
                                maybePush(false);
                        } catch (e) {
                            console.error(e);
                            $('#status').textContent = '계산 오류: ' + e.message;
                        }
                        busy.style.display = 'none';
                    }
                    , 10);
                }
                window.recompute = recompute;
                window.scheduleRecompute = () => {
                    clearTimeout(timer);
                    timer = setTimeout(recompute, 150);
                }
                ;
                window.resetAll = () => {
                    for (const [id,v] of Object.entries(D.control_defaults)) {
                        const el = document.getElementById(id);
                        if (!el)
                            continue;
                        if (el.type === 'checkbox')
                            el.checked = v;
                        else
                            el.value = v;
                    }
                    document.querySelectorAll('.codes input').forEach(e => e.checked = true);
                    for (const [n,v] of Object.entries(D.seg_defaults))
                        setSeg(n, v);
                    depot = Object.assign({}, defaultDepot);
                    ST.depotAt = Date.now();
                    persist();
                    recompute();
                    if (map)
                        fitAll();
                }
                ;
                // 진행 체크
                window.toggleDone = (ids) => {
                    markEv('done', ids.filter(id => !ST.done.has(id)));
                    markEv('undone', ids.filter(id => ST.done.has(id)));
                    persist();
                    recompute();
                }
                ;
                window.toggleTide = (step) => {
                    if (!CUR)
                        return;
                    const z = CUR.zones.find(z => z.step === step);
                    if (!z)
                        return;
                    markEv(z.tide ? 'untide' : 'tide', z.objects.map(ob => ob.id));
                    persist();
                    recompute();
                }
                ;
                window.doneZone = (step) => {
                    if (!CUR)
                        return;
                    const z = CUR.zones.find(z => z.step === step);
                    if (!z)
                        return;
                    markEv('done', z.objects.map(ob => ob.id));
                    persist();
                    recompute();
                }
                ;
                window.resetDone = () => {
                    markEv('undone', [...ST.done]);
                    persist();
                    recompute();
                }
                ;
                // 내 위치
                window.locateMe = () => {
                    if (!navigator.geolocation) {
                        $('#status').textContent = '이 기기에서는 위치를 쓸 수 없습니다';
                        return;
                    }
                    navigator.geolocation.getCurrentPosition(pos => {
                        const lat = pos.coords.latitude
                          , lon = pos.coords.longitude;
                        if (map) {
                            if (!meMarker) {
                                meMarker = L.circleMarker([lat, lon], {
                                    radius: 9,
                                    color: '#fff',
                                    weight: 3,
                                    fillColor: '#e34948',
                                    fillOpacity: 1
                                }).addTo(map);
                                meMarker.bindTooltip('내 위치');
                            } else
                                meMarker.setLatLng([lat, lon]);
                        }
                        if (!CUR || !CUR.zones.length) {
                            $('#status').textContent = '내 위치 표시';
                            if (map)
                                map.flyTo([lat, lon], 17);
                            return;
                        }
                        const xy = ll2xy(lon, lat);
                        let best = null;
                        if (T) {
                            forcePassable(xy);
                            const {dist} = dijkstra(CUR.mode, cellOf(xy[0], xy[1]));
                            for (const z of CUR.zones) {
                                const d = z.firstCell != null ? dist[z.firstCell] : Infinity;
                                if (isFinite(d) && (best === null || d < best.d))
                                    best = {
                                        z,
                                        d
                                    };
                            }
                        }
                        if (!best) {
                            for (const z of CUR.zones) {
                                const d = Math.hypot(z.cx - xy[0], z.cy - xy[1]) * P.detour;
                                if (best === null || d < best.d)
                                    best = {
                                        z,
                                        d
                                    };
                            }
                        }
                        const min = best.d / 1000 / P.walk_kmh * 60;
                        $('#status').textContent = `내 위치에서 가장 가까운 구역: ${best.z.step}. ${best.z.name} · ${Math.round(best.d)} m · 약 ${Math.round(min)}분`;
                        if (map) {
                            map.flyToBounds(L.latLngBounds([[lat, lon], [best.z.lat, best.z.lon]]).pad(0.3));
                        }
                        highlight(best.z.step, false);
                    }
                    , err => {
                        $('#status').textContent = '위치를 가져오지 못했습니다 (' + err.message + '). HTTPS 주소에서, 위치 권한을 허용해야 합니다';
                    }
                    , {
                        enableHighAccuracy: true,
                        timeout: 10000
                    });
                }
                ;
                // 실측 보정
                window.applyMeasured = () => {
                    if (!CUR)
                        return;
                    const step = parseInt($('#cal-zone').value, 10);
                    const kg = parseFloat($('#cal-kg').value);
                    const z = CUR.zones.find(z => z.step === step);
                    if (!z || !isFinite(kg) || kg <= 0) {
                        $('#cal-msg').textContent = '구역과 실측 무게(kg)를 입력하세요';
                        return;
                    }
                    const base = z.objects.reduce( (s, ob) => s + ob.kg / ((ST.calib[ob.code] || 1)), 0);
                    if (base <= 0) {
                        $('#cal-msg').textContent = '이 구역의 예상 무게가 0 입니다';
                        return;
                    }
                    const f = kg / base;
                    const codes = [...new Set(z.objects.map(ob => ob.code))];
                    for (const c of codes) {
                        ST.calib[c] = Math.round(f * 100) / 100;
                        ST.calibAt[c] = Date.now();
                        const el = $('#c-cal-' + c);
                        if (el)
                            el.value = ST.calib[c];
                    }
                    persist();
                    $('#cal-msg').textContent = `구역 ${step} 예상 ${fmtKg(base)} kg → 실측 ${fmtKg(kg)} kg : ${codes.map(c => D.mats[c] ? D.mats[c].ko : c).join('·')} 보정 계수 ×${f.toFixed(2)} 적용`;
                    recompute();
                }
                ;
                window.resetCalib = () => {
                    for (const c of Object.keys(ST.calib)) {
                        ST.calib[c] = 1;
                        ST.calibAt[c] = Date.now();
                        const el = $('#c-cal-' + c);
                        if (el)
                            el.value = 1;
                    }
                    persist();
                    $('#cal-msg').textContent = '보정 계수를 1로 되돌렸습니다';
                    recompute();
                }
                ;
                // 시나리오 비교 + 민감도
                window.compareScenarios = () => {
                    readControls();
                    const cur = opts();
                    const rows = [['현재 설정', cur]];
                    if (T)
                        rows.push(['도보 · 현장 적치', opts({
                            travel: 'walk',
                            carry: 'pile'
                        })], ['보트 지원 · 현장 적치', opts({
                            travel: 'boat',
                            carry: 'pile'
                        })]);
                    rows.push([`${cur.workers * 2}명 (인원 2배)`, opts({
                        workers: cur.workers * 2
                    })], ['팀 2개', opts({
                        teams: 2
                    })], ['들고 이동', opts({
                        carry: 'carry'
                    })], ['무게 우선 순서', opts({
                        objective: 'weight'
                    })], ['무게 추정 최소값', opts({
                        wsrc: 'ours',
                        wstat: 'min'
                    })], ['무게 추정 최대값', opts({
                        wsrc: 'ours',
                        wstat: 'max'
                    })], ['무게 ×0.5 (민감도)', opts({
                        scale: 0.5
                    })], ['무게 ×1.5 (민감도)', opts({
                        scale: 1.5
                    })]);
                    const out = rows.map( ([name,o]) => {
                        const R = computePlan(o);
                        return `<tr><td>${esc(name)}</td><td class="num">${R.totals.zones}</td><td class="num">${fmtKg(R.totals.kg)}</td><td class="num">${R.totals.bags}</td><td class="num">${(R.routeLen / 1000).toFixed(1)}</td><td class="num">${fmtMin(R.totalMin)}</td><td class="num">${Math.max(0, ...R.teams.map(x => x.days))}</td><td class="num">${R.totals.heavy}</td></tr>`;
                    }
                    ).join('');
                    $('#scen').innerHTML = `<table><thead><tr><th>시나리오</th><th class="num">구역</th><th class="num">예상 kg</th><th class="num">마대</th><th class="num">이동 km</th><th class="num">총 시간</th><th class="num">일수</th><th class="num">2인 운반</th></tr></thead><tbody>${out}</tbody></table><p class="sub">같은 완료·종류·고급 설정에서 조건 하나씩만 바꾼 결과. 민감도 행은 모든 무게에 배수를 곱해 가정값(채움률·두께·밀도)의 영향을 본 것.</p>`;
                }
                ;
                window.downloadCsv = () => {
                    if (!CUR)
                        return;
                    const rows = [['순서', '팀', '일차', '구역', '개수', '구성', '계획 무게(kg)', '최소(kg)', '최대(kg)', '기업 지수(참고)', '부피(m³)', '마대', '접근(m)', '이동(분)', '작업(분)', '누적(분)', '복귀', '주의']];
                    for (const z of CUR.zones)
                        rows.push([z.step, teamName(z.team), z.day, z.name, z.n, Object.entries(z.byCode).map( ([c,n]) => `${D.mats[c] ? D.mats[c].ko : c} ${n}`).join(' '), z.kg.toFixed(2), z.kmin.toFixed(2), z.kmax.toFixed(2), z.ckg.toFixed(3), z.m3.toFixed(3), z.bags, Math.round(z.dIn), z.walkMin.toFixed(1), z.workMin.toFixed(1), z.cumMin.toFixed(1), z.returns, z.notes.join(' / ')]);
                    const csv = '﻿' + rows.map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(',')).join('\n');
                    const a = document.createElement('a');
                    a.href = URL.createObjectURL(new Blob([csv],{
                        type: 'text/csv;charset=utf-8'
                    }));
                    a.download = '수거계획_작업순서.csv';
                    a.click();
                }
                ;
                window.downloadJson = () => {
                    if (!CUR)
                        return;
                    const out = {
                        params: P,
                        depot,
                        options: CUR.o,
                        calib: ST.calib,
                        done: [...ST.done],
                        totals: CUR.totals,
                        teams: CUR.teams.map(t => ({
                            team: t.team,
                            zones: t.zones,
                            n: t.n,
                            kg: t.kg,
                            bags: t.bags,
                            minutes: t.minutes,
                            days: t.days
                        })),
                        days: CUR.days,
                        zones: CUR.zones.map(z => ({
                            step: z.step,
                            team: z.team,
                            day: z.day,
                            name: z.name,
                            n: z.n,
                            kg: z.kg,
                            kmin: z.kmin,
                            kmax: z.kmax,
                            bags: z.bags,
                            dIn: z.dIn,
                            distTot: z.distTot,
                            walkMin: z.walkMin,
                            workMin: z.workMin,
                            returns: z.returns,
                            objects: z.objects.map(ob => ob.id),
                            notes: z.notes
                        }))
                    };
                    const a = document.createElement('a');
                    a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)],{
                        type: 'application/json'
                    }));
                    a.download = '수거계획_설정결과.json';
                    a.click();
                }
                ;
                document.querySelectorAll('.ctrl input, .ctrl select').forEach(el => el.addEventListener('input', scheduleRecompute));
                document.querySelectorAll('.seg button').forEach(b => b.addEventListener('click', () => {
                    setSeg(b.parentElement.dataset.name, b.dataset.v);
                    recompute();
                }
                ));
                for (const [c,v] of Object.entries(ST.calib)) {
                    const el = $('#c-cal-' + c);
                    if (el)
                        el.value = v;
                }
                window.toggleSide = () => {
                    const l = $('#layout');
                    l.classList.toggle('collapsed');
                    try {
                        localStorage.setItem(LS_KEY + ':side', l.classList.contains('collapsed') ? '1' : '0');
                    } catch (e) {}
                    setTimeout( () => {
                        if (map) {
                            map.invalidateSize();
                        }
                    }
                    , 250);
                }
                ;
                try {
                    if (localStorage.getItem(LS_KEY + ':side') === '1')
                        $('#layout').classList.add('collapsed');
                } catch (e) {}

                // ───────── 공유 동기화 (claude.ai 공개 링크: db 캐퍼빌리티) ─────────
                // 완료 체크·실측 보정 계수·출발지를 모든 접속자가 같이 본다. 다른 사람이 바꾸면 onSnapshot 으로 바로 반영.
                let SHDOC = null
                  , SH_READONLY = false
                  , SH_LAST = ''
                  , SH_TIMER = null
                  , SH_WRITING = null
                  , SH_USER = null
                  , SH_NAMES = {};
                const sharedState = () => ({
                    done: [...ST.done].sort(),
                    tide: [...ST.tide].sort(),
                    calib: ST.calib,
                    calibAt: ST.calibAt,
                    depotAt: ST.depotAt,
                    ev: ST.ev,
                    depot: {
                        lon: depot.lon,
                        lat: depot.lat,
                        name: depot.name
                    }
                });
                const sharedSig = (o) => JSON.stringify([o.done, o.tide || [], o.calib, o.depot && [+o.depot.lon.toFixed(6), +o.depot.lat.toFixed(6)]]);
                // 원격 상태를 덮어쓰지 않고 합친다: 완료·만조는 물체별 최신 시각, 보정 계수는 재질별 최신 시각, 출발지는 최신 시각.
                // 예전 형식(시각 없음)은 그 저장 시각(updatedAt) 으로 취급한다. 반환값: 내 상태가 바뀌었는지
                function mergeRemote(d) {
                    const rts = d.updatedAt || Date.now();
                    let changed = false;
                    if (d.ev) {
                        for (const k of EV_KEYS)
                            for (const [id,ts] of Object.entries(d.ev[k] || {}))
                                if ((ST.ev[k][id] || 0) < ts) {
                                    ST.ev[k][id] = ts;
                                    changed = true;
                                }
                    } else {
                        for (const id of d.done || [])
                            if ((ST.ev.done[id] || 0) < rts && (ST.ev.undone[id] || 0) < rts) {
                                ST.ev.done[id] = rts;
                                changed = true;
                            }
                    }
                    deriveSets();
                    const rca = d.calibAt || null;
                    for (const [c,v] of Object.entries(d.calib || {})) {
                        const rt = rca ? (rca[c] || 0) : rts;
                        if ((ST.calibAt[c] || 0) < rt && (ST.calib[c] || 1) !== v) {
                            ST.calib[c] = v;
                            ST.calibAt[c] = rt;
                            changed = true;
                        }
                    }
                    for (const c of Object.keys(D.mats)) {
                        const el = $('#c-cal-' + c);
                        if (el)
                            el.value = ST.calib[c] || 1;
                    }
                    const rda = d.depotAt != null ? d.depotAt : rts;
                    if (d.depot && isFinite(d.depot.lon) && isFinite(d.depot.lat) && (ST.depotAt || 0) < rda && (Math.abs(d.depot.lon - depot.lon) > 1e-7 || Math.abs(d.depot.lat - depot.lat) > 1e-7)) {
                        depot = {
                            lon: d.depot.lon,
                            lat: d.depot.lat,
                            name: d.depot.name || '지정 출발지'
                        };
                        ST.depotAt = rda;
                        if (depotMarker)
                            depotMarker.setLatLng([depot.lat, depot.lon]);
                        changed = true;
                    }
                    persist();
                    return changed;
                }
                function syncStatus(msg, ok) {
                    const el = $('#sync');
                    if (!el)
                        return;
                    el.textContent = msg;
                    el.style.color = ok === false ? '#c0392b' : '';
                }
                // Supabase 백엔드 (GitHub Pages 등 일반 호스팅): 테이블 shared_state(site, data jsonb, updated_at, updated_by)
                let SB = null
                  , SB_ROW = null
                  , SB_POLL = null;
                const myName = () => {
                    try {
                        return localStorage.getItem('shoresweep:name') || '';
                    } catch (e) {
                        return '';
                    }
                }
                ;
                window.setMyName = (v) => {
                    try {
                        localStorage.setItem('shoresweep:name', v || '');
                    } catch (e) {}
                }
                ;
                function applyRemoteBody(d, meta) {
                    const sig = sharedSig({
                        done: d.done || [],
                        tide: d.tide || [],
                        calib: d.calib || {},
                        depot: d.depot || depot
                    });
                    const when = d.updatedAt ? new Date(d.updatedAt).toLocaleTimeString('ko', {
                        hour: '2-digit',
                        minute: '2-digit'
                    }) : '';
                    syncStatus(`공유 동기화 켜짐 (${meta})${when ? ' · 마지막 변경 ' + when : ''}${d.byName ? ' · ' + d.byName : ''}`);
                    if (sig === SH_LAST)
                        return;
                    SH_LAST = sig;
                    mergeRemote(d);
                    // recompute → maybePush: 합친 결과가 원격과 다르면 그 결과를 올려 모두에게 전파
                    recompute();
                }
                async function initSupabase() {
                    const cfg = D.supabase;
                    if (!cfg)
                        return false;
                    if (typeof supabase === 'undefined' || !supabase.createClient) {
                        syncStatus('Supabase 라이브러리를 불러오지 못했습니다 (인터넷 확인) · 이 브라우저에만 저장', false);
                        return true;
                    }
                    try {
                        SB = supabase.createClient(cfg.url, cfg.key);
                    } catch (e) {
                        syncStatus('Supabase 연결 실패: ' + e.message, false);
                        return true;
                    }
                    const site = D.site || 'site';
                    syncStatus('Supabase 연결 중…');
                    const load = async () => {
                        const {data, error} = await SB.from(cfg.table).select('data,updated_at,updated_by').eq('site', site).maybeSingle();
                        if (error) {
                            syncStatus('Supabase 읽기 실패: ' + error.message + ' (supabase_setup.sql 실행·RLS 정책 확인)', false);
                            return;
                        }
                        if (!data) {
                            syncStatus('공유 동기화 켜짐 (Supabase) · 아직 저장된 변경 없음');
                            maybePush(true);
                            return;
                        }
                        const body = Object.assign({}, data.data || {}, {
                            updatedAt: data.updated_at ? Date.parse(data.updated_at) : null,
                            byName: data.updated_by || ''
                        });
                        applyRemoteBody(body, 'Supabase');
                    }
                    ;
                    await load();
                    try {
                        SB.channel('shared_state_' + site).on('postgres_changes', {
                            event: '*',
                            schema: 'public',
                            table: cfg.table,
                            filter: 'site=eq.' + site
                        }, payload => {
                            const r = payload.new;
                            if (!r || !r.data)
                                return;
                            applyRemoteBody(Object.assign({}, r.data, {
                                updatedAt: r.updated_at ? Date.parse(r.updated_at) : null,
                                byName: r.updated_by || ''
                            }), 'Supabase 실시간');
                        }
                        ).subscribe();
                    } catch (e) {}
                    SB_POLL = setInterval(load, 15000);
                    // 실시간이 꺼져 있어도 15 초마다 맞춤
                    SHDOC = {
                        set: async (body) => {
                            const {error} = await SB.from(cfg.table).upsert({
                                site,
                                data: {
                                    done: body.done,
                                    tide: body.tide,
                                    calib: body.calib,
                                    calibAt: body.calibAt,
                                    depotAt: body.depotAt,
                                    ev: body.ev,
                                    depot: body.depot
                                },
                                updated_at: new Date().toISOString(),
                                updated_by: myName() || '이름 없음'
                            }, {
                                onConflict: 'site'
                            });
                            if (error) {
                                const err = new Error(error.message);
                                err.code = /permission|policy|row-level/i.test(error.message) ? 'invalid_argument' : 'unavailable';
                                throw err;
                            }
                        }
                    };
                    return true;
                }
                async function initShared() {
                    if (D.supabase) {
                        await initSupabase();
                        return;
                    }
                    if (!D.shared || !window.claude || typeof window.claude.use !== 'function') {
                        return;
                    }
                    syncStatus('공유 저장소 연결 중…');
                    let db = null
                      , user = null;
                    try {
                        [db,user] = await Promise.all([window.claude.use('db'), window.claude.use('user')]);
                    } catch (e) {
                        db = null;
                    }
                    if (!db) {
                        syncStatus('공유 저장 사용 불가 (로그인 필요) · 이 브라우저에만 저장', false);
                        return;
                    }
                    SH_USER = user;
                    try {
                        if (user && user.can) {
                            const w = await user.can('data.write');
                            if (w === false) {
                                SH_READONLY = true;
                            }
                        }
                    } catch (e) {}
                    SHDOC = db.doc('state/shared');
                    SHDOC.onSnapshot(snap => {
                        if (!snap.exists) {
                            syncStatus('공유 동기화 켜짐 (아직 저장된 변경 없음)' + (SH_READONLY ? ' · 읽기 전용' : ''));
                            if (!SH_READONLY)
                                maybePush(true);
                            return;
                        }
                        const d = snap.data();
                        const sig = sharedSig({
                            done: d.done || [],
                            tide: d.tide || [],
                            calib: d.calib || {},
                            depot: d.depot || depot
                        });
                        const when = d.updatedAt ? new Date(d.updatedAt).toLocaleTimeString('ko', {
                            hour: '2-digit',
                            minute: '2-digit'
                        }) : '';
                        const byName = d.by ? (SH_NAMES[d.by] || '다른 사용자') : '';
                        syncStatus(`공유 동기화 켜짐${when ? ' · 마지막 변경 ' + when : ''}${byName ? ' · ' + byName : ''}${SH_READONLY ? ' · 읽기 전용(변경은 이 브라우저에만)' : ''}${snap.metadata && snap.metadata.hasPendingWrites ? ' · 저장 중' : ''}`);
                        if (d.by && SH_USER && SH_USER.profiles && !SH_NAMES[d.by]) {
                            SH_USER.profiles([d.by]).then(ps => {
                                const nm = (ps && ps[d.by] && ps[d.by].name) || '';
                                if (nm) {
                                    SH_NAMES[d.by] = nm;
                                    syncStatus(`공유 동기화 켜짐${when ? ' · 마지막 변경 ' + when : ''} · ${nm}${SH_READONLY ? ' · 읽기 전용' : ''}`);
                                }
                            }
                            ).catch( () => {}
                            );
                        }
                        if (sig === SH_LAST)
                            return;
                        // 내가 보낸 것 또는 이미 반영된 것
                        SH_LAST = sig;
                        mergeRemote(d);
                        recompute();
                    }
                    , err => {
                        syncStatus('공유 동기화 중단 (' + (err && err.code || '오류') + ') · 이 브라우저에만 저장', false);
                        SHDOC = null;
                    }
                    );
                }
                function maybePush(force) {
                    if (!SHDOC || SH_READONLY)
                        return;
                    const st = sharedState();
                    const sig = sharedSig(st);
                    if (!force && sig === SH_LAST)
                        return;
                    SH_LAST = sig;
                    clearTimeout(SH_TIMER);
                    SH_TIMER = setTimeout(async () => {
                        const body = Object.assign({}, st, {
                            updatedAt: Date.now(),
                            by: null
                        });
                        try {
                            if (SH_USER && SH_USER.id)
                                body.by = await SH_USER.id();
                        } catch (e) {}
                        const run = async () => {
                            try {
                                await SHDOC.set(body);
                            } catch (e) {
                                const code = e && e.code;
                                if (code === 'invalid_argument') {
                                    SH_READONLY = true;
                                    syncStatus('읽기 전용: 공유 변경 권한이 없어 이 브라우저에만 저장됩니다 (공유 메뉴에서 Contributor 이상으로 초대 필요)', false);
                                } else if (code === 'unavailable') {
                                    setTimeout( () => {
                                        SHDOC && SHDOC.set(body).catch( () => {}
                                        );
                                    }
                                    , 800 + Math.random() * 700);
                                } else {
                                    syncStatus('공유 저장 실패: ' + (code || e), false);
                                }
                            }
                        }
                        ;
                        SH_WRITING = (SH_WRITING || Promise.resolve()).then(run, run);
                    }
                    , 400);
                }
                initShared();
                if (D.supabase) {
                    const ni = $('#c-name');
                    if (ni) {
                        ni.style.display = 'block';
                        ni.value = myName();
                    }
                }
                initMap();
                recompute();
            }
            )();