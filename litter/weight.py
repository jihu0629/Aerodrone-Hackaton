"""
무게 추정 — 물리 사전값 + 업체 실측 무게로 학습한 보정 + 예측구간.

1) 물리 사전값 (데이터 없이도 동작)
     3D 부피를 믿을 수 있으면   V × ρ_app(클래스)
     아니면 (얇은 물체·DSM 없음)  면적 × 유효두께 × ρ_app   (Andriolo W3)
     로프는 길이 × kg/m 도 계산해서 큰 쪽
2) 학습 보정 (업체가 준 물체별 무게가 있으면)
     log(무게) ~ GBM(클래스, log면적, log길이, 길쭉함, log높이, log부피, log사전값)
     사전값이 입력에 들어가니 데이터가 적어도 물리 모델이 바닥을 받쳐준다.
3) 예측구간 (split-conformal, log 공간)
     교차검증 잔차의 분위수로 [하한, 상한]. 상한이 23 kg을 넘으면 2인 권장.
     → "평균적으로 몇 kg"이 아니라 "최악이면 몇 kg"으로 인력을 정한다.

`evaluate()`는 같은 데이터에서 방법별 오차를 비교하는 어블레이션 —
발표 핵심 슬라이드 (개수×14 g → 2D → +3D).
"""
import numpy as np

FEATS_2D = ["log_area", "log_length", "log_elong", "log_prior2d"]
FEATS_3D = ["log_h90", "log_vol", "log_prior3d"]


def prior(it, cfg):
    p = cfg["classes"].get(it["cls"], cfg["classes"]["other"])
    area = it.get("area_m2", 0.0)
    w2d = area * p["thick_m"] * p["rho_app"]
    if p.get("kg_per_m"):
        w2d = max(w2d, it.get("length_m", 0.0) * p["kg_per_m"])
    w3d = None
    if it.get("valid_3d"):
        w3d = it["volume_m3"] * p["rho_app"]
        if p.get("kg_per_m"):
            w3d = max(w3d, it.get("length_m", 0.0) * p["kg_per_m"])
    it["prior2d_kg"] = max(w2d, 1e-3)
    it["prior3d_kg"] = max(w3d, 1e-3) if w3d is not None else None
    it["prior_kg"] = it["prior3d_kg"] or it["prior2d_kg"]
    it["bulk_m3"] = it.get("volume_m3") if it.get("valid_3d") else area * p["thick_m"]
    return it


def _features(items, use3d):
    L = lambda v: np.log(max(float(v), 1e-4))
    rows = []
    for it in items:
        r = [L(it.get("area_m2", 0)), L(it.get("length_m", 0)), L(it.get("elongation", 1)), L(it["prior2d_kg"])]
        if use3d:
            ok = it.get("valid_3d")
            r += [L(it["h_p90_m"]) if ok else np.nan, L(it["volume_m3"]) if ok else np.nan,
                  L(it["prior3d_kg"]) if ok else np.nan]
        rows.append(r)
    return np.array(rows, float)


class WeightModel:
    def __init__(self, cfg, use3d=True):
        self.cfg, self.use3d = cfg, use3d
        self.classes = sorted(cfg["classes"])
        self.q = None

    def _X(self, items):
        X = _features(items, self.use3d)
        c = np.array([self.classes.index(it["cls"]) for it in items], float)[:, None]
        return np.hstack([c, X])

    def _new(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            categorical_features=[0], max_iter=300, learning_rate=0.05,
            max_leaf_nodes=15, min_samples_leaf=8, l2_regularization=1.0, random_state=0)

    def fit(self, items, n_folds=5):
        """학습 + 교차검증 잔차로 conformal 분위수 계산 (클래스별로 충분하면 클래스별)."""
        from sklearn.model_selection import KFold

        X, y = self._X(items), np.log([it["weight_kg"] for it in items])
        oof = np.zeros(len(y))
        for tr, te in KFold(n_folds, shuffle=True, random_state=0).split(X):
            oof[te] = self._new().fit(X[tr], y[tr]).predict(X[te])
        res = np.abs(y - oof)
        cov = self.cfg["weight"]["coverage"]
        self.q = {"_all": float(np.quantile(res, cov))}
        cls = np.array([it["cls"] for it in items])
        for c in set(cls):
            if (cls == c).sum() >= 30:
                self.q[c] = float(np.quantile(res[cls == c], cov))
        self.model = self._new().fit(X, y)
        return oof

    def predict(self, items):
        mu = np.exp(self.model.predict(self._X(items)))
        for it, m in zip(items, mu):
            q = self.q.get(it["cls"], self.q["_all"])
            it.update(weight_est_kg=float(m), weight_lo_kg=float(m * np.exp(-q)), weight_hi_kg=float(m * np.exp(q)),
                      weight_method="learned" + ("+3d" if self.use3d and it.get("valid_3d") else ""))
        return items


def estimate(items, cfg):
    """무게 정답이 있으면 학습, 없으면 물리 사전값 + 고정 배수 구간."""
    for it in items:
        prior(it, cfg)
    labeled = [it for it in items if it.get("weight_kg")]
    use3d = any(it.get("valid_3d") for it in items)
    if len(labeled) >= 30:
        wm = WeightModel(cfg, use3d=use3d)
        wm.fit(labeled)
        wm.predict(items)
        print(f"  무게 모델 학습: {len(labeled)}개 · 3D={use3d} · 90% 구간 배수 ×/÷{np.exp(wm.q['_all']):.2f}")
        return items, wm
    f = cfg["weight"]["no_data_factor"]
    for it in items:
        m = it["prior_kg"]
        it.update(weight_est_kg=m, weight_lo_kg=m / f, weight_hi_kg=m * f,
                  weight_method="physics" + ("+3d" if it.get("prior3d_kg") else ""))
    print(f"  무게 정답 {len(labeled)}개 (<30) → 물리 사전값만 사용, 구간 ×/÷{f}")
    return items, None


def evaluate(items, cfg, n_folds=5):
    """방법별 무게 오차 비교 (교차검증). 정답 무게가 있는 물체만."""
    from sklearn.model_selection import KFold

    items = [prior(dict(it), cfg) for it in items if it.get("weight_kg")]
    y = np.array([it["weight_kg"] for it in items])
    cls = np.array([it["cls"] for it in items])
    lift = cfg["plan"]["lift_limit_kg"]
    preds, his = {}, {}
    preds["개수×14g (Andriolo W2)"] = np.full(len(y), cfg["baseline_item_kg"])
    # 클래스 평균무게 (Andriolo W1) — 학습 fold의 클래스 중앙값
    w1 = np.zeros(len(y))
    folds = list(KFold(n_folds, shuffle=True, random_state=0).split(y))
    for tr, te in folds:
        med = {c: np.median(y[tr][cls[tr] == c]) for c in set(cls[tr])}
        w1[te] = [med.get(c, np.median(y[tr])) for c in cls[te]]
    preds["클래스 평균무게 (W1)"] = w1
    preds["물리 2D (면적×두께×ρ)"] = np.array([it["prior2d_kg"] for it in items])
    has3d = any(it.get("valid_3d") for it in items)
    if has3d:
        preds["물리 3D (부피×ρ)"] = np.array([it["prior_kg"] for it in items])
    for name, use3d in [("학습 2D", False)] + ([("학습 2D+3D (제안)", True)] if has3d else []):
        wm = WeightModel(cfg, use3d=use3d)
        p = np.zeros(len(y))
        hi = np.zeros(len(y))
        for tr, te in folds:
            wm.fit([items[i] for i in tr])
            sub = wm.predict([dict(items[i]) for i in te])
            p[te] = [s["weight_est_kg"] for s in sub]
            hi[te] = [s["weight_hi_kg"] for s in sub]
        preds[name], his[name] = p, hi
    heavy = y >= lift
    rows = []
    for name, p in preds.items():
        ape = np.abs(p - y) / y
        dec = his.get(name, p)  # 인력 판단에 쓰는 값: 학습 모델은 상한, 나머지는 점추정
        rows.append({
            "method": name,
            "total_err_pct": float((p.sum() - y.sum()) / y.sum() * 100),
            "median_ape_pct": float(np.median(ape) * 100),
            "within_50pct": float(np.mean(ape <= 0.5) * 100),
            "heavy_n": int(heavy.sum()),
            "heavy_recall_pct": float(np.mean(dec[heavy] >= lift) * 100) if heavy.any() else None,
            "false_2p_pct": float(np.mean(dec[~heavy] >= lift) * 100) if (~heavy).any() else None,
        })
    return rows
