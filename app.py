from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

app = Flask(__name__)

# =========================
# Config
# =========================
DEFAULT_TICKER = "TSLA"
LOOKBACK_DAYS = 420         # histórico para features/modelo
HORIZON_DAYS = 1            # "próximo día" (swing diario)
ML_TEST_RATIO = 0.2

BUY_PROBA = 0.55
SELL_PROBA = 0.45

# Lista “líquida” para Top movers (puedes ampliarla luego)
UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK.B","LLY","AVGO","JPM","V",
    "XOM","UNH","MA","COST","HD","PG","NFLX","AMD","ADBE","CRM","INTC","CSCO","ORCL",
    "WMT","KO","PEP","BAC","WFC","DIS","NKE","PFE","ABT","TMO","MCD","QCOM","TXN",
    "IBM","GE","CAT","BA","UPS","SBUX","UBER","PLTR","SNOW","SHOP","SQ","PYPL"
]

# =========================
# Data (Stooq)
# =========================
def fetch_daily_ohlcv_stooq(ticker: str) -> pd.DataFrame:
    """
    Stooq para US: usa sufijo .us (ej: tsla.us)
    Devuelve columnas: Date, Open, High, Low, Close, Volume
    """
    t = ticker.strip().lower()
    # Stooq usa brk-b.us en vez de BRK.B
    t = t.replace(".", "-")
    url = f"https://stooq.com/q/d/l/?s={t}.us&i=d"
    df = pd.read_csv(url)
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"No pude descargar datos para {ticker}. Revisa el ticker.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # filtra filas con Close vacío o cero
    df = df[df["Close"].notna()].copy()
    return df

# =========================
# Indicators / Features
# =========================
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def bollinger_z(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    z = (close - mid) / (std.replace(0, np.nan))
    return z

def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd = ema(close, fast) - ema(close, slow)
    sig = ema(macd, signal)
    return macd - sig

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["Close"]

    out["ret_1"] = c.pct_change(1)
    out["ret_2"] = c.pct_change(2)
    out["ret_5"] = c.pct_change(5)
    out["ret_10"] = c.pct_change(10)

    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_20"] = out["ret_1"].rolling(20).std()

    out["sma_20"] = sma(c, 20)
    out["sma_50"] = sma(c, 50)
    out["sma_ratio_20_50"] = out["sma_20"] / out["sma_50"] - 1

    out["rsi_14"] = rsi(c, 14)
    out["bb_z"] = bollinger_z(c, 20, 2.0)
    out["macd_hist"] = macd_hist(c)

    out["hl_range"] = (out["High"] - out["Low"]) / c.replace(0, np.nan)
    out["oc_range"] = (out["Close"] - out["Open"]) / out["Open"].replace(0, np.nan)

    out["volu_ratio_20"] = out["Volume"] / out["Volume"].rolling(20).mean()

    return out

def make_label(df: pd.DataFrame, horizon: int = 1, thresh: float = 0.0) -> pd.Series:
    future_ret = df["Close"].shift(-horizon) / df["Close"] - 1
    return (future_ret > thresh).astype(int)

FEATURE_COLS = [
    "ret_1","ret_2","ret_5","ret_10",
    "vol_10","vol_20",
    "sma_ratio_20_50",
    "rsi_14","bb_z","macd_hist",
    "hl_range","oc_range","volu_ratio_20"
]

# =========================
# Probabilities (Methods)
# =========================
@dataclass
class MethodResult:
    name: str
    proba_up: float
    proba_down: float
    signal: str
    detail: str

def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def sigmoid(x: float) -> float:
    # estable
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

def signal_from_proba(p: float) -> str:
    if p >= BUY_PROBA:
        return "BUY"
    if p <= SELL_PROBA:
        return "SELL / NO ENTRAR"
    return "NEUTRAL"

def method_bollinger(last_bb_z: float) -> Tuple[float, str]:
    # idea: z muy negativo → más probable rebote (subir)
    # z muy positivo → más probable corrección (bajar)
    # mapeo suave con sigmoid
    p_up = sigmoid(-0.9 * last_bb_z)  # z=-2 => p_up alta, z=+2 => p_up baja
    detail = f"bb_z={last_bb_z:.2f} (negativo=precio bajo vs media; positivo=alto)"
    return clip01(p_up), detail

def method_rsi(last_rsi: float) -> Tuple[float, str]:
    # RSI <30 sobrevendido → más prob subir; RSI >70 sobrecomprado → más prob bajar
    # mapeo lineal suave alrededor de 50
    p_up = clip01(0.5 + (50 - last_rsi) / 80)  # 30 -> ~0.75, 70 -> ~0.25 aprox
    detail = f"RSI14={last_rsi:.1f} (<30 sobrevendido, >70 sobrecomprado)"
    return p_up, detail

def method_macd(last_hist: float) -> Tuple[float, str]:
    # hist > 0 = momentum alcista; hist < 0 = bajista
    p_up = sigmoid(8.0 * last_hist)  # escala; hist suele ser pequeño
    detail = f"MACD_hist={last_hist:.4f} (positivo=momentum alcista)"
    return clip01(p_up), detail

def method_sma_crossover(sma_ratio_20_50: float) -> Tuple[float, str]:
    # >0: SMA20 > SMA50 (tendencia alcista); <0 tendencia bajista
    p_up = sigmoid(18.0 * sma_ratio_20_50)
    detail = f"SMA20/SMA50-1={sma_ratio_20_50:.4%} (positivo=tendencia alcista)"
    return clip01(p_up), detail

def method_momentum(ret_10: float) -> Tuple[float, str]:
    # momentum simple: retorno 10 días
    p_up = sigmoid(10.0 * ret_10)
    detail = f"ret_10={ret_10:.2%} (momentum 10 días)"
    return clip01(p_up), detail

def method_ml_classifier(df_feat: pd.DataFrame) -> Tuple[float, str, float | None]:
    """
    Entrena un modelo con split temporal y produce proba para el último día.
    Retorna: proba_up, detalle, auc (si se puede)
    """
    df = df_feat.copy()
    df["y"] = make_label(df, HORIZON_DAYS, thresh=0.0)
    df = df.dropna().copy()
    df = df.iloc[:-HORIZON_DAYS].copy()

    if len(df) < 350:
        raise ValueError("Muy pocos datos para entrenar el modelo ML con estabilidad.")

    X = df[FEATURE_COLS]
    y = df["y"].astype(int)

    n = len(df)
    test_size = int(n * ML_TEST_RATIO)
    train_size = n - test_size

    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            random_state=42,
            max_depth=3,
            learning_rate=0.06,
            max_iter=350
        ))
    ])
    model.fit(X_train, y_train)

    # AUC opcional
    auc = None
    if len(np.unique(y_test.values)) == 2:
        proba_test = model.predict_proba(X_test)[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_test, proba_test))

    # reentrena con todo
    model.fit(X, y)

    last_row = df_feat.dropna().iloc[-1:]
    p_up = float(model.predict_proba(last_row[FEATURE_COLS])[:, 1][0])
    detail = f"ML (HistGB) entrenado con ~{len(df)} filas, test={ML_TEST_RATIO:.0%}, AUC={auc:.3f}" if auc is not None \
             else f"ML (HistGB) entrenado con ~{len(df)} filas, test={ML_TEST_RATIO:.0%}, AUC=N/A"

    return clip01(p_up), detail, auc

def compute_methods(df_feat: pd.DataFrame) -> Tuple[List[MethodResult], float]:
    last = df_feat.dropna().iloc[-1]
    results: List[MethodResult] = []

    # 1) ML
    p_ml, detail_ml, _auc = method_ml_classifier(df_feat)
    results.append(MethodResult(
        name="Modelo ML (HistGradientBoosting)",
        proba_up=p_ml, proba_down=1 - p_ml,
        signal=signal_from_proba(p_ml),
        detail=detail_ml
    ))

    # 2) Bollinger
    p, d = method_bollinger(float(last["bb_z"]))
    results.append(MethodResult(
        name="Bandas de Bollinger (z-score 20)",
        proba_up=p, proba_down=1 - p,
        signal=signal_from_proba(p),
        detail=d
    ))

    # 3) RSI
    p, d = method_rsi(float(last["rsi_14"]))
    results.append(MethodResult(
        name="RSI (14)",
        proba_up=p, proba_down=1 - p,
        signal=signal_from_proba(p),
        detail=d
    ))

    # 4) MACD
    p, d = method_macd(float(last["macd_hist"]))
    results.append(MethodResult(
        name="MACD Histograma",
        proba_up=p, proba_down=1 - p,
        signal=signal_from_proba(p),
        detail=d
    ))

    # 5) Cruce SMA
    p, d = method_sma_crossover(float(last["sma_ratio_20_50"]))
    results.append(MethodResult(
        name="Tendencia (SMA20 vs SMA50)",
        proba_up=p, proba_down=1 - p,
        signal=signal_from_proba(p),
        detail=d
    ))

    # 6) Momentum
    p, d = method_momentum(float(last["ret_10"]))
    results.append(MethodResult(
        name="Momentum (retorno 10 días)",
        proba_up=p, proba_down=1 - p,
        signal=signal_from_proba(p),
        detail=d
    ))

    # promedio simple como “consenso”
    consensus = float(np.mean([r.proba_up for r in results]))
    return results, clip01(consensus)

# =========================
# Top movers
# =========================
def compute_top_movers(universe: List[str], top_n: int = 10) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    changes = []
    for t in universe:
        try:
            df = fetch_daily_ohlcv_stooq(t)
            if len(df) < 2:
                continue
            last = df.iloc[-1]["Close"]
            prev = df.iloc[-2]["Close"]
            if prev and prev != 0:
                chg = (last / prev - 1.0) * 100.0
                changes.append((t, float(chg)))
        except Exception:
            continue

    changes.sort(key=lambda x: x[1], reverse=True)
    top_up = changes[:top_n]
    top_down = sorted(changes, key=lambda x: x[1])[:top_n]
    return top_up, top_down

# =========================
# Explanations (paragraphs)
# =========================
METHOD_EXPLANATIONS = [
    ("Modelo ML (HistGradientBoosting)",
     "Entrena un clasificador con datos históricos (retornos, volatilidad, RSI, MACD, Bollinger, etc.) para estimar la probabilidad de que el precio suba el próximo día. No “adivina”; aprende patrones estadísticos y puede fallar en cambios de régimen."),
    ("Bandas de Bollinger (z-score 20)",
     "Mide cuán lejos está el precio de su media móvil de 20 días en unidades de desviación estándar (z-score). Valores muy negativos suelen interpretarse como ‘precio relativamente bajo’ (posible rebote), y valores muy positivos como ‘precio relativamente alto’ (posible corrección)."),
    ("RSI (14)",
     "Oscilador de 0 a 100 que estima fuerza del movimiento reciente. Por debajo de ~30 suele considerarse sobreventa (mayor probabilidad de rebote) y por encima de ~70 sobrecompra (mayor probabilidad de retroceso)."),
    ("MACD Histograma",
     "Compara dos medias móviles exponenciales para detectar cambios de momentum. El histograma positivo sugiere aceleración alcista; negativo, aceleración bajista. Es útil en tendencias, pero puede dar ruido en rangos."),
    ("Tendencia (SMA20 vs SMA50)",
     "Si la SMA20 está por encima de la SMA50, se interpreta como tendencia alcista a corto/medio plazo; si está por debajo, tendencia bajista. Funciona mejor en mercados tendenciales que en laterales."),
    ("Momentum (retorno 10 días)",
     "Asume persistencia de movimientos recientes: si el activo subió con fuerza en 10 días, puede seguir con sesgo alcista (y viceversa). Suele degradarse en mercados con reversión a la media.")
]

# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    ticker = DEFAULT_TICKER
    error = None

    if request.method == "POST":
        ticker = (request.form.get("ticker") or DEFAULT_TICKER).strip().upper()

    try:
        df = fetch_daily_ohlcv_stooq(ticker)
        df = df.tail(LOOKBACK_DAYS).copy()
        df_feat = make_features(df).dropna().copy()

        last_date = df_feat.iloc[-1]["Date"].date()
        last_close = float(df_feat.iloc[-1]["Close"])

        methods, consensus = compute_methods(df_feat)

        top_up, top_down = compute_top_movers(UNIVERSE, top_n=10)

        return render_template(
            "index.html",
            ticker=ticker,
            last_date=last_date,
            last_close=last_close,
            methods=methods,
            consensus=consensus,
            consensus_signal=signal_from_proba(consensus),
            explanations=METHOD_EXPLANATIONS,
            top_up=top_up,
            top_down=top_down,
            error=None
        )

    except Exception as e:
        error = str(e)
        # aún mostramos movers aunque falle el ticker
        top_up, top_down = compute_top_movers(UNIVERSE, top_n=10)
        return render_template(
            "index.html",
            ticker=ticker,
            error=error,
            methods=[],
            consensus=None,
            consensus_signal=None,
            explanations=METHOD_EXPLANATIONS,
            top_up=top_up,
            top_down=top_down,
            last_date=None,
            last_close=None
        )

if __name__ == "__main__":
    app.run(debug=True)
