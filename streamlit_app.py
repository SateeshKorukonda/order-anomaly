import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import poisson
from datetime import datetime

"""Streamlit ‑ Order‑count anomaly detector

Key changes (June 2025)
-----------------------
1. **Tolerance slider** – choose how big a drop (fraction of expected) counts as
   an anomaly. 0.5 → flag anything below 50 % of expected; 0.8 → only large dips.
2. **Calendar baseline uses median** instead of mean, so a few double‑entries in
   a 20‑min slot no longer inflate the expectation.
3. All earlier robustness around timestamp parsing remains.
"""

SEQ_LENGTH = 6   # number of 20‑min intervals (2 h window)
EPOCHS      = 15

# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def preprocess(df_raw: pd.DataFrame, client: str) -> pd.DataFrame:
    """Aggregate to 20‑minute bins and append calendar features."""

    # 1 · Filter by client if column present
    if 'client_code' in df_raw.columns:
        df = df_raw[df_raw['client_code'] == client].copy()
    else:
        df = df_raw.copy()

    # 2 · Build a timestamp column robustly
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True,
                                         errors='coerce')
    elif {'date', 'time'} <= set(df.columns):
        combo = df['date'].astype(str) + ' ' + df['time'].astype(str)
        df['timestamp'] = pd.to_datetime(combo, dayfirst=True,
                                         errors='coerce')
    else:
        raise ValueError("CSV needs a 'timestamp' column or both 'date' and 'time'.")

    if df['timestamp'].isna().any():
        samp = df[df['timestamp'].isna()].iloc[0].to_dict()
        raise ValueError(f"Could not parse date/time for row like: {samp}")

    df.set_index('timestamp', inplace=True)

    # 3 · Ensure order_count numeric
    df['order_count'] = pd.to_numeric(df['order_count'], errors='coerce').fillna(0)

    # 4 · Aggregate to 20‑min bins – **sum** is still correct for volume
    df_agg = (df['order_count']
              .resample('20min').sum()
              .reset_index()
              .rename(columns={'timestamp': 'interval_20min'}))

    # 5 · Calendar features
    df_agg['hour']        = df_agg['interval_20min'].dt.hour
    df_agg['minute']      = df_agg['interval_20min'].dt.minute
    df_agg['day_of_week'] = df_agg['interval_20min'].dt.dayofweek

    return df_agg.sort_values('interval_20min')

# ──────────────────────────────────────────────────────────────────────────────

def create_supervised(df: pd.DataFrame,
                      feats=('order_count', 'hour', 'minute', 'day_of_week'),
                      n: int = SEQ_LENGTH):
    """Convert a df into (X, y) for sequence‑to‑one forecasting."""
    X, y = [], []
    for i in range(len(df) - n):
        X.append(df[feats].iloc[i:i + n].values)
        y.append(df['order_count'].iloc[i + n])
    return np.array(X), np.array(y)

# ──────────────────────────────────────────────────────────────────────────────

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ──────────────────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    feats = ['order_count', 'hour', 'minute', 'day_of_week']
    X, y = create_supervised(df, feats, SEQ_LENGTH)

    sx = MinMaxScaler()
    X_scaled = sx.fit_transform(X.reshape(-1, len(feats))).reshape(X.shape)

    sy = MinMaxScaler()
    y_scaled = sy.fit_transform(y.reshape(-1, 1))

    model = build_model((SEQ_LENGTH, len(feats)))
    early = EarlyStopping(monitor='val_loss', patience=3,
                          restore_best_weights=True)
    model.fit(X_scaled, y_scaled,
              epochs=EPOCHS, batch_size=32,
              validation_split=0.2, callbacks=[early],
              verbose=0)
    return model, sx, sy

# ──────────────────────────────────────────────────────────────────────────────

def detect_anomaly(history: pd.DataFrame,
                   model, sx, sy,
                   ts: datetime, order_cnt: int,
                   drop_threshold: float = 0.5):
    """Return diagnostics dict or (None, error_msg)."""

    ts_floor = pd.to_datetime(ts).floor('20min')
    h_max    = history['interval_20min'].max()
    feats    = ['order_count', 'hour', 'minute', 'day_of_week']
    actual   = order_cnt

    # ── Mode switch ──────────────────────────────────────────────────────────
    if (ts_floor - h_max) <= pd.Timedelta(hours=4):
        # Recent → use LSTM forecast
        past = history[history['interval_20min'] < ts_floor].tail(SEQ_LENGTH)
        if len(past) < SEQ_LENGTH:
            return None, 'Not enough history for sequence forecast.'
        Xs = sx.transform(past[feats].values)
        Xs = Xs.reshape(1, SEQ_LENGTH, len(feats))
        pred_scaled = model.predict(Xs, verbose=0)[0, 0]
        expected    = sy.inverse_transform([[pred_scaled]])[0, 0]
        mode        = 'sequence'
    else:
        # Far future → calendar median baseline
        mask = (
            (history['hour']   == ts_floor.hour) &
            (history['minute'] == ts_floor.minute) &
            (history['day_of_week'] == ts_floor.dayofweek)
        )
        subset = history[mask]
        if subset.empty:
            return None, 'No history for this weekday/time slot.'
        expected = subset['order_count'].median()  # ← changed to median
        mode     = 'calendar'

    expected = max(expected, 1e-6)  # safety against zero

    # ── Statistical tests ───────────────────────────────────────────────────
    p_val       = poisson.cdf(actual, expected)
    ratio_flag  = (actual / expected) < drop_threshold
    poisson_flag = p_val < 0.001
    anomaly     = ratio_flag and poisson_flag

    return {
        'timestamp': ts_floor,
        'actual': actual,
        'expected': expected,
        'p_value': p_val,
        'anomaly': anomaly,
        'mode': mode,
        'drop_threshold': drop_threshold,
    }, None

# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.title('📉 Order‑count anomaly detector')

    file = st.file_uploader('⬆️ CSV with order data')
    if file is None:
        st.info('Upload a CSV to continue.')
        st.stop()

    df_raw = pd.read_csv(file)
    if 'order_count' not in df_raw.columns:
        st.error("CSV must include an 'order_count' column.")
        st.stop()

    if 'client_code' not in df_raw.columns:
        df_raw['client_code'] = 'ALL'

    client = st.selectbox('Client to model', sorted(df_raw['client_code'].unique()))

    try:
        df_client = preprocess(df_raw, client)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.write(f'Aggregated rows: **{len(df_client)}**')

    if len(df_client) < SEQ_LENGTH + 1:
        st.error(f'Need at least {SEQ_LENGTH + 1} rows after aggregation.')
        st.stop()

    if st.button('🔄 Train / Retrain model'):
        with st.spinner('Training…'):
            model, sx, sy = train_model(df_client)
        st.session_state.update(model=model, sx=sx, sy=sy, history=df_client)
        st.success('Model trained! Enter a test point below.')

    # ── Detection UI ─────────────────────────────────────────────────────────
    if {'model', 'sx', 'sy', 'history'} <= st.session_state.keys():
        st.subheader('🔍 Check a new data point')
        col1, col2 = st.columns(2)
        with col1:
            ts_str = st.text_input('Timestamp (YYYY-MM-DD HH:MM)',
                                   value=datetime.now().strftime('%Y-%m-%d %H:%M'))
        with col2:
            cnt = st.number_input('Order count', min_value=0, value=0)

        drop_threshold = st.slider('Tolerance – flag if actual is below this fraction of expected',
                                   min_value=0.1, max_value=1.0, value=0.5, step=0.05)

        if st.button('Detect'):
            try:
                ts = pd.to_datetime(ts_str, dayfirst=False, errors='raise')
            except ValueError:
                st.error('Invalid timestamp format. Use YYYY-MM-DD HH:MM')
                st.stop()

            res, err = detect_anomaly(st.session_state['history'],
                                      st.session_state['model'],
                                      st.session_state['sx'],
                                      st.session_state['sy'],
                                      ts, int(cnt), drop_threshold)
            if err:
                st.error(err)
            else:
                st.write(f"**Expected:** {res['expected']:.2f} (mode: {res['mode']})")
                st.write(f"**P-value:** {res['p_value']:.3g}")
                st.write(f"**Anomaly:** {res['anomaly']}")
                st.markdown('### 🚨 **Anomaly!**' if res['anomaly'] else '### ✅ Normal behaviour')
    else:
        st.info('Train the model first.')

if __name__ == '__main__':
    main()
