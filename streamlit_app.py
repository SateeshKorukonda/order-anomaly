import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import poisson
from datetime import datetime

SEQ_LENGTH = 6          # 6 × 20‑min = 2 h window
EPOCHS = 15

# -------------------------------------------------
@st.cache_data(show_spinner=False)
def preprocess(df_raw: pd.DataFrame, client: str) -> pd.DataFrame:
    """Filter for one client & aggregate to 20‑min bins."""
    df = df_raw[df_raw['client_code'] == client].copy()
    # Combine date & time columns to datetime
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('timestamp', inplace=True)

    # 20‑minute resample
    df_agg = (df
              .resample('20min')['order_count']
              .sum()
              .rename('order_count')
              .reset_index())

    # Calendar features
    df_agg['hour'] = df_agg['timestamp'].dt.hour
    df_agg['minute'] = df_agg['timestamp'].dt.minute
    df_agg['day_of_week'] = df_agg['timestamp'].dt.dayofweek
    df_agg.rename(columns={'timestamp': 'interval_20min'}, inplace=True)
    return df_agg.sort_values('interval_20min')

# -------------------------------------------------
def create_supervised(df: pd.DataFrame,
                      feats=('order_count', 'hour', 'minute', 'day_of_week'),
                      n: int = SEQ_LENGTH):
    """Return X.shape=(samples,n,len(feats)), y.shape=(samples,)"""
    X, y = [], []
    for i in range(len(df) - n):
        X.append(df[feats].iloc[i:i+n].values)
        y.append(df['order_count'].iloc[i+n])
    return np.array(X), np.array(y)

# -------------------------------------------------
def train_model(df: pd.DataFrame):
    feats = ['order_count', 'hour', 'minute', 'day_of_week']
    X, y = create_supervised(df, feats, SEQ_LENGTH)

    # Scale inputs & target separately
    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(
        X.reshape(-1, len(feats))
    ).reshape(X.shape)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    model = Sequential([
        LSTM(32, activation='relu', input_shape=(SEQ_LENGTH, len(feats))),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    early = EarlyStopping(monitor='val_loss', patience=3,
                          restore_best_weights=True)
    model.fit(X_scaled, y_scaled,
              epochs=EPOCHS, batch_size=32,
              validation_split=0.2, callbacks=[early],
              verbose=0)
    return model, scaler_x, scaler_y

# -------------------------------------------------
def detect_anomaly(history: pd.DataFrame,
                   model, scaler_x, scaler_y,
                   ts: datetime, order_cnt: int):
    ts_floor = pd.to_datetime(ts).floor('20min')

    # Build supervised window from *past* SEQ_LENGTH intervals
    past = history[history['interval_20min'] < ts_floor].tail(SEQ_LENGTH)
    if len(past) < SEQ_LENGTH:
        return None, (f'Need ≥ {SEQ_LENGTH} prior intervals '
                      f'for {ts_floor} prediction.')

    feats = ['order_count', 'hour', 'minute', 'day_of_week']
    X = past[feats].values
    X_scaled = scaler_x.transform(X).reshape(1, SEQ_LENGTH, len(feats))

    pred_scaled = model.predict(X_scaled, verbose=0)[0, 0]
    pred_order = scaler_y.inverse_transform([[pred_scaled]])[0, 0]

    actual = order_cnt
    p_val = poisson.cdf(actual, pred_order)
    rolling = history['order_count'].tail(12)  # exclude test point
    anomaly = (
        (p_val < 0.001) and
        (actual < 0.5 * pred_order) and
        (actual < (rolling.mean() - 2 * rolling.std()))
    )

    return {
        'timestamp': ts_floor,
        'actual': actual,
        'predicted': pred_order,
        'p_value': p_val,
        'anomaly': anomaly
    }, None

# -------------------------------------------------
def main():
    st.title('Order‑count anomaly detector')

    file = st.file_uploader('⬆️ CSV with date, time, client_code, order_count')
    if file is None:
        st.info('Waiting for a CSV…')
        st.stop()

    df_raw = pd.read_csv(file)
    required = {'date', 'time', 'client_code', 'order_count'}
    if not required.issubset(df_raw.columns):
        missing = ", ".join(required)
        st.error(f'CSV must contain: {missing}')
        st.stop()

    client = st.selectbox('Client to model',
                          sorted(df_raw['client_code'].unique()))
    df_client = preprocess(df_raw, client)
    st.write(f'Aggregated rows: **{len(df_client)}**')

    if len(df_client) < SEQ_LENGTH + 1:
        st.error(f'Need ≥ {SEQ_LENGTH + 1} rows for training.')
        st.stop()

    if st.button('Train / Retrain model'):
        with st.spinner('Training…'):
            model, sx, sy = train_model(df_client)
        st.session_state.update(model=model, scaler_x=sx,
                                scaler_y=sy, history=df_client)
        st.success('Model ready! Enter a new data point below.')

    if {'model', 'scaler_x', 'scaler_y', 'history'} <= st.session_state.keys():
        st.subheader('🔍 Check a new point')
        col1, col2 = st.columns(2)
        with col1:
            ts_str = st.text_input('Timestamp (YYYY‑MM‑DD HH:MM)',
                                   value=datetime.now()
                                   .strftime('%Y‑%m‑%d %H:%M'))
        with col2:
            cnt = st.number_input('Order count', min_value=0, value=0)

        if st.button('Detect'):
            try:
                ts = datetime.strptime(ts_str, '%Y‑%m‑%d %H:%M')
            except ValueError:
                st.error('Invalid timestamp format.')
                st.stop()

            res, err = detect_anomaly(st.session_state['history'],
                                      st.session_state['model'],
                                      st.session_state['scaler_x'],
                                      st.session_state['scaler_y'],
                                      ts, int(cnt))
            if err:
                st.error(err)
            else:
                st.write(f'**Predicted (expected):** {res["predicted"]:.2f}')
                st.write(f'**P‑value:** {res["p_value"]:.4f}')
                st.write(f'**Anomaly:** {res["anomaly"]}')
                st.markdown('### 🚨 **Anomaly!**' if res['anomaly']
                            else '### ✅ All good.')
    else:
        st.info('Train the model first.')

if __name__ == '__main__':
    main()
