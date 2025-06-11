import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import poisson
from datetime import datetime

SEQ_LENGTH = 6          # 6 × 20-min = 2 h window
EPOCHS = 15
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def preprocess(df_raw: pd.DataFrame, client: str) -> pd.DataFrame:
    """Filter for one client & aggregate to 20-min bins."""
    df = df_raw.copy()
    df['order_count'] = pd.to_numeric(df['order_count'], errors='coerce')
    df['datetime'] = pd.to_datetime(
        df['date'] + ' ' + df['time'],
        format='mixed', dayfirst=True, errors='coerce'
    )
    df = df[(df['client_code'] == client) &
            df['order_count'].notna() & df['datetime'].notna()]

    df['interval_20min'] = df['datetime'].dt.floor('20min')
    df_agg = df.groupby('interval_20min')['order_count'].sum().reset_index()
    df_agg['hour'] = df_agg['interval_20min'].dt.hour
    df_agg['minute'] = df_agg['interval_20min'].dt.minute
    df_agg['day_of_week'] = df_agg['interval_20min'].dt.dayofweek
    return df_agg.sort_values('interval_20min')


def create_sequences(arr: np.ndarray, n: int = SEQ_LENGTH) -> np.ndarray:
    return np.array([arr[i:i + n] for i in range(len(arr) - n + 1)])


def train_model(df: pd.DataFrame):
    feats = ['order_count', 'hour', 'minute', 'day_of_week']
    X = df[feats].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq = create_sequences(X_scaled)

    model = Sequential([
        LSTM(32, activation='relu',
             input_shape=(SEQ_LENGTH, len(feats)), return_sequences=False),
        RepeatVector(SEQ_LENGTH),
        LSTM(32, activation='relu', return_sequences=True),
        TimeDistributed(Dense(len(feats)))
    ])
    model.compile(optimizer='adam', loss='mse')
    early = EarlyStopping(monitor='val_loss', patience=3,
                          restore_best_weights=True)
    model.fit(X_seq, X_seq, epochs=EPOCHS, batch_size=32,
              validation_split=0.2, callbacks=[early], verbose=0)
    return model, scaler


def detect_anomaly(history, model, scaler, ts: datetime, order_cnt: int):
    ts_floor = pd.to_datetime(ts).floor('20min')
    new = pd.DataFrame({
        'interval_20min': [ts_floor],
        'order_count': [order_cnt]
    })
    new['hour'] = new['interval_20min'].dt.hour
    new['minute'] = new['interval_20min'].dt.minute
    new['day_of_week'] = new['interval_20min'].dt.dayofweek

    df_full = (pd.concat([history, new])
               .drop_duplicates('interval_20min', keep='last')
               .sort_values('interval_20min'))

    if len(df_full) < SEQ_LENGTH:
        return None, (f'Need ≥ {SEQ_LENGTH} prior intervals '
                      f'(≈ {SEQ_LENGTH*20//60} h) for detection.')

    feats = ['order_count', 'hour', 'minute', 'day_of_week']
    X_scaled = scaler.transform(df_full[feats].values)
    X_seq = create_sequences(X_scaled)
    X_pred = model.predict(X_seq, verbose=0)
    last_pred_scaled = X_pred[-1, -1, :]
    pred_order = scaler.inverse_transform([last_pred_scaled])[0][0]

    actual = order_cnt
    p_val = poisson.cdf(actual, pred_order)
    rolling = df_full['order_count'].tail(12)
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


def main():
    st.title('🛎️ Single-Client Order Anomaly Detector')
    st.write('Upload historical data → train → test a single data point.')

    file = st.file_uploader('⬆️ CSV with date, time, client_code, order_count')
    if file is None:
        st.info('Waiting for a CSV…')
        st.stop()

    df_raw = pd.read_csv(file)
    required = {'date', 'time', 'client_code', 'order_count'}
    if not required.issubset(df_raw.columns):
        st.error(f'CSV must contain: {\", \".join(required)}')
        st.stop()

    client = st.selectbox('Client to model', sorted(df_raw['client_code'].unique()))
    df_client = preprocess(df_raw, client)
    st.write(f'Aggregated rows: **{len(df_client)}**')

    if len(df_client) < SEQ_LENGTH:
        st.error(f'Need ≥ {SEQ_LENGTH} rows (≈ 2 h) for training.')
        st.stop()

    if st.button('Train / Retrain model'):
        with st.spinner('Training…'):
            model, scaler = train_model(df_client)
        st.session_state.update(model=model, scaler=scaler, history=df_client)
        st.success('Model ready! Enter a new data point below.')

    if {'model', 'scaler', 'history'} <= st.session_state.keys():
        st.subheader('🔍 Check a new point')
        col1, col2 = st.columns(2)
        with col1:
            d = st.date_input('Date', value=datetime.today())
            t = st.time_input('Time', value=datetime.now()
                              .replace(second=0, microsecond=0).time())
        cnt = st.number_input('Order count', min_value=0, step=1)

        if st.button('Detect anomaly'):
            ts = datetime.combine(d, t)
            res, err = detect_anomaly(st.session_state['history'],
                                      st.session_state['model'],
                                      st.session_state['scaler'],
                                      ts, int(cnt))
            if err:
                st.error(err)
            else:
                st.write(f'**Predicted:** {res[\"predicted\"]:.2f}')
                st.write(f'**P-value:** {res[\"p_value\"]:.4f}')
                st.markdown('### 🚨 **Anomaly!**' if res['anomaly']
                            else '### ✅ All good.')

    else:
        st.info('Train the model first.')


if __name__ == '__main__':
    main()
