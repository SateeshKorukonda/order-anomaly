import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import poisson
from datetime import datetime

SEQ_LENGTH = 6  # number of 20‑min intervals = 2 hours
EPOCHS = 15

@st.cache_data(show_spinner=False)
def preprocess(df_raw: pd.DataFrame, client: str) -> pd.DataFrame:
    """Filter for selected client and aggregate to 20‑minute intervals."""
    df = df_raw.copy()
    df['order_count'] = pd.to_numeric(df['order_count'], errors='coerce')
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='mixed', dayfirst=True, errors='coerce')
    df = df[(df['client_code'] == client) & df['order_count'].notna() & df['datetime'].notna()]
    df['interval_20min'] = df['datetime'].dt.floor('20min')
    df_agg = df.groupby('interval_20min')['order_count'].sum().reset_index()
    
    # add time‑based covariates
    df_agg['hour'] = df_agg['interval_20min'].dt.hour
    df_agg['minute'] = df_agg['interval_20min'].dt.minute
    df_agg['day_of_week'] = df_agg['interval_20min'].dt.dayofweek
    return df_agg.sort_values('interval_20min')

def create_sequences(data: np.ndarray, seq_length: int = SEQ_LENGTH) -> np.ndarray:
    """Convert a (n × f) array into (n‑seq+1) × seq × f sequences"""
    return np.array([data[i:i + seq_length] for i in range(len(data) - seq_length + 1)])

def train_model(df: pd.DataFrame):
    features = ['order_count', 'hour', 'minute', 'day_of_week']
    X = df[features].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq = create_sequences(X_scaled)

    model = Sequential([
        LSTM(32, activation='relu', input_shape=(SEQ_LENGTH, len(features)), return_sequences=False),
        RepeatVector(SEQ_LENGTH),
        LSTM(32, activation='relu', return_sequences=True),
        TimeDistributed(Dense(len(features)))
    ])
    model.compile(optimizer='adam', loss='mse')
    early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_seq, X_seq, epochs=EPOCHS, batch_size=32, validation_split=0.2,
              callbacks=[early], verbose=0)
    return model, scaler

def detect_anomaly(history: pd.DataFrame, model, scaler, ts: datetime, order_cnt: int):
    # align to 20‑minute grid
    ts_floor = pd.to_datetime(ts).floor('20min')

    new_row = pd.DataFrame({
        'interval_20min': [ts_floor],
        'order_count': [order_cnt]
    })
    new_row['hour'] = new_row['interval_20min'].dt.hour
    new_row['minute'] = new_row['interval_20min'].dt.minute
    new_row['day_of_week'] = new_row['interval_20min'].dt.dayofweek

    df_full = pd.concat([history, new_row]).drop_duplicates('interval_20min', keep='last').sort_values('interval_20min')

    if len(df_full) < SEQ_LENGTH:
        return None, f'Need at least {SEQ_LENGTH} past intervals (≈ {SEQ_LENGTH*20//60} hours) for detection.'

    features = ['order_count', 'hour', 'minute', 'day_of_week']
    X = df_full[features].values
    X_scaled = scaler.transform(X)
    X_seq = create_sequences(X_scaled)

    X_pred = model.predict(X_seq, verbose=0)
    last_pred_scaled = X_pred[-1, -1, :]
    pred_order = scaler.inverse_transform([last_pred_scaled])[0][0]

    actual = order_cnt
    p_val = poisson.cdf(actual, pred_order)
    rolling = df_full['order_count'].tail(12)  # last 4 hours
    rolling_mean, rolling_std = rolling.mean(), rolling.std()
    anomaly = (p_val < 0.001) and (actual < 0.5 * pred_order) and (actual < (rolling_mean - 2 * rolling_std))

    result = {
        'timestamp': ts_floor,
        'actual': actual,
        'predicted': pred_order,
        'p_value': p_val,
        'anomaly': anomaly
    }
    return result, None

def main():
    st.title('Single‑client Order Anomaly Detector')
    st.write('Upload historical order data, train a model, and check new data points for anomalies.')

    file = st.file_uploader('⬆️ Upload CSV', type='csv')
    if file is None:
        st.info('Waiting for a CSV file…')
        st.stop()

    df_raw = pd.read_csv(file)
    required_cols = {'date', 'time', 'client_code', 'order_count'}
    if not required_cols.issubset(df_raw.columns):
        st.error('CSV must contain **date, time, client_code, order_count** columns.')
        st.stop()

    clients = sorted(df_raw['client_code'].unique())
    client = st.selectbox('Select client to model', clients)

    df_client = preprocess(df_raw, client)
    st.write(f'Aggregated rows for {client}: **{len(df_client)}**')

    if len(df_client) < SEQ_LENGTH:
        st.error(f'Need at least {SEQ_LENGTH} aggregated rows (≈ {SEQ_LENGTH*20//60} hours) for training.')
        st.stop()

    if st.button('Train / Retrain model'):
        with st.spinner('Training…'):
            model, scaler = train_model(df_client)
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['history'] = df_client
        st.success('Model ready! ➡️ Enter a new data point below.')

    if all(key in st.session_state for key in ('model', 'scaler', 'history')):
        st.subheader('🔍 Check a new data point')
        col1, col2 = st.columns(2)
        with col1:
            date_val = st.date_input('Date', value=datetime.today())
            time_val = st.time_input('Time', value=datetime.now().time().replace(second=0, microsecond=0))
        order_val = st.number_input('Order count', min_value=0, value=0, step=1)

        if st.button('Detect anomaly'):
            ts = datetime.combine(date_val, time_val)
            result, err = detect_anomaly(st.session_state['history'], st.session_state['model'], st.session_state['scaler'], ts, int(order_val))
            if err:
                st.error(err)
            else:
                st.write(f'**Predicted count:** {result["predicted"]:.2f}')
                st.write(f'**P‑value:** {result["p_value"]:.4f}')
                st.markdown('### 🚨 Anomaly detected!' if result['anomaly'] else '✅ No anomaly.')
    else:
        st.info('Train the model to enable the anomaly checker.')

if __name__ == '__main__':
    main()
