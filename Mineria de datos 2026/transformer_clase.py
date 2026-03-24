import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import layers, callbacks


# =========================================================
# CONFIGURACIÓN
# =========================================================
CSV_PATH = "/home/enrique/flower/Pruebas varias/data-model-consumoA-60T.csv"
TARGET_COL = "cons_total"
DATE_COL = "Date"

WINDOW_SIZE = 48          # prueba también 24 o 72
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 1e-3

MODEL_DIM = 64
HEAD_SIZE = 32
NUM_HEADS = 4
FF_DIM = 128
NUM_TRANSFORMER_BLOCKS = 2
DENSE_UNITS = 64
DROPOUT_RATE = 0.1

SEED = 42
OUTPUT_DIR = "outputs_transformer_mejorado"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# SEMILLA Y GPU
# =========================================================
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


# =========================================================
# CARGA Y PREPARACIÓN
# =========================================================
def load_dataframe(csv_path, target_col, date_col):
    df = pd.read_csv(csv_path, sep=";")

    for col in ["Unnamed: 0", "index", "Index"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe. Columnas: {list(df.columns)}")

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        df = df.sort_values(date_col).reset_index(drop=True)

        # Features cíclicas de tiempo
        df["hour_sin"] = np.sin(2 * np.pi * df[date_col].dt.hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df[date_col].dt.hour / 24.0)

        df["dow_sin"] = np.sin(2 * np.pi * df[date_col].dt.dayofweek / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df[date_col].dt.dayofweek / 7.0)

        df["month_sin"] = np.sin(2 * np.pi * (df[date_col].dt.month - 1) / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * (df[date_col].dt.month - 1) / 12.0)

    df = df.dropna().reset_index(drop=True)
    return df


def build_numeric_features(df, target_col, date_col):
    """
    Mantiene TODAS las columnas numéricas, incluyendo el histórico de cons_total.
    Eso es clave para este problema.
    """
    work_df = df.copy()

    if date_col in work_df.columns:
        dates = work_df[date_col].copy()
        work_df = work_df.drop(columns=[date_col])
    else:
        dates = pd.Series(np.arange(len(work_df)))

    numeric_cols = work_df.select_dtypes(include=[np.number]).columns.tolist()

    if target_col not in numeric_cols:
        raise ValueError(f"La columna objetivo '{target_col}' no es numérica.")

    X_all = work_df[numeric_cols].to_numpy(dtype=np.float32)
    y_all = work_df[target_col].to_numpy(dtype=np.float32)
    dates_all = dates.to_numpy()

    return X_all, y_all, dates_all, numeric_cols


def make_windows_predicting_delta(X_all, y_all, dates_all, window_size):
    """
    Para cada instante t:
      entrada  = [t-window_size, ..., t-1]
      target   = y[t] - y[t-1]
      baseline = y[t-1]
      valor real final = y[t]
    """
    X_seq = []
    y_delta = []
    last_target = []
    y_true = []
    target_dates = []

    for i in range(window_size, len(X_all)):
        X_seq.append(X_all[i - window_size:i])
        y_delta.append(y_all[i] - y_all[i - 1])
        last_target.append(y_all[i - 1])
        y_true.append(y_all[i])
        target_dates.append(dates_all[i])

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_delta = np.asarray(y_delta, dtype=np.float32).reshape(-1, 1)
    last_target = np.asarray(last_target, dtype=np.float32).reshape(-1, 1)
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1, 1)
    target_dates = np.asarray(target_dates)

    return X_seq, y_delta, last_target, y_true, target_dates


def split_by_time(X, y_delta, last_target, y_true, dates, train_ratio=0.70, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        X[:train_end], y_delta[:train_end], last_target[:train_end], y_true[:train_end], dates[:train_end],
        X[train_end:val_end], y_delta[train_end:val_end], last_target[train_end:val_end], y_true[train_end:val_end], dates[train_end:val_end],
        X[val_end:], y_delta[val_end:], last_target[val_end:], y_true[val_end:], dates[val_end:]
    )


def scale_data(X_train, X_val, X_test, y_delta_train, y_delta_val, y_delta_test):
    n_features = X_train.shape[-1]

    x_scaler = StandardScaler()
    delta_scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d = X_val.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    X_train_scaled = x_scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled = x_scaler.transform(X_val_2d).reshape(X_val.shape)
    X_test_scaled = x_scaler.transform(X_test_2d).reshape(X_test.shape)

    y_delta_train_scaled = delta_scaler.fit_transform(y_delta_train)
    y_delta_val_scaled = delta_scaler.transform(y_delta_val)
    y_delta_test_scaled = delta_scaler.transform(y_delta_test)

    return (
        X_train_scaled.astype(np.float32),
        X_val_scaled.astype(np.float32),
        X_test_scaled.astype(np.float32),
        y_delta_train_scaled.astype(np.float32),
        y_delta_val_scaled.astype(np.float32),
        y_delta_test_scaled.astype(np.float32),
        x_scaler,
        delta_scaler,
    )


# =========================================================
# POSICIONAL + TRANSFORMER
# =========================================================
class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.position_embedding = layers.Embedding(input_dim=seq_len, output_dim=d_model)

    def call(self, x):
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        pos_embed = self.position_embedding(positions)   # (seq_len, d_model)
        return x + pos_embed


class TransformerRegressor:
    def __init__(
        self,
        input_shape,
        model_dim=64,
        head_size=32,
        num_heads=4,
        ff_dim=128,
        num_transformer_blocks=2,
        dense_units=64,
        dropout_rate=0.1,
        learning_rate=1e-3,
    ):
        self.input_shape = input_shape
        self.seq_len = input_shape[0]
        self.n_features = input_shape[1]

        self.model_dim = model_dim
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

    def _transformer_encoder(self, inputs):
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=self.head_size,
            num_heads=self.num_heads,
            dropout=self.dropout_rate
        )(x, x)
        x = layers.Dropout(self.dropout_rate)(x)
        res = x + inputs

        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Conv1D(filters=self.model_dim, kernel_size=1)(x)

        return x + res

    def build(self):
        inputs = tf.keras.Input(shape=self.input_shape)

        # Proyección inicial a model_dim
        x = layers.Dense(self.model_dim)(inputs)

        # Información posicional
        x = PositionalEmbedding(self.seq_len, self.model_dim)(x)

        for _ in range(self.num_transformer_blocks):
            x = self._transformer_encoder(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(self.dense_units, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Salida = delta predicho
        outputs = layers.Dense(1, activation="linear")(x)

        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.Huber(),
            metrics=["mae"]
        )
        return model


# =========================================================
# MÉTRICAS
# =========================================================
def regression_metrics(y_true, y_pred, name="split"):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"{name.upper()} -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.6f}")
    return rmse, mae, r2


def evaluate_transformer(model, X_scaled, y_true_abs, last_target_abs, delta_scaler, name="test"):
    pred_delta_scaled = model.predict(X_scaled, verbose=0)
    pred_delta = delta_scaler.inverse_transform(pred_delta_scaled)

    y_pred_abs = last_target_abs + pred_delta

    regression_metrics(y_true_abs, y_pred_abs, name=name)
    return y_pred_abs


def evaluate_baseline_last_value(y_true_abs, last_target_abs, name="baseline"):
    print(f"\nBaseline '{name}' (pred = último valor):")
    regression_metrics(y_true_abs, last_target_abs, name=name)


# =========================================================
# GRÁFICAS
# =========================================================
def plot_training_history(history, save_path=None):
    plt.figure(figsize=(12, 5))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Val loss")
    plt.title("Evolución de la pérdida")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_predictions(dates, y_true, y_pred, save_path=None, max_points=300):
    n = min(max_points, len(y_true))

    plt.figure(figsize=(15, 6))
    plt.plot(dates[:n], y_true[:n], label="Real")
    plt.plot(dates[:n], y_pred[:n], label="Predicción")
    plt.title("Predicción vs valor real (test)")
    plt.xlabel("Fecha")
    plt.ylabel(TARGET_COL)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(SEED)
    configure_gpu()

    print("Leyendo dataset...")
    df = load_dataframe(CSV_PATH, TARGET_COL, DATE_COL)

    print(f"Filas: {len(df)}")
    print("Columnas:")
    print(df.columns.tolist())

    X_all, y_all, dates_all, feature_cols = build_numeric_features(df, TARGET_COL, DATE_COL)

    print(f"\nNúmero de variables de entrada: {len(feature_cols)}")
    print(feature_cols)

    X_seq, y_delta, last_target, y_true, target_dates = make_windows_predicting_delta(
        X_all, y_all, dates_all, WINDOW_SIZE
    )

    print(f"\nShapes:")
    print("X_seq       :", X_seq.shape)
    print("y_delta     :", y_delta.shape)
    print("last_target :", last_target.shape)
    print("y_true      :", y_true.shape)

    (
        X_train, y_delta_train, last_train, y_train_abs, dates_train,
        X_val, y_delta_val, last_val, y_val_abs, dates_val,
        X_test, y_delta_test, last_test, y_test_abs, dates_test
    ) = split_by_time(
        X_seq, y_delta, last_target, y_true, target_dates,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO
    )

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_delta_train_scaled,
        y_delta_val_scaled,
        y_delta_test_scaled,
        x_scaler,
        delta_scaler,
    ) = scale_data(
        X_train, X_val, X_test,
        y_delta_train, y_delta_val, y_delta_test
    )

    print("\nBaseline antes de entrenar:")
    evaluate_baseline_last_value(y_val_abs, last_val, name="val_last_value")
    evaluate_baseline_last_value(y_test_abs, last_test, name="test_last_value")

    model = TransformerRegressor(
        input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
        model_dim=MODEL_DIM,
        head_size=HEAD_SIZE,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
    ).build()

    model.summary()

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(OUTPUT_DIR, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    print("\nEntrenando Transformer...")
    history = model.fit(
        X_train_scaled,
        y_delta_train_scaled,
        validation_data=(X_val_scaled, y_delta_val_scaled),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=1
    )

    print("\nResultados del Transformer:")
    y_train_pred = evaluate_transformer(model, X_train_scaled, y_train_abs, last_train, delta_scaler, name="train")
    y_val_pred = evaluate_transformer(model, X_val_scaled, y_val_abs, last_val, delta_scaler, name="val")
    y_test_pred = evaluate_transformer(model, X_test_scaled, y_test_abs, last_test, delta_scaler, name="test")

    # Guardar
    model.save(os.path.join(OUTPUT_DIR, "final_model.keras"))
    pd.DataFrame(history.history).to_csv(os.path.join(OUTPUT_DIR, "history.csv"), index=False)

    pred_df = pd.DataFrame({
        "Date": dates_test,
        "y_true": y_test_abs.ravel(),
        "y_pred": y_test_pred.ravel(),
        "baseline_last_value": last_test.ravel(),
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "predicciones_test.csv"), index=False)

    plot_training_history(
        history,
        save_path=os.path.join(OUTPUT_DIR, "loss_curve.png")
    )

    plot_predictions(
        dates_test,
        y_test_abs.ravel(),
        y_test_pred.ravel(),
        save_path=os.path.join(OUTPUT_DIR, "pred_vs_real.png"),
        max_points=300
    )

    print("\nArchivos guardados en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()