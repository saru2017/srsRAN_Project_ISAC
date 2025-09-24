import zmq, struct, numpy as np, time
import numpy as np

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect("tcp://localhost:5556")
sock.setsockopt(zmq.SUBSCRIBE, b"")
time.sleep(0.1)

# (sfn,slot)ごとに集約
current_key = None
buf = None  # shape -> (n_rx, n_tx, length); filled with NaN



def maybe_print_done(key, buf):
    if not np.isnan(buf).any():
        n_rx, n_tx, L = buf.shape
        print(f"[complete] key={key} shape={buf.shape}")

        # 表示設定（サブキャリアが長いので省略表示する）
        np.set_printoptions(precision=3, suppress=True, threshold=20)

        # 全体を出したい場合
        print("Full coeff matrix:")
        print(buf)

        # 先頭のサブキャリアだけ見たい場合
        print("\nFirst subcarrier across Rx/Tx:")
        print(buf[:, :, 0])   # shape (n_rx, n_tx)

        # 先頭の数サブキャリアを行列で
        print("\nFirst 5 subcarriers (per Rx/Tx):")
        print(buf[:, :, :5])

        return True
    return False



while True:
    hdr, data = sock.recv_multipart()
    sfn, slot, tx_port, rx_port_idx, n_rx, n_tx, k0, step, length = struct.unpack("<I H H H H H H H H", hdr)
    coeffs = np.frombuffer(data, dtype=np.complex64)

    # 受け取ったペイロードの形に応じて分岐
    if coeffs.size == length:
        # 1 (rx,tx) スライス分だけが送られてきたケース（いまのあなたの実装）
        key = (sfn, slot)
        if key != current_key or buf is None or buf.shape != (n_rx, n_tx, length):
            # 新しいスロット開始：バッファを作り直し（NaN で初期化）
            buf = np.full((n_rx, n_tx, length), np.nan + 1j*np.nan, dtype=np.complex64)
            current_key = key
        buf[rx_port_idx, tx_port, :] = coeffs

        # 全部埋まったら出力
        if maybe_print_done(current_key, buf):
            buf = None
            current_key = None

    elif coeffs.size == n_rx * length and n_tx == 1:
        # （将来的に）tx_port 1本分をまとめて送った場合の例
        arr = coeffs.reshape(n_rx, length)
        print(f"[rx-pack] sfn={sfn} slot={slot} tx={tx_port} shape={arr.shape}")

    elif coeffs.size == n_rx * n_tx * length:
        # （将来的に）全 (rx,tx) 分を一括送信した場合の例
        arr = coeffs.reshape(n_rx, n_tx, length)
        print(f"[full] sfn={sfn} slot={slot} shape={arr.shape}")

    else:
        print(f"[warn] unexpected payload size={coeffs.size} (n_rx={n_rx} n_tx={n_tx} len={length})")

