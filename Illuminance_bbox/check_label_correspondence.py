import pandas as pd
import numpy as np

def parse_custom_timestamp(ts_str):
    """
    '2026-06-17 06:16:34 533468' 形式をパースする。
    最後のスペース区切りをマイクロ秒として扱う。
    """
    ts_str = str(ts_str).strip()
    # 最後のスペースで分割
    parts = ts_str.rsplit(' ', 1)
    if len(parts) == 2 and parts[1].isdigit():
        # 'YYYY-MM-DD HH:MM:SS' + マイクロ秒
        dt_part, us_part = parts
        # マイクロ秒は6桁に揃える
        us_part = us_part.zfill(6)[:6]
        return pd.to_datetime(f'{dt_part}.{us_part}', format='%Y-%m-%d %H:%M:%S.%f')
    else:
        return pd.to_datetime(ts_str)

def analyze_label_timeline(csv_path, name):
    print(f'\n=== {name}: {csv_path} ===')
    df = pd.read_csv(csv_path)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # サンプル表示（フォーマット確認用）
    print(f'timestamp サンプル: {df["timestamp"].iloc[0]!r}')
    
    # カスタムパーサーで変換を試みる
    try:
        df['ts'] = df['timestamp'].apply(parse_custom_timestamp)
        df['sec'] = (df['ts'] - df['ts'].iloc[0]).dt.total_seconds()
    except Exception as e:
        print(f'カスタムパース失敗 ({e})、数値として扱います')
        df['sec'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['sec'] = df['sec'] - df['sec'].iloc[0]
    
    total_sec = df['sec'].iloc[-1]
    print(f'総時間: {total_sec:.1f} 秒, 総行数: {len(df)}')
    
    # 各ラベルが連続して現れる区間（セグメント）を抽出
    df['label_change'] = (df['label'] != df['label'].shift()).cumsum()
    segments = df.groupby(['label_change', 'label']).agg(
        start_sec=('sec', 'min'),
        end_sec=('sec', 'max'),
        count=('label', 'size')
    ).reset_index()
    
    print(f'\n連続セグメント（時系列順、上位20件）:')
    print(f'{"順":>3} {"label":>6} {"start[s]":>10} {"end[s]":>10} {"duration[s]":>12} {"count":>6}')
    for i, row in segments.head(20).iterrows():
        dur = row['end_sec'] - row['start_sec']
        print(f'{i+1:>3} {int(row["label"]):>6} {row["start_sec"]:>10.1f} {row["end_sec"]:>10.1f} {dur:>12.1f} {int(row["count"]):>6}')
    
    # 各ラベルの登場時間の重心
    print(f'\n各ラベルの平均登場時刻:')
    centroid = df.groupby('label')['sec'].mean().sort_values()
    for lbl, avg_sec in centroid.items():
        pct = avg_sec / total_sec * 100 if total_sec > 0 else 0
        print(f'  label={int(lbl)}: 平均 {avg_sec:>7.1f}秒 ({pct:>5.1f}%地点)')
    
    # BBox 高さ統計（列名を確認して集計）
    print(f'\nBBox 統計:')
    cols = df.columns.tolist()
    print(f'  利用可能な列: {cols}')
    
    # 一般的な列名パターンに対応
    z1_col = next((c for c in ['z1', 'z_min', 'zmin'] if c in cols), None)
    z2_col = next((c for c in ['z2', 'z_max', 'zmax'] if c in cols), None)
    
    if z1_col and z2_col:
        df['height'] = df[z2_col] - df[z1_col]
        df['cz'] = (df[z1_col] + df[z2_col]) / 2
        print(f'  高さ(z範囲) 統計:')
        print(df.groupby('label')[['height', 'cz']].agg(['mean', 'std']).round(3))


# 実環境データ
analyze_label_timeline(
    '/nas/data_2/ueno1212/light/data/keita_2026_0617/1_tsunoda_1/bbox_label.csv',
    'REAL (1_tsunoda_1)'
)

# シミュレーションデータ
analyze_label_timeline(
    '/nas/data_2/ueno1212/light/data/sim_lecture_02/loop1/bbox_label.csv',
    'SIM (loop1)'
)

analyze_label_timeline(
    '/nas/data_2/ueno1212/light/data/sim_lecture_02/loop2/bbox_label.csv',
    'SIM (loop2)'
)
