import cv2
import pandas as pd
import numpy as np
from matplotlib.colors import to_rgb

def visualize_areas_with_colors(frame, csv_path, output_path, alpha=0.5):
    """
    フレームに指定されたエリアを描画し、CSVで指定された色を使用する。

    Parameters:
    - frame (numpy.ndarray): 画像フレーム
    - csv_path (str): エリア情報と色が記載されたCSVファイルのパス
    - output_path (str): 保存する画像ファイルのパス
    - alpha (float): オーバーレイの透明度
    """
    # CSVファイルを読み込む
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error: CSVファイルを読み込めませんでした - {e}")
        return

    # オーバーレイを作成
    overlay = frame.copy()

    # 各エリアを描画
    for _, row in df.iterrows():
        # 座標と色を取得
        x_start, y_start = int(row['開始x座標']), int(row['開始y座標'])
        x_end, y_end = int(row['終了x座標']), int(row['終了y座標'])
        color_name = row['色']

        # 色をBGR形式に変換
        try:
            color_rgb = to_rgb(color_name)  # RGB形式 (0~1)
            color_bgr = tuple(int(c * 255) for c in reversed(color_rgb))  # BGR形式 (0~255)
        except ValueError as e:
            print(f"Error: 色の指定が無効です ({color_name}) - {e}")
            continue

        # 四角形の領域を描画
        points = np.array([
            (x_start, y_start),
            (x_end, y_start),
            (x_end, y_end),
            (x_start, y_end)
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [points], color=color_bgr)

    # オーバーレイを適用
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # 保存
    cv2.imwrite(output_path, result)
    print(f"画像を保存しました: {output_path}")

# 使用例
video_path = "input/2024-1229_0800-1830_mov_4.mp4"  # 動画ファイルのパス
output_path = "output/output_frame.jpg"  # 保存する画像ファイルのパス
csv_path = "integrate_id/area_overlay.csv"  # 座標を記載したCSVファイルのパス

# 動画の1フレーム目を取得
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if ret:
    visualize_areas_with_colors(frame, csv_path, output_path)
else:
    print("Error: 動画の1フレーム目を読み込めませんでした")