import cv2
import pandas as pd
import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image, ImageDraw, ImageFont

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
    overlay_with_text = frame.copy()
    pil_image = Image.fromarray(cv2.cvtColor(overlay_with_text, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = ImageFont.truetype(font_path, 30)  # フォントサイズを大きく設定

    # 各エリアを描画
    for _, row in df.iterrows():
        # 座標と色を取得
        x_start, y_start = int(row['開始x座標']), int(row['開始y座標'])
        x_end, y_end = int(row['終了x座標']), int(row['終了y座標'])
        color_name = row['色']
        area_name = row['エリア名'] if 'エリア名' in row else ""

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

        # 枠線を黒色で描画
        cv2.polylines(overlay, [points], isClosed=True, color=(0, 0, 0), thickness=2)

        # 領域名を描画 (領域の内側、上側に沿う形で描画)
        if area_name:
            text_x = x_start + 5  # 領域の左端に揃える
            text_y = y_start + 5  # 領域の上端に揃える
            draw.text((text_x, text_y), area_name, fill="black", font=font)  # 黒字で描画

    # オーバーレイを適用
    overlay_with_text_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    result = cv2.addWeighted(overlay, alpha, overlay_with_text_bgr, 1 - alpha, 0)

    # テキスト描画（最後に直接フレームに適用、透明度なし）
    pil_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    for _, row in df.iterrows():
        # 座標と領域名
        x_start, y_start = int(row['開始x座標']), int(row['開始y座標'])
        area_name = row['エリア名'] if 'エリア名' in row else ""
        
        if area_name:
            text_x = x_start + 5
            text_y = y_start + 5

            # 黒字を最後に描画
            draw.text((text_x, text_y), area_name, fill="black", font=font)

    # 最終結果を保存
    final_result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 保存
    cv2.imwrite(output_path, final_result)
    print(f"画像を保存しました: {output_path}")

# 使用例
video_path = "input/2024-1113_0800-1830_mov_4.mp4"  # 動画ファイルのパス
output_path = "output/output_frame.jpg"  # 保存する画像ファイルのパス
csv_path = "integrate_id/area_after_color.csv"  # 座標を記載したCSVファイルのパス

# 動画の1フレーム目を取得
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if ret:
    visualize_areas_with_colors(frame, csv_path, output_path)
else:
    print("Error: 動画の1フレーム目を読み込めませんでした")
