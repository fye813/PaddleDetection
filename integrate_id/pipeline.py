import pandas as pd
import numpy as np
import time
import yaml
import os
from tqdm import tqdm
import argparse

current_dir = os.path.dirname(__file__)

def main():
    input_folder_name = "output"
    target_filename = "time_tracking_data.csv"

    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="Process area_before or area_after CSV files.")
    parser.add_argument("mode", choices=["before", "after"], help="Specify whether to use area_before or area_after")
    args = parser.parse_args()
    area_file_path = os.path.join(current_dir, f"area_{args.mode}.csv")

    # 統合設定を取得
    general_settings = load_config("general")
    max_frame_diff = general_settings['max_frame_diff']
    threshold = general_settings['threshold']
    min_detection_duration = general_settings['min_detection_duration']

    # # 入力フォルダとファイルのパスを設定
    input_path = os.path.join(current_dir, "..", input_folder_name)
    input_file_dirs = [os.path.join(input_path, x) for x in os.listdir(input_path) if not os.path.isfile(os.path.join(input_path, x))]

    for input_file_dir in input_file_dirs:
        input_file_path = os.path.join(input_file_dir, target_filename)
        print(f"処理開始: {input_file_path}")

        # CSVファイルを読み込み、空行を削除
        df = pd.read_csv(input_file_path).dropna(how='all')

        # 列名の前後に空白がないか確認して削除
        df.columns = df.columns.str.strip()
        total_data_points = len(df)
        print(f"データポイントの総数: {total_data_points}")

        # オリジナルIDの保持
        df["original_ID"] = df["Detection ID"]

        # 入力データの列の順序を保存
        original_columns = df.columns.tolist()

        # 設定情報をDataFrameに保持
        df["max_frame_diff"] = max_frame_diff
        df["threshold"] = threshold
        df["min_detection_duration"] = min_detection_duration

        # 入力データの 'Center X' と 'Center Y' のデータ型を取得
        center_x_dtype = df['Center X'].dtype
        center_y_dtype = df['Center Y'].dtype

        print("------------------補完処理開始------------------")
        processing_area_df = fill_missing_detections(df)
        print("------------------補完処理終了------------------")

        print("------------------クレンジング処理開始------------------")
        processing_area_df = cleanse_data(processing_area_df, min_detection_duration)
        print("------------------クレンジング処理終了------------------")

        # processing_area_df = assign_motion_flag(processing_area_df)
        # write_to_csv(processing_area_df,input_file_dir,"flagged_data")

        print("------------------統合処理開始------------------")
        processing_area_df = merge_similar_detections(processing_area_df, max_frame_diff, threshold)
        print("------------------統合処理終了------------------")

        # 'Center X' と 'Center Y' のデータ型を元に戻す
        processing_area_df['Center X'] = processing_area_df['Center X'].astype(center_x_dtype)
        processing_area_df['Center Y'] = processing_area_df['Center Y'].astype(center_y_dtype)

        # 列の順序を入力時と同じに並び替える
        processing_area_df = processing_area_df[original_columns]

        print("------------------エリア付与開始------------------")
        area_settings = load_area_settings(area_file_path)
        processing_area_df = assign_areas(processing_area_df, area_settings)

        # Place列を指定された位置に移動（Center YとScoreの間）
        cols = processing_area_df.columns.tolist()
        place_index = cols.index("Center Y") + 1
        cols.insert(place_index, cols.pop(cols.index("Place")))
        processing_area_df = processing_area_df[cols]
        print("------------------エリア付与終了------------------")

        print("------------------秒数に基づく除外処理開始------------------")
        # IDごとの滞在時間を算出
        processing_area_df = calc_duration(processing_area_df)

        # この秒数以内のデータは除外する
        threshold_sec = 180
        processing_area_df = processing_area_df[processing_area_df["Duration"] > threshold_sec]
        print("ユニークID数:",len(processing_area_df["Detection ID"].unique()))
        print("------------------秒数に基づく除外処理終了------------------")

        # CSVファイルへの書き出し
        write_to_csv(processing_area_df,input_file_dir,"excluded_data")

        print("------------------エリアごとの滞在時間出力開始------------------")
        # エリアごとの滞在時間を算出
        area_stay_time_df = aggregate_area_stay_time(processing_area_df)

        # PlaceごとのユニークなDetection ID数を表示
        unique_detection_counts = area_stay_time_df.groupby("Place")["Detection ID"].nunique()
        print("PlaceごとのユニークなDetection ID数:",unique_detection_counts)
        print("ユニークID数:",len(area_stay_time_df["Detection ID"].unique()))
        print("------------------エリアごとの滞在時間出力終了------------------")

        # CSVファイルへの書き出し
        write_to_csv(area_stay_time_df,input_file_dir,"area_stay_time_data")


# 設定読み込みの関数化
def load_config(section=None):
    yaml_filename = 'config.yaml'
    yaml_path = os.path.join(current_dir, yaml_filename)

    with open(yaml_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 引数で指定されたセクションのみ返す
    if section == 'general':
        general_settings = config['settings']['general']
        return general_settings
    elif section == 'areas':
        areas = config['areas']
        return areas
    elif section == 'exclusion_areas':
        areas = config['exclusion_areas']
        return areas
    else:
        # セクションが指定されない場合、全ての設定を返す
        general_settings = config['settings']['general']
        areas = config['areas']
        return general_settings, areas

# area.csvファイルの読み込み
def load_area_data():
    area_data_filename = "area.csv"
    area_data_path = os.path.join(current_dir, area_data_filename)

    area_df = pd.read_csv(area_data_path, usecols=['エリア名', '開始x座標', '開始y座標', '終了x座標', '終了y座標'])

    # 小さい方を開始座標に、大きい方を終了座標にする
    area_df['開始x座標'], area_df['終了x座標'] = area_df[['開始x座標', '終了x座標']].min(axis=1), area_df[['開始x座標', '終了x座標']].max(axis=1)
    area_df['開始y座標'], area_df['終了y座標'] = area_df[['開始y座標', '終了y座標']].min(axis=1), area_df[['開始y座標', '終了y座標']].max(axis=1)

    return area_df

def classify_by_coordinates(df, areas):
    # 各エリアのデータを格納するリスト
    filtered_dfs = []

    for area_name, coordinates in areas.items():
        # 'min' と 'max' を動的に処理
        x_start = coordinates.get('x_start')
        x_end = coordinates.get('x_end')
        y_start = coordinates.get('y_start')
        y_end = coordinates.get('y_end')

        # 'min' の場合は float('-inf')、'max' の場合は float('inf') に置き換え
        x_start = float('-inf') if x_start == 'min' else x_start
        x_end = float('inf') if x_end == 'max' else x_end
        y_start = float('-inf') if y_start == 'min' else y_start
        y_end = float('inf') if y_end == 'max' else y_end

        # 数値型でない場合のチェック
        if not isinstance(x_start, (int, float)):
            raise ValueError(f"Invalid value for x_start in {area_name}: {x_start}")
        if not isinstance(x_end, (int, float)):
            raise ValueError(f"Invalid value for x_end in {area_name}: {x_end}")
        if not isinstance(y_start, (int, float)):
            raise ValueError(f"Invalid value for y_start in {area_name}: {y_start}")
        if not isinstance(y_end, (int, float)):
            raise ValueError(f"Invalid value for y_end in {area_name}: {y_end}")

        # フィルタリング条件を設定
        condition = (
            (df["Center X"] >= x_start) & (df["Center X"] <= x_end) &
            (df["Center Y"] >= y_start) & (df["Center Y"] <= y_end)
        )

        # 各エリアに該当するデータをフィルタリングしてリストに追加
        filtered_dfs.append(df[condition])

    # リスト内のすべてのデータフレームを1つに結合して返す
    return pd.concat(filtered_dfs, ignore_index=True).sort_values(["Elapsed Seconds", "Detection ID"])


def assign_motion_flag(df, threshold=20, future_frames=5):
    # 時間順にソート
    df = df.sort_values(by=['Elapsed Seconds', 'Detection ID']).reset_index(drop=True)

    # 'next_5frames_center_x' と 'next_5frames_center_y' を初期化
    df['next_5frames_center_x'] = np.nan
    df['next_5frames_center_y'] = np.nan

    # tqdmを使ってDetection ID毎の処理を進捗表示
    for detection_id in tqdm(df['Detection ID'].unique(), desc="フラグ付与"):
        id_df = df[df['Detection ID'] == detection_id]

        # tqdmを使って各フレームに対して処理を進捗表示
        for i, row in tqdm(id_df.iterrows(), total=id_df.shape[0], desc=f"{detection_id} - フレーム処理", leave=False):
            # 現在のフレームのElapsed Seconds
            current_time = row['Elapsed Seconds']

            # 現在のフレームから次の5フレーム分を取得
            future_frames_df = id_df[(id_df['Elapsed Seconds'] > current_time) &
                                      (id_df['Elapsed Seconds'] <= current_time + future_frames)]

            # 次の5フレーム分のX座標とY座標の平均を計算
            if len(future_frames_df) > 0:
                future_avg_x = future_frames_df['Center X'].mean()
                future_avg_y = future_frames_df['Center Y'].mean()
                # 平均値をnext_5frames_center_x, next_5frames_center_yに設定
                df.loc[i, 'next_5frames_center_x'] = future_avg_x
                df.loc[i, 'next_5frames_center_y'] = future_avg_y
            else:
                # 次の5フレームが不足している場合は -1 を設定
                df.loc[i, 'next_5frames_center_x'] = -1
                df.loc[i, 'next_5frames_center_y'] = -1

            # next_5frames_center_x または next_5frames_center_y が -1 の場合は "cant judge"
            if df.loc[i, 'next_5frames_center_x'] == -1 or df.loc[i, 'next_5frames_center_y'] == -1:
                df.loc[i, 'motion_flag'] = 'cant judge'
                df.loc[i, 'distance'] = -1
            else:
                # 現在のフレームと次の平均座標との差分を計算
                distance = np.linalg.norm([row['Center X'] - df.loc[i, 'next_5frames_center_x'], row['Center Y'] - df.loc[i, 'next_5frames_center_y']])
                df.loc[i, 'distance'] = distance

                # 差分が閾値以下ならstayingフラグを立てる
                if distance <= threshold:
                    df.loc[i, 'motion_flag'] = 'staying'
                else:
                    df.loc[i, 'motion_flag'] = 'moving'

    return df

def merge_similar_detections(df, max_frame_diff, threshold):
    print("許容フレーム差", max_frame_diff)
    print(f"閾値", threshold)

    # データを時系列順にソート
    df = df.sort_values(by=['Elapsed Seconds', 'Detection ID']).reset_index(drop=True)

    # 処理するためにユニークなIDリストを作成
    unique_ids = df['Detection ID'].unique()

    integrated_ids = []
    integrate_cnt = 0

    # 並列処理、tqdmを使って進捗バーを表示
    results = []
    for detection_id in tqdm(unique_ids, desc="Processing IDs"):
        if detection_id not in integrated_ids:
            result = process_detection_id((detection_id, df, max_frame_diff, threshold, integrated_ids))
            if result is not None:
                results.append(result)

    # 統合結果を反映
    for result in results:
        target_id = result['target_id']
        detection_id = result['detection_id']
        distance = result['distance']
        frame_diff = result['frame_diff']

        df.loc[df['Detection ID'] == target_id, 'Detection ID'] = detection_id
        if 'distance' not in df.columns:
            df['distance'] = np.nan
            df['frame_diff'] = np.nan
        integrated_record_id = df[df['original_ID'] == target_id]['Elapsed Seconds'].idxmin()
        df.loc[integrated_record_id, "distance"] = distance
        df.loc[integrated_record_id, "frame_diff"] = frame_diff
        integrate_cnt += 1
        integrated_ids.append(target_id)  # 統合されたIDを追加
        integrated_ids.append(detection_id)  # 統合先のIDも追加

    print("統合回数", integrate_cnt)

    return df

# 並列処理用の関数
def process_detection_id(args):
    detection_id, df, max_frame_diff, threshold, integrated_ids = args

    current_id_df = df[df['Detection ID'] == detection_id]

    # current_id_df が空の場合はスキップ
    if current_id_df.empty: return None

    # 最後の出現フレームと位置を取得
    last_frame = current_id_df['Elapsed Seconds'].max()
    last_position = current_id_df[['Center X', 'Center Y']].iloc[-1].values

    # 一定時間内に出現した他のIDを取得
    potential_ids_df = df[(df['Elapsed Seconds'] > last_frame) & (df['Elapsed Seconds'] <= last_frame + max_frame_diff)]

    for target_id in potential_ids_df['Detection ID'].unique():
        # 統合済みのIDだった場合はスキップ
        if target_id in integrated_ids: continue

        target_id_df = potential_ids_df[potential_ids_df['Detection ID'] == target_id]
        # target_id_df が空の場合はスキップ
        if target_id_df.empty: continue

        # 新しいIDの最初の登場フレームと位置を取得
        target_id_first_frame = target_id_df['Elapsed Seconds'].min()
        first_position_target_id = target_id_df[['Center X', 'Center Y']].iloc[0].values

        # フレーム間の差を計算
        frame_diff = target_id_first_frame - last_frame

        # フレームの差が最大許容範囲を超える場合はスキップ
        if frame_diff > max_frame_diff: continue

        # フレーム間の移動許容距離を計算 (フレーム数 × 1フレームあたりの平均移動距離)
        allowed_distance = frame_diff * threshold

        # 距離を計算
        distance = np.sqrt(np.sum((last_position - first_position_target_id) ** 2))
        # 許容距離より大きければ統合しない
        if distance > allowed_distance: continue

        # 統合された後の更新情報を返す
        integrated_ids.append(target_id)  # 統合対象IDをセットに追加
        integrated_ids.append(detection_id)  # 統合先のIDもセットに追加

        return {
            'target_id': target_id,
            'detection_id': detection_id,
            'distance': distance,
            'frame_diff': frame_diff
        }

    return None

def fill_missing_detections(df, max_gap=100):
    # 補完後のデータを格納するリスト
    filled_df_list = []

    # 総補完回数を記録
    total_interpolated_points = 0

    # 入力データの 'Center X' と 'Center Y' のデータ型を取得
    center_x_dtype = df['Center X'].dtype
    center_y_dtype = df['Center Y'].dtype

    # ベースのdatetimeを計算
    min_elapsed = df['Elapsed Seconds'].min()
    reference_row = df[df['Elapsed Seconds'] == min_elapsed].iloc[0]
    base_datetime = pd.to_datetime(reference_row['datetime'], errors='coerce') - pd.to_timedelta(min_elapsed, unit='s')

    for detection_id in tqdm(df['Detection ID'].unique(), desc="補完処理中"):
        id_df = df[df['Detection ID'] == detection_id].copy()
        id_min_elapsed = int(id_df['Elapsed Seconds'].min())
        id_max_elapsed = int(id_df['Elapsed Seconds'].max())

        # Detection IDごとに存在するElapsed Secondsの範囲で再インデックス
        id_df.set_index('Elapsed Seconds', inplace=True)
        original_index = id_df.index.copy()  # 元のインデックスを保存
        id_df = id_df.reindex(range(id_min_elapsed, id_max_elapsed + 1))

        # 'Detection ID' と 'original_ID' を設定
        id_df['Detection ID'] = detection_id
        id_df['original_ID'] = detection_id

        # 補完前の欠損値の数をカウント
        nans_before_interpolation = id_df['Center X'].isna().sum()

        # 補完したデータかどうかを示す列を追加（初期値は False）
        id_df['is_interpolated'] = False

        # 欠損値を補完
        id_df['Center X'] = id_df['Center X'].interpolate(method='linear', limit=max_gap, limit_area='inside')
        id_df['Center Y'] = id_df['Center Y'].interpolate(method='linear', limit=max_gap, limit_area='inside')

        # 補完後のインデックスを取得
        new_index = id_df.index

        # 新たに追加された行を特定（元のインデックスに無い行）
        interpolated_rows = new_index.difference(original_index)

        # 新しく補完された行に is_interpolated を True に設定
        id_df.loc[interpolated_rows, 'is_interpolated'] = True

        # 補完後の欠損値の数をカウント
        nans_after_interpolation = id_df['Center X'].isna().sum()

        # 実際に補完されたデータポイント数を計算
        interpolated_points = nans_before_interpolation - nans_after_interpolation

        # その他の列を前の行からコピー
        id_df = id_df.ffill(limit=max_gap)

        # 'Center X' と 'Center Y' を指定した小数点以下の桁数に丸める
        id_df['Center X'] = id_df['Center X'].round(2).astype(center_x_dtype)
        id_df['Center Y'] = id_df['Center Y'].round(2).astype(center_y_dtype)

        # datetimeをElapsed Secondsから計算
        id_df['datetime'] = id_df.index.to_series().apply(
            lambda x: (base_datetime + pd.to_timedelta(x, unit='s')).strftime('%Y/%m/%d %H:%M:%S')
        )

        # 補完後のデータをリストに追加
        id_df.reset_index(inplace=True)
        filled_df_list.append(id_df)

        # 総補完回数を更新
        total_interpolated_points += interpolated_points

    # リスト内のデータフレームを結合
    filled_df = pd.concat(filled_df_list, ignore_index=True)

    print(f"補完可能な時間（秒）: {max_gap}")
    print(f"補完回数（補完したデータポイント数）: {total_interpolated_points}")

    return filled_df

# クレンジング処理: 出現が短いID（数フレームしか現れないゴミデータ）を削除
def cleanse_data(df, min_duration):
    print(f"データを有効と見なす最小の出現フレーム数",min_duration)

    id_durations = df.groupby('Detection ID')['Elapsed Seconds'].nunique()
    valid_ids = id_durations[id_durations >= min_duration].index
    cleansed_df = df[df['Detection ID'].isin(valid_ids)]

    print(f"除去回数（除去したデータポイント数）: {len(df) - len(cleansed_df)}")
    return cleansed_df

def load_area_settings(file_path):
    """エリア設定をCSVから読み込む"""
    area_df = pd.read_csv(file_path)
    area_settings = {}
    for _, row in area_df.iterrows():
        area_name = str(row["エリア名"])
        area_settings[area_name] = {
            "x_min": row["開始x座標"],
            "x_max": row["終了x座標"],
            "y_min": row["開始y座標"],
            "y_max": row["終了y座標"]
        }
    return area_settings

def assign_areas(df, area_settings):
    # XY座標に基づいて、Place列の値を設定
    def classify_area(x, y, settings):
        """XY座標に基づいてエリアを分類"""
        for area_name, bounds in settings.items():
            if bounds["x_min"] <= x <= bounds["x_max"] and bounds["y_min"] <= y <= bounds["y_max"]:
                return area_name
        return "none"

    # XY座標に基づいてエリアを分類
    df["Place"] = df.apply(lambda row: classify_area(row["Center X"], row["Center Y"], area_settings), axis=1)

    return df

def aggregate_area_stay_time(df):
    # datetimeカラムをdatetime型に変換
    df['datetime'] = pd.to_datetime(df['datetime'])
    # Detection IDとPlaceごとの最初と最後の時間を取得
    area_stay_time_df = df.groupby(['Detection ID', 'Place']).agg({
        'datetime': ['min', 'max']
    }).reset_index()

    # カラム名を分かりやすく変更
    area_stay_time_df.columns = ['Detection ID', 'Place', 'Start Time', 'End Time']
    # 滞在時間を計算
    area_stay_time_df['Area Stay Time'] = (area_stay_time_df['End Time'] - area_stay_time_df['Start Time']).dt.total_seconds()
    area_stay_time_df = area_stay_time_df.sort_values(["Detection ID",'Start Time'])

    return area_stay_time_df

# Detection IDごとに、対象エリア内の滞在時間がthreshold_sec未満のデータを削除
def calc_duration(df):
    # Detection ID と datetime を基準に重複を排除
    df_unique = df.drop_duplicates(subset=["Detection ID", "datetime"])

    # Detection ID ごとのレコード数を Duration 列に追加
    df["Duration"] = df_unique.groupby("Detection ID")["Detection ID"].transform("count")

    return df

# 加工結果を新しいCSVファイルに出力
def write_to_csv(df, output_dir, filename):
    vis_path = os.path.join(output_dir, f"{filename}.csv")
    df.to_csv(vis_path, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()