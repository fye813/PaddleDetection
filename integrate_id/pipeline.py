import pandas as pd
import numpy as np
import time
import yaml
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

current_dir = os.path.dirname(__file__)

def main():
    input_folder_name = "output"  # 読み込むフォルダ名 人検知プログラムの結果出力先
    target_filename = "time_tracking_data.csv"  # 使用するファイル名

    # 統合設定を取得
    general_settings = load_config("general")
    max_frame_diff = general_settings['max_frame_diff']
    threshold = general_settings['threshold']
    min_detection_duration = general_settings['min_detection_duration']

    # 除外エリアの読み込み
    exclusion_area_settings = load_config("exclusion_areas")
    area_df = load_area_data()
    exclusion_areas = area_df[area_df["エリア名"].isin(exclusion_area_settings)][['開始x座標', '開始y座標', '終了x座標', '終了y座標']].values.tolist() if exclusion_area_settings else None

    # # 人検知プログラムの結果フォルダごとに実行
    input_path = os.path.join(current_dir, "..", input_folder_name)
    input_file_dirs = [os.path.join(input_path, x) for x in os.listdir(input_path) if not os.path.isfile(os.path.join(input_path, x))]

    for input_file_dir in input_file_dirs:
        input_file_path = os.path.join(input_file_dir, target_filename)

        # CSVファイルを読み込み、空行を削除
        df = pd.read_csv(input_file_path).dropna(how='all')

        # 列名の前後に空白がないか確認して削除
        df.columns = df.columns.str.strip()

        # オリジナルIDの保持
        df["original_ID"] = df["Detection ID"]

        # 設定情報をDataFrameに保持
        df["max_frame_diff"] = max_frame_diff
        df["threshold"] = threshold
        df["min_detection_duration"] = min_detection_duration

        # 除外エリアのデータを除外
        print("jogai")
        if exclusion_areas:
            df = exclude_data_by_xy_ranges(df, exclusion_areas)

        # # エリア設定を取得
        # area_settings = load_config("areas")
        # processing_area = area_settings['processing']
        # no_processing_area = area_settings['no_processing']

        # # DataFrameをxy座標ごとに分類
        # processing_area_df = classify_by_coordinates(df, processing_area)
        # no_processing_area_df = classify_by_coordinates(df, no_processing_area)

        # 結合対象のDataFrameに対する結合処理----------------------------------------------------------------------------------
        # 同一ID情報の補完：同一IDにおいて、最初に現れた時間と最後に現れた時間の間の欠損値を補完して、連続情報にする
        # processing_area_df = fill_missing_detections(processing_area_df)
        print("hokan")
        processing_area_df = fill_missing_detections(df)

        # 統合処理：あるIDが消失してから〇秒後に出現したIDの中で、いずれも一定距離内にいた場合、同一IDと見なして情報を統合する
        print("tougou")
        processing_area_df = merge_similar_detections(processing_area_df, max_frame_diff, threshold)

        # クレンジング処理
        print(f"データを有効と見なす最小の出現フレーム数",min_detection_duration)
        processing_area_df = cleanse_data(processing_area_df, min_detection_duration)
        # no_processing_area_df = cleanse_data(no_processing_area_df, min_detection_duration)

        # 同一人物の統合確認
        # confirm_results(processing_area_df)

        # CSVファイルへの書き出し
        write_to_csv(processing_area_df,input_file_dir,"processing")
        # write_to_csv(no_processing_area_df,input_file_dir,"no_processing")

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

def load_area_data():
    area_data_filename = "area.csv"
    area_data_path = os.path.join(current_dir, area_data_filename)
    # area.csvファイルの読み込み
    area_df = pd.read_csv(area_data_path, usecols=['エリア名', '開始x座標', '開始y座標', '終了x座標', '終了y座標'])

    # 小さい方を開始座標に、大きい方を終了座標にする
    area_df['開始x座標'], area_df['終了x座標'] = area_df[['開始x座標', '終了x座標']].min(axis=1), area_df[['開始x座標', '終了x座標']].max(axis=1)
    area_df['開始y座標'], area_df['終了y座標'] = area_df[['開始y座標', '終了y座標']].min(axis=1), area_df[['開始y座標', '終了y座標']].max(axis=1)

    return area_df



# 現在判明している同一人物情報から結果を確認する
def confirm_results(df):
    integrate_lists = [
        [12,313],
        [8,425],
        [25,160],
        [29,132],
        [18,134,192],
        [56,67],
    ]
    # integrate_lists = [
    #     [25,39,80,104,107,149],
    #     [12,120,296,346,394],
    #     [9,219,279,443,500],
    #     [1,534],
    # ]

    for integrate_list in integrate_lists:
        print("--------------")
        detection_ids = [df.loc[df['original_ID'] == id, 'Detection ID'].values[0] if not df.loc[df['original_ID'] == id].empty else None for id in integrate_list]
        for i,(original_id,detection_id) in enumerate(zip(integrate_list,detection_ids)):
            if i == 0:
                print("base:",original_id)
            else:
                print(original_id,detection_id,"◯" if detection_id==integrate_list[0] else "✕")

# 指定の範囲内のデータを除外する
def exclude_data_by_xy_ranges(df, xy_ranges):
    # 各範囲に対するマスクを作成
    masks = []
    for x_min, y_min, x_max, y_max in xy_ranges:
        mask = (
            (df['Center X'] < x_min) | (df['Center X'] > x_max) |
            (df['Center Y'] < y_min) | (df['Center Y'] > y_max)
        )
        masks.append(mask)

    # すべてのマスクを組み合わせる
    final_mask = np.logical_and.reduce(masks)

    # マスクを適用してデータをフィルタリング
    filtered_df = df[final_mask]

    return filtered_df

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


# 統合処理：あるIDが消失してから〇フレーム後に出現したIDの中で、いずれも一定距離内にいた場合、同一IDと見なして情報を統合する
def merge_similar_detections(df, max_frame_diff,threshold):

    print("許容フレーム差",max_frame_diff)
    print(f"閾値",threshold)

    # データを時系列順にソート
    df = df.sort_values(by=['Elapsed Seconds', 'Detection ID']).reset_index(drop=True)

    # 処理するためにユニークなIDリストを作成
    unique_ids = df['Detection ID'].unique()

    integrated_ids = []
    integrate_cnt = 0

    # 並列処理、tqdmを使って進捗バーを表示
    with Pool(cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_detection_id, 
                      [(detection_id, df, max_frame_diff, threshold, integrated_ids) for detection_id in unique_ids]),
            total=len(unique_ids), 
            desc="Processing IDs"
        ))

    # 統合結果を反映
    for result in results:
        if result is None:
            continue
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
    
    print("統合回数",integrate_cnt)
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
    potential_ids_df = df[df['Elapsed Seconds'] >= last_frame]

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

        # フレームの差が0以下の場合はスキップ
        if frame_diff <= 0: continue

        # フレーム間の移動許容距離を計算 (フレーム数 × 1フレームあたりの平均移動距離)
        allowed_distance = frame_diff * threshold

        # 距離を計算
        distance = np.sqrt(np.sum((last_position - first_position_target_id) ** 2))
        # 許容距離より大きければ統合しない
        if distance > allowed_distance: continue

        # 統合された後の更新情報を返す
        return {
            'target_id': target_id,
            'detection_id': detection_id,
            'distance': distance,
            'frame_diff': frame_diff
        }

    return None

# 同一ID情報の補完：同一IDにおいて、最初に現れたフレームと最後に現れたフレームの間の欠損値を補完して、連続情報にする
def fill_missing_detections(df):
    unique_ids = df['Detection ID'].unique()
    filled_df = pd.DataFrame()

    for detection_id in unique_ids:
        id_df = df[df['Detection ID'] == detection_id].sort_values('Elapsed Seconds')

        # 最初と最後の出現フレームを取得
        first_appearance = int(id_df['Elapsed Seconds'].min())
        last_appearance = int(id_df['Elapsed Seconds'].max())

        # 全体のフレーム範囲を生成
        full_time_range = range(first_appearance, last_appearance + 1)
        id_df = id_df.set_index('Elapsed Seconds').reindex(full_time_range)

        # 欠損値を補完（前のデータで補完、20フレーム間隔以上は補完しない）
        last_valid_index = None
        for time in full_time_range:
            if time in id_df.index and not id_df.loc[time].isnull().all():
                last_valid_index = time
            elif last_valid_index is not None and time - last_valid_index <= 10:
                id_df.loc[time] = id_df.loc[last_valid_index]
            else:
                id_df.loc[time] = np.nan

        # データフレームを元に戻す
        id_df = id_df.reset_index()
        id_df['Detection ID'] = detection_id  # ID列を保持

        # 結果を結合
        filled_df = pd.concat([filled_df, id_df])

    filled_df = filled_df.sort_values(by=['Detection ID', 'Elapsed Seconds'])
    return filled_df.dropna(how='any')

# クレンジング処理: 出現が短いID（数フレームしか現れないゴミデータ）を削除
def cleanse_data(df, min_duration):
    id_durations = df.groupby('Detection ID')['Elapsed Seconds'].nunique()
    valid_ids = id_durations[id_durations >= min_duration].index
    cleansed_df = df[df['Detection ID'].isin(valid_ids)]
    return cleansed_df

def write_to_csv(df, output_dir, filename):
    # 加工結果を新しいCSVファイルに出力
    vis_path = os.path.join(output_dir,f"{filename}.csv")
    df.to_csv(vis_path, index=False)

if __name__ == "__main__":
    main()