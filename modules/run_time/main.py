import argparse
import os
import pandas as pd

def main():
    args = parse_args()

    input_folder_path = args.input_folder_path
    input_files = ["excluded_data.csv","time_tracking_data_after_id_integration.csv"]

    output_folder_path = input_folder_path
    for input_file in input_files:
        print(f"Processing {input_file}")
        # ステップ1：エリア滞在区間計算処理
        input_file_path = os.path.join(input_folder_path, input_file)
        summary_df = calculate_stay_intervals(input_file_path)

        output_filename = f"{input_file.split('.', 1)[0]}_summary.csv"
        output_file_path_summary = os.path.join(output_folder_path, output_filename)
        summary_df.to_csv(output_file_path_summary, index=False, encoding='utf-8-sig')

        # ステップ2：Duration集計処理
        # sum
        result_df = aggregate_sum(summary_df)
        output_filename = f"{input_file.split('.', 1)[0]}_sum.csv"
        output_file_path_agg = os.path.join(output_folder_path, output_filename)
        result_df.to_csv(output_file_path_agg, index=False, encoding='utf-8-sig')

        # trans
        result_df = aggregate_trans(summary_df)
        output_filename = f"{input_file.split('.', 1)[0]}_trans.csv"
        output_file_path_agg = os.path.join(output_folder_path, output_filename)
        result_df.to_csv(output_file_path_agg, index=False, encoding='utf-8-sig')

        # total
        result_df = aggregate_total(summary_df)
        output_filename = f"{input_file.split('.', 1)[0]}_total.csv"
        output_file_path_agg = os.path.join(output_folder_path, output_filename)
        result_df.to_csv(output_file_path_agg, index=False, encoding='utf-8-sig')


def parse_args():
    parser = argparse.ArgumentParser(description='Run all steps in sequence with a single top-level output directory.')

    parser.add_argument(
        '--input_folder',
        type=str, default='input', help='Input directory for the first step'
    )

    return parser.parse_args()


def calculate_stay_intervals(input_file_path):
    """エリア滞在区間計算処理"""
    df = pd.read_csv(input_file_path)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.sort_values(['Detection ID', 'datetime'])

    results = []
    for det_id, group in df.groupby('Detection ID'):
        group = group.reset_index(drop=True)
        current_area, start_time, count = None, None, 0

        for _, row in group.iterrows():
            if row['Place'] != current_area:
                if current_area is not None:
                    results.append([det_id, current_area, start_time, end_time, count])
                current_area, start_time, count = row['Place'], row['datetime'], 1
            else:
                count += 1
            end_time = row['datetime']

        if current_area is not None:
            results.append([det_id, current_area, start_time, end_time, count])

    return pd.DataFrame(results, columns=['Detection ID', 'Place', 'Start time', 'End time', 'Duration'])


def aggregate_sum(df):
    """通常の集計 (ID, PlaceごとにDuration合計)"""
    return df.groupby(['Detection ID', 'Place'], as_index=False)['Duration'].sum().rename(columns={'Duration': 'Duration(countup_noneあり)'})


def aggregate_trans(df):
    """Duration <= 10 の行を最大Duration行へ統合"""
    base_result = df.groupby(['Detection ID', 'Place'], as_index=False)['Duration'].sum()
    merged_rows = []

    for _, sub_df in base_result.groupby('Detection ID'):
        sub_df = sub_df.reset_index(drop=True)
        small_mask = sub_df['Duration'] <= 10
        small_sum = sub_df.loc[small_mask, 'Duration'].sum()
        large_df = sub_df.loc[~small_mask]

        if not large_df.empty:
            max_idx = large_df['Duration'].idxmax()
            sub_df.loc[max_idx, 'Duration'] += small_sum
            sub_df = sub_df.drop(small_mask[small_mask].index)
        elif not sub_df.empty:
            max_idx = sub_df['Duration'].idxmax()
            sub_df = sub_df.iloc[[max_idx]]
            sub_df.loc[max_idx, 'Duration'] = small_sum
        merged_rows.append(sub_df)

    return pd.concat(merged_rows, ignore_index=True).rename(columns={'Duration': 'Duration(countup_noneあり)'})


def aggregate_total(df):
    """none相当区間を考慮した総集計"""
    df['Start time'] = pd.to_datetime(df['Start time'], errors='coerce')
    df['End time'] = pd.to_datetime(df['End time'], errors='coerce')

    none_mask = df['Place'].isna() | df['Place'].str.lower().eq('null')

    total_df = df.groupby('Detection ID', as_index=False)['Duration'].sum().rename(columns={'Duration': 'Duration(countup_noneあり)'})
    none_duration_summary = df[none_mask].groupby('Detection ID', as_index=False)['Duration'].sum().rename(columns={'Duration': 'none_duration'})

    time_summary = df.groupby('Detection ID', as_index=False).agg(
        start_min=('Start time', 'min'),
        end_max=('End time', 'max')
    )
    time_summary['Duration(MAX-MIN_noneあり)'] = (time_summary['end_max'] - time_summary['start_min']).dt.total_seconds()

    none_time_df = df[none_mask].copy()
    none_time_df['interval'] = (none_time_df['End time'] - none_time_df['Start time']).dt.total_seconds()
    none_time_sum = none_time_df.groupby('Detection ID', as_index=False)['interval'].sum().rename(columns={'interval': 'none_time_sum'})

    merged_df = total_df.merge(none_duration_summary, on='Detection ID', how='left').merge(
        time_summary[['Detection ID', 'Duration(MAX-MIN_noneあり)']], on='Detection ID', how='left'
    ).merge(none_time_sum, on='Detection ID', how='left').fillna(0)

    merged_df['Duration(countup_noneなし)'] = merged_df['Duration(countup_noneあり)'] - merged_df['none_duration']
    merged_df['Duration(MAX-MIN_noneなし)'] = merged_df['Duration(MAX-MIN_noneあり)'] - merged_df['none_time_sum']

    return merged_df[['Detection ID', 'Duration(countup_noneあり)', 'Duration(countup_noneなし)', 'Duration(MAX-MIN_noneあり)', 'Duration(MAX-MIN_noneなし)']]


if __name__ == '__main__':
    main()
