import os
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="エリア定義を用いてCSVデータを処理します")

    parser.add_argument("--input_folder", required=True, help="入力フォルダのパス")
    parser.add_argument("--area_csv", help="エリア定義のExcelファイルのパス (エリア判定時に必要)")
    args = parser.parse_args()

    input_folder = args.input_folder


    input_file = "time_tracking_data.csv"
    input_file_path = os.path.join(input_folder, input_file)

    # 入力ファイルの存在確認
    if not os.path.exists(input_file_path):
        raise ValueError(f"エラー: 指定したディレクトリに{input_file} が見つかりません。")

    df = pd.read_csv(input_file_path)

    # エリア判定
    if not args.area_csv:
        raise ValueError("エリア判定を行うには --area_csv を指定してください")
    assigned_df = assign_areas(df, args.area_csv)
    output_file_path = os.path.join(input_folder, "area_data.csv")
    assigned_df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    print(f"Processed area assignment: {output_file_path}")

    # エリアごとにカウント
    counted_df = count_areas(assigned_df)
    output_file_path = os.path.join(input_folder, "count_data.csv")
    counted_df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    print(f"Processed area count: {output_file_path}")


def assign_areas(df, area_csv_path):
    """CSVファイルに対してエリア判定を行い、"Place"列を追加して保存する"""
    areas = load_areas(area_csv_path)

    df["Place"] = df.apply(
        lambda row: judge_area(row["Center X"], row["Center Y"], areas),
        axis=1
    )

    return df


def count_areas(df):
    count_df = df.groupby(["datetime", "Place"])["Detection ID"].count().reset_index(name="count ID")
    pivot_df = count_df.pivot(index="datetime", columns="Place", values="count ID").fillna(0).astype(int)
    pivot_df["Total"] = pivot_df.sum(axis=1)
    pivot_df.reset_index(inplace=True)

    return pivot_df


def load_areas(area_csv_path, sheet_name = "after"):
    df_area = pd.read_excel(area_csv_path, sheet_name=sheet_name)
    return [
        {
            "name": row["エリア名"],
            "xmin": row["開始x座標"],
            "ymin": row["開始y座標"],
            "xmax": row["終了x座標"],
            "ymax": row["終了y座標"]
        }
        for _, row in df_area.iterrows()
    ]


def judge_area(x, y, areas):
    for area in areas:
        if area["xmin"] <= x <= area["xmax"] and area["ymin"] <= y <= area["ymax"]:
            return area["name"]
    return "none"


if __name__ == "__main__":
    main()
