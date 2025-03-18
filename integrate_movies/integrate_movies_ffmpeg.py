import os
import argparse
import subprocess
import time
import shutil

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='動画ファイルを結合し、出力FPSを3に設定するプログラム')
    parser.add_argument('--input_folder', type=str, required=True, help='動画ファイルが保存されているフォルダのパスを指定してください')
    args = parser.parse_args()

    input_folder = args.input_folder
    movie_filename = next(f for f in os.listdir(input_folder) if f.endswith('.mp4'))
    movie_filename_date_parts = movie_filename.split("_")[1]
    output_file = f'20{movie_filename_date_parts[:2]}-{movie_filename_date_parts[2:6]}_0800-1830_mov_4.mp4'
    output_foldername = "integrated_movies"
    output_folder_path = os.path.join(os.path.join("/mnt","s3"), output_foldername)
    # output_folder_path = os.path.join("/mnt","efs","efs",output_foldername)
    output_path = os.path.join(output_folder_path, output_file)

    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # フォルダが存在するか確認
    if not os.path.exists(input_folder):
        print(f'エラー: 指定されたフォルダ "{input_folder}" は存在しません。')
        exit()

    # フォルダ内の動画ファイルを取得
    video_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.mp4')]
    if not video_files:
        print(f'エラー: フォルダ "{input_folder}" 内に動画ファイルが見つかりません。')
        exit()

    # 統合対象ファイルが21個か確認
    if not len(video_files) == 21:
        print("統合対象ファイル数が21ではありません")
    print(f"対象ファイル数：{len(video_files)}")

    # 動画ファイルの時間を取得するコード
    total_duration = 0
    for video_file in video_files:
        # FFmpegを使用して動画ファイルの時間を取得
        duration_command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_file]
        try:
            result = subprocess.run(duration_command, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            total_duration += duration
        except subprocess.CalledProcessError as e:
            print(f"エラー: {video_file} の時間取得に失敗しました。")

    if not 37500 < total_duration < 38100:
        print("統合対象ファイルの合計時間が10時間25～35分ではありません")
    print(f"全動画ファイルの合計時間: {total_duration}秒")

    # ファイル名でソート
    video_files.sort()

    # FFmpeg用の一時リストファイル作成
    with open('file_list.txt', 'w') as f:
        for video_file in video_files:
            f.write(f"file '{os.path.abspath(video_file)}'\n")

    # FFmpegコマンドで結合とFPS設定を実行
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # 上書きを許可
        '-f', 'concat',  # ファイルリストを結合
        '-safe', '0',
        '-i', 'file_list.txt',
        '-vf', 'fps=30',  # FPS設定
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',  # 画質調整
        output_file
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f'結合完了: {output_file}')

        # 処理完了後にファイルを移動
        # shutil.move(output_file, output_path)
        shutil.copy2(output_file, output_path)
        os.remove(output_file)
        print(f'最終出力先: {output_path}')

        # 出力ファイルの長さを確認
        duration_command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', output_path]
        result = subprocess.run(duration_command, capture_output=True, text=True, check=True)
        output_duration = float(result.stdout.strip())
        if not 37500 < total_duration < 38100:
            print("統合対象ファイルの合計時間が10時間25～35分ではありません")
        print(f'出力ファイルの長さ: {output_duration}秒')

    except subprocess.CalledProcessError as e:
        print(f'エラー: {e}')
    finally:
        # 一時ファイル削除
        if os.path.exists('file_list.txt'):
            os.remove('file_list.txt')

    # キャッシュを削除
    cache_path = '/var/s3_cache'
    if os.path.exists(cache_path):
        for filename in [f for f in os.listdir(cache_path) if f.startswith('.')]:
            original_movies_path = os.path.join(cache_path, filename, "original_movies")
            if not os.path.exists(original_movies_path):
                continue
            for day_folder in os.listdir(original_movies_path):
                if day_folder != os.path.basename(input_folder):
                    continue
                day_folder_path = os.path.join(original_movies_path, day_folder)
                try:
                    if os.path.isfile(day_folder_path) or os.path.islink(day_folder_path):
                        os.unlink(day_folder_path)
                    elif os.path.isdir(day_folder_path):
                        shutil.rmtree(day_folder_path)
                except Exception as e:
                    print(f"Failed to delete {day_folder_path}. Reason: {e}")
    else:
        print(f"キャッシュパス '{cache_path}' が見つかりません。")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f'処理時間: {end_time - start_time}秒')

