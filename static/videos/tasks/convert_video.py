import os
import subprocess
import json

# --- 配置区 ---

# 1. 设置你的视频所在的文件夹路径
INPUT_FOLDER = "/Users/micdz/Westlake/Traj2Action/static/videos/tasks" 

# 2. 设置处理完成后视频的保存路径
OUTPUT_FOLDER = "/Users/micdz/Westlake/Traj2Action/static/videos/tasks/output"

# 3. 定义要处理的视频文件扩展名 (小写)
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.flv')

# --- 脚本主代码 ---

def has_audio_stream(video_path):
    """
    使用 ffprobe 检查视频文件是否包含音频流。
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a', # 只选择音频流
        '-show_entries', 'stream=codec_type',
        '-of', 'json',
        video_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return len(json.loads(result.stdout).get('streams', [])) > 0
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return False

def process_video(input_path, output_path):
    """
    使用 ffmpeg 处理单个视频文件。
    - 3倍速播放 (视频和音频)
    - 设置帧率为 15 fps
    - 在左上角添加 'x3' 文字 (更大、更粗)
    """
    print(f"开始处理: {os.path.basename(input_path)}...")
    
    has_audio = has_audio_stream(input_path)

    # --- 主要修改在这里 ---
    # 基础视频滤镜链
    # fontsize 调大到 72
    # borderw=2 和 bordercolor=white 增加了一个2像素的白色边框，实现加粗效果
    # x 和 y 也相应调大，避免贴边
    video_filter = "setpts=PTS/3,drawtext=text='x3':x=20:y=20:fontcolor=white:fontsize=72:borderw=2:bordercolor=white"
    # --- 修改结束 ---

    if has_audio:
        print("  -> 检测到音频流，将同时加速音频。")
        command = [
            'ffmpeg', '-i', input_path,
            '-filter_complex', f"[0:v]{video_filter}[v];[0:a]atempo=2.0,atempo=1.5[a]",
            '-map', '[v]', '-map', '[a]',
            '-r', '15', '-y', output_path
        ]
    else:
        print("  -> 未检测到音频流，仅处理视频。")
        command = [
            'ffmpeg', '-i', input_path,
            '-vf', video_filter,
            '-r', '15', '-y', output_path
        ]
        
    try:
        subprocess.run(command, check=True)
        print(f"处理成功: {os.path.basename(output_path)}")
    except subprocess.CalledProcessError as e:
        print(f"处理失败: {os.path.basename(input_path)}")
        print(f"FFmpeg 返回了错误: {e}")
    except FileNotFoundError:
        print("错误: 'ffmpeg' 或 'ffprobe' 命令未找到。")
        print("请确保它们已正确安装并已添加到系统的 PATH 环境变量中。")
        exit()

def main():
    if not os.path.isdir(INPUT_FOLDER):
        print(f"错误: 输入文件夹 '{INPUT_FOLDER}' 不存在。请检查路径。")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print(f"正在从 '{INPUT_FOLDER}' 查找视频...")
    
    found_videos = False
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(VIDEO_EXTENSIONS):
            if filename.startswith('processed_'):
                continue
            found_videos = True
            input_file_path = os.path.join(INPUT_FOLDER, filename)
            output_file_path = os.path.join(OUTPUT_FOLDER, f"processed_{filename}")
            process_video(input_file_path, output_file_path)

    if not found_videos:
        print("在输入文件夹中没有找到任何视频文件。")

if __name__ == '__main__':
    main()