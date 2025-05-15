import os
import re
import pandas as pd
from glob import glob   

def load_subtitles_dataset(dataset_path):
    subtitles_path = glob(os.path.join(dataset_path, '*.srt'))
    scripts = []
    episode_num = []
    episode_season = []

    for path in subtitles_path:
        # Đọc file
        with open(path, 'r', encoding='utf-8') as f:
            text = f.readlines()

        lines = []  # Không khởi tạo với None
        current_captions = []
        i = 0
        while i < len(text):
            line = text[i].strip()
            if not line:
                i += 1
                continue

            if line.isdigit():
                i += 1
                if i < len(text) and ' --> ' in text[i]:  # Dòng thời gian
                    i += 1
                    caption_text = []
                    while i < len(text) and text[i].strip():
                        caption_text.append(text[i].strip())
                        i += 1
                    
                    if caption_text:
                        lines.extend(caption_text)  # Thêm trực tiếp vào `lines`
            else:
                i += 1

        script = " ".join(lines)  # Ghép nội dung lại

        match = re.search(r'S(\d+)E(\d+)', path)
        if match:
            season = int(match.group(1))
            episode = int(match.group(2))
            scripts.append(script)
            episode_num.append(episode)
            episode_season.append(season)

    df = pd.DataFrame.from_dict({"season": episode_season, "episode": episode_num, "script": scripts})
    return df
