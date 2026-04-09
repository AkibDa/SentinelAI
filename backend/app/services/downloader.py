# backend/app/services/downloader.py

import os
import yt_dlp

def download_video_with_ytdlp(url: str, output_dir: str) -> str:
    """Downloads a video using yt-dlp and returns the file path."""
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        return filename