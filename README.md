# WhisperSUB_V

WhisperSUB_V là phần mềm tạo phụ đề `.srt` từ video/audio bằng `whisper.cpp` và có SRT Editor để chỉnh tay.

## Thành phần bắt buộc để app chạy
Đặt đúng file vào đúng thư mục như sau:

```text
WhisperSUB_V/
├─ WhisperSUB_V.py
├─ WhisperSUB_V.spec
├─ README.md
├─ models/
│  └─ ggml-large-v3.bin
├─ Release/
│  ├─ whisper-cli.exe (hoặc main.exe)
│  ├─ ffmpeg.exe
│  ├─ mpv.exe
│  └─ silero_vad.bin / silero_v5.1.2.bin / silero_vad.onnx
└─ docs/images/ (tuỳ chọn, để chứa screenshot)
```

## Link tải chính thức
- whisper.cpp releases: https://github.com/ggml-org/whisper.cpp/releases
- Model `ggml-large-v3.bin`: https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin?download=true
- FFmpeg download: https://www.ffmpeg.org/download.html
- MPV releases: https://github.com/mpv-player/mpv/releases
- Whisper VAD model (`silero_v5.1.2.bin`): https://huggingface.co/ggml-org/whisper-vad/resolve/main/silero_v5.1.2.bin?download=true
- Silero VAD repo: https://github.com/snakers4/silero-vad

## Tính năng
- Xuất `.srt` từ nhiều định dạng media.
- Hỗ trợ GPU/CPU.
- VAD giảm lặp câu.
- SRT Editor tích hợp (sửa text/timeline, thêm/xóa dòng).
- Advanced mode + preset ngôn ngữ.
- Nút `Tai file thieu` để tải nhanh thành phần còn thiếu.

## Cách dùng nhanh
1. Chạy:
```powershell
python WhisperSUB_V.py
```
2. Nếu thiếu file, bấm `Tai file thieu` trong app.
3. Dán link tải cho từng thành phần rồi bấm `Tai ngay`.
4. Chọn file media và bấm `Xuat SRT`.

## Build EXE
```powershell
pyinstaller --noconfirm --clean WhisperSUB_V.spec
```
Output: `dist/WhisperSUB_V/WhisperSUB_V.exe`

## Lưu ý
- Nút tải tự động hoạt động tốt nhất với URL file trực tiếp hoặc file `.zip`.
- Với URL trang chủ release/download, bạn cần mở trang đó và lấy link file trực tiếp phù hợp Windows x64.
- App hiện xuất chính `.srt`.

---

# English

WhisperSUB_V generates `.srt` subtitles from video/audio using `whisper.cpp` and includes a built-in subtitle editor.

## Required files/folders
Place files exactly like this:

```text
WhisperSUB_V/
├─ WhisperSUB_V.py
├─ WhisperSUB_V.spec
├─ models/ggml-large-v3.bin
└─ Release/
   ├─ whisper-cli.exe (or main.exe)
   ├─ ffmpeg.exe
   ├─ mpv.exe
   └─ silero_vad.bin / silero_v5.1.2.bin / silero_vad.onnx
```

## Download links
- whisper.cpp releases: https://github.com/ggml-org/whisper.cpp/releases
- ggml-large-v3 model: https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin?download=true
- FFmpeg: https://www.ffmpeg.org/download.html
- MPV: https://github.com/mpv-player/mpv/releases
- Whisper VAD model: https://huggingface.co/ggml-org/whisper-vad/resolve/main/silero_v5.1.2.bin?download=true

## Run
```powershell
python WhisperSUB_V.py
```

## Build EXE
```powershell
pyinstaller --noconfirm --clean WhisperSUB_V.spec
```
