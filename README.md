# WhisperSUB_V (Whisper GUI)

Ứng dụng desktop dịch video/audio ra phụ đề `.srt` bằng `whisper.cpp` (`whisper-cli.exe`) và model `ggml-large-v3.bin`.

## Tính năng chính
- Giao diện đơn giản để chọn file và xuất SRT.
- Hỗ trợ GPU (CUDA/OpenVINO nếu bản build Whisper có backend tương ứng).
- Hỗ trợ VAD để giảm lặp câu.
- SRT Editor tích hợp: xem video, sửa timeline/text, thêm/xóa dòng sub.
- Save/Load progress để mở lại dự án đã làm.
- Chế độ nâng cao (Advanced mode): chỉnh tham số giải mã + preset theo ngôn ngữ.

## Cấu trúc thư mục cần thiết
Dự án được thiết kế để chạy tại thư mục gốc như sau:

- `app.py`
- `models/ggml-large-v3.bin`
- `Release/whisper-cli.exe`
- `Release/ffmpeg.exe` (hoặc ffmpeg trong PATH)
- `Release/mpv.exe` (hoặc mpv trong PATH)
- `Release/silero_vad.onnx` (nếu muốn bật VAD)
- `save_files/` (tự tạo khi chạy)

Lưu ý: app ưu tiên tìm file trong `Release/` trước, sau đó mới đến PATH hệ thống.

## Yêu cầu hệ thống
- Windows 10/11
- Python 3.10+
- Dung lượng trống đủ cho model (~3GB với large-v3)
- Nếu dùng GPU: cần bản `whisper-cli.exe` đã build hỗ trợ CUDA/OpenVINO

## Chạy ứng dụng
```powershell
python app.py
```

## Quy trình sử dụng cơ bản
1. Mở app.
2. Chọn `Whisper CLI` (`Release/whisper-cli.exe`).
3. Chọn file video/audio đầu vào.
4. Chọn ngôn ngữ nguồn (`auto`, `en`, `vi`, `ja`, `ko`, `zh`, `fr`, `de`, `es`).
5. Chọn profile cơ bản (`Nhanh`, `Cân bằng`, `Chính xác`) nếu cần.
6. Bấm `Xuất SRT`.
7. Sau khi xong, file `.srt` được lưu cùng thư mục với file media gốc (cùng tên file).

Ví dụ:
- Input: `M:\Video\sample.mp4`
- Output: `M:\Video\sample.srt`

## Chế độ nâng cao
Bấm nút `Chế độ nâng cao` trên màn hình chính.

### Nhóm tham số
- Sampling/Decoding: `temperature`, `temperature-inc`, `best-of`, `beam-size`
- Drop đoạn: `no-speech-thold`, `entropy-thold`, `logprob-thold`
- VAD: `--vad`, `vad-threshold`
- Context: `max-context`, `no-fallback`
- Language: `--language`

### Preset ngôn ngữ có sẵn
- `auto`
- `en (English)`
- `vi (Vietnamese)`
- `ja (Japanese)`
- `ko (Korean)`
- `zh (Chinese)`
- `fr (French)`
- `de (German)`
- `es (Spanish)`

Preset sẽ tự động điền các tham số phù hợp để giảm drop câu và tăng độ ổn định theo từng ngôn ngữ.

## Save/Load progress
- `Save Progress`: lưu thông tin media, SRT và cấu hình app vào file `.json` (thường trong `save_files/`).
- `Load Progress`: mở lại project để tiếp tục sửa SRT mà không cần chạy lại từ đầu.

## Ghi chú quan trọng
- App hiện tại xuất file `.srt` (không xuất `.txt`/`.wav`).
- Để tránh lỗi tên file Unicode/ký tự đặc biệt, app dùng file tạm ASCII trong quá trình xử lý và copy kết quả về tên gốc.
- Nếu bật VAD mà fail model, app sẽ fallback chạy không VAD.

## Lỗi thường gặp
1. `Không tìm thấy model`
- Kiểm tra file `models/ggml-large-v3.bin` tồn tại đúng đường dẫn.

2. `Không tìm thấy whisper-cli.exe`
- Kiểm tra `Release/whisper-cli.exe` hoặc chọn lại bằng nút `Chọn`.

3. Không đọc được audio/video
- Kiểm tra ffmpeg (`Release/ffmpeg.exe` hoặc PATH).
- Thử chạy tay: `ffmpeg -version`.

4. Không phát được video trong editor
- Kiểm tra mpv (`Release/mpv.exe` hoặc PATH).
- Thử chạy tay: `mpv --version`.

5. Vẫn chạy CPU thay vì GPU
- Kiểm tra bản `whisper-cli.exe` có build hỗ trợ GPU không.
- Kiểm tra log trong app: nếu backend GPU hợp lệ sẽ hiện thông tin CUDA/OpenVINO.

## Đóng gói exe (tham khảo)
Nếu đã cài PyInstaller:
```powershell
pyinstaller --noconfirm --clean --onefile --windowed app.py
```
Sau đó copy thêm các thư mục/file cần thiết cạnh file exe:
- `models/`
- `Release/`
- `save_files/` (tự tạo nếu chưa có)

---

# English

WhisperSUB_V is a desktop app that generates `.srt` subtitles from video/audio using `whisper.cpp` (`whisper-cli.exe`) and `ggml-large-v3.bin`.

## Main Features
- Simple UI for selecting media and exporting SRT.
- GPU support (CUDA/OpenVINO if your Whisper build supports it).
- Optional VAD to reduce repeated lines.
- Built-in SRT Editor: preview video, edit timeline/text, add/delete subtitle entries.
- Save/Load progress for long projects.
- Advanced mode with decoding parameters and language presets.

## Required Folder Structure
- `app.py`
- `models/ggml-large-v3.bin`
- `Release/whisper-cli.exe`
- `Release/ffmpeg.exe` (or ffmpeg in PATH)
- `Release/mpv.exe` (or mpv in PATH)
- `Release/silero_vad.onnx` (optional, for VAD)
- `save_files/` (auto-created)

## Requirements
- Windows 10/11
- Python 3.10+
- Enough disk space for model files (~3GB for large-v3)
- GPU usage requires a GPU-enabled `whisper-cli.exe` build

## Run
```powershell
python app.py
```

## Basic Workflow
1. Select `Whisper CLI`.
2. Select input video/audio.
3. Choose source language (`auto`, `en`, `vi`, `ja`, `ko`, `zh`, `fr`, `de`, `es`).
4. Click `Xuat SRT`.
5. Output `.srt` is saved next to the source media with the same base name.

## Advanced Mode
Use `Che do nang cao` to tune:
- `temperature`, `temperature-inc`, `best-of`, `beam-size`
- `no-speech-thold`, `entropy-thold`, `logprob-thold`
- `--vad`, `vad-threshold`
- `max-context`, `no-fallback`, `--language`

Language presets included:
- `auto`, `en`, `vi`, `ja`, `ko`, `zh`, `fr`, `de`, `es`

## Notes
- Output format is `.srt` only.
- The app uses temporary ASCII filenames internally, then copies the final result back to the original Unicode path.
- If VAD fails to initialize, the app can fallback to non-VAD mode.

## Packaging (Reference)
```powershell
pyinstaller --noconfirm --clean --onefile --windowed app.py
```
Then place `models/` and `Release/` next to the built exe.

