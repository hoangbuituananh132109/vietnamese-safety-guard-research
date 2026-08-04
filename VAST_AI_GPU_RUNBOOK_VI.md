# Thuê Vast.ai và cho Codex điều khiển qua SSH

## 1. Tạo SSH key trên máy Windows

Mở PowerShell:

```powershell
ssh-keygen -t ed25519 -C "nemotron-safety-vast"
Get-Content "$HOME\.ssh\id_ed25519.pub"
```

Đưa **public key** `.pub` vào trang Keys của Vast.ai trước khi tạo instance. Không gửi
private key, API key Vast hay API Gemini cho tôi. Private key giữ ở
`C:\Users\Tuan Anh\.ssh\id_ed25519`.

Tài liệu chính thức: [Vast.ai Windows SSH guide](https://docs.vast.ai/guides/instances/connect/windows-guide)
và [SSH guide](https://docs.vast.ai/guides/instances/connect/ssh).

## 2. Chọn instance

Chọn image PyTorch/CUDA, direct SSH, disk tối thiểu 100–150 GB và **xác minh cột GPU
RAM thực tế**. Không suy VRAM chỉ từ tên quảng cáo GPU. Ưu tiên máy reliability
cao và băng thông disk/network tốt; model + data hiện cần khoảng vài GB nhưng checkpoint,
cache và môi trường cần headroom.

Sau khi thuê, bấm SSH trong Vast để lấy `HOST` và `PORT`.

## 3. Tạo alias SSH

Thêm vào `$HOME\.ssh\config`:

```sshconfig
Host safety-vast
  HostName HOST_FROM_VAST
  Port PORT_FROM_VAST
  User root
  IdentityFile C:\Users\Tuan Anh\.ssh\id_ed25519
  ServerAliveInterval 30
  ServerAliveCountMax 120
```

Kiểm tra:

```powershell
ssh safety-vast "nvidia-smi"
```

Khi lệnh này chạy, chỉ cần cho Codex biết alias `safety-vast`; không cần đưa private key.
Codex có thể gọi `ssh safety-vast ...` từ shell cục bộ sau khi bạn chấp thuận network
access của phiên.

## 4. Upload đúng phần cần thiết

Từ workspace hiện tại:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\vast\sync_to_vast.ps1" `
  -HostAlias safety-vast
```

Script chỉ gửi code, model và manifest nghiên cứu. Nó cố ý không gửi `API.txt`, key pool,
raw translation runner state hay các thư mục không liên quan.

Trên remote:

```bash
cd /workspace/safety-dataset
bash scripts/vast/prepare_remote_workspace.sh /workspace/safety-dataset
```

## 5. Chạy trong tmux

```bash
tmux new -s guard
cd /workspace/safety-dataset
source /venv/main/bin/activate
jupyter lab --ip=127.0.0.1 --port=8888 --no-browser
```

Detach `tmux`: `Ctrl+B`, rồi `D`. Vào lại: `tmux attach -t guard`.

E1 đặc biệt phải chạy trong tmux vì GLiNER2 hiện chỉ warm-restore adapter, không exact
resume optimizer/scheduler/global step.

## 6. Mở Jupyter về máy local

Giữ một PowerShell riêng:

```powershell
ssh -N -L 8888:127.0.0.1:8888 safety-vast
```

Mở URL token mà Jupyter in ra, dạng `http://127.0.0.1:8888/lab?token=...`, rồi mở
`notebooks/phase0_vast_runner.ipynb`.

## 7. Trình tự an toàn

```bash
python scripts/preflight_phase0_experiments.py

python scripts/benchmark_mmbert_lora_context.py \
  --lengths 512,1024,2048,4096,8192 \
  --batch-size 1 --repeats 2 --gradient-checkpointing \
  --output reports/remote_profile/mmbert_fixed_b1.json

python scripts/benchmark_mmbert_schema_context.py \
  --lengths 512,1024,2048,4096,8192 \
  --batch-size 1 --repeats 2 --gradient-checkpointing \
  --output reports/remote_profile/mmbert_schema_b1.json
```

Sau profiler, chạy mini 20–100 step bằng notebook. Chỉ đổi `ALLOW_FULL_RUN=True` khi
profile và mini reload/eval đều qua.

## 8. Lấy kết quả về trước khi hủy instance

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\vast\pull_results.ps1" `
  -HostAlias safety-vast
```

Disk của instance có thể mất khi hủy máy. Luôn kéo `reports/experiment_runs`, profile và
checkpoint quan trọng về local trước khi destroy.

## 9. Vast CLI là tùy chọn

Nếu muốn tạo/quản lý instance bằng CLI, tài liệu chính thức dùng `pip install vastai`,
`vastai set api-key ...`, và hỗ trợ `--ssh --direct`. Không ghi Vast API key vào repo.
Xem [Vast CLI hello world](https://docs.vast.ai/cli/hello-world) và
[create instance](https://docs.vast.ai/cli/reference/create-instance).

## 10. Profile thực tế trên instance 45474443

Instance Quadro RTX 6000 có 23.040 MiB VRAM (khoảng 22 GiB khả dụng), compute capability 7.5
và không có BF16 native. Pipeline khóa `precision=auto` về FP16, dùng GradScaler khởi tạo 512
và growth interval 1.000.000 để tránh các optimizer update bị bỏ qua âm thầm.

Kết quả profile backward có gradient LoRA thật:

- mmBERT fixed và dynamic-schema đều chạy được 8.192 token ở micro-batch 8;
- E2 binary ≤512 chạy micro-batch 32;
- GLiGuard 512 chạy micro-batch 8, effective batch 32, dùng khoảng 10–11 GiB VRAM;
- E2 full hoàn tất 7.226/7.226 update trong 24,79 phút, không overflow và không truncate.

Theo dõi run trực tiếp từ Windows:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\vast\show_experiment_status.ps1" `
  -RunId E1-G-EV-512 -Watch
```

E2 đã được xuất thành `reports/vast_download/exports/e2_results.tar.gz` và xác minh SHA-256.
Báo cáo breakdown local nằm tại
`reports/vast_download/summary/E2-M-EV-GLI-COMPAT-8K.md`.

## 11. Cảnh báo âm thanh khi training dừng hoặc hoàn tất

Watcher chạy trên Windows, kiểm tra state từ Vast qua SSH mỗi 60 giây và không dùng
token Codex. Nó cảnh báo khi runner thất bại, biến mất, state không cập nhật, mất SSH
nhiều lần liên tiếp, hồi phục hoặc hoàn tất.

Với Phase-0, lệnh tất-cả-trong-một sẽ khởi động cả runner Vast và watcher local:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\vast\start_overnight_phase0_with_alert.ps1"
```

Khởi động nền:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\vast\start_vast_training_alert.ps1"
```

Xem heartbeat và các sự kiện gần nhất:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\vast\show_vast_training_alert.ps1"
```

Dừng watcher:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\vast\stop_vast_training_alert.ps1"
```

Máy Windows phải còn thức và có thiết bị âm thanh. Watcher không ngăn sleep mặc định.

## 12. Dashboard trực quan theo thời gian thực

Khởi động dashboard local và tự mở trình duyệt:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\vast\start_training_dashboard.ps1"
```

Dashboard ở `http://127.0.0.1:8765/`, cập nhật dữ liệu Vast qua SSH mỗi 15 giây.
Nó hiển thị tiến độ/ETA từng thí nghiệm, epoch và optimizer step, loss curve, GPU,
evaluation matrix, confusion matrix Safe/Unsafe, calibration và ý nghĩa các metric.

Xem trạng thái hoặc dừng:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\vast\show_training_dashboard_status.ps1"

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\vast\stop_training_dashboard.ps1"
```
