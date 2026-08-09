# Kiểm kê phần dịch còn lại — 2026-08-03

## Kết luận

Phần còn lại không phải toàn bộ là mẫu khó cần Sol. Sau khi chạy lại validator hiện tại:

- 7 candidate Luna cũ được phục hồi tại chỗ, không cần gọi model (5 train, 2 valid).
- 2.379 record nên tiếp tục bằng Luna.
- 16 record valid đã có candidate nhưng vẫn còn lỗi chất lượng sau retry được đưa vào Sol web fallback.

## Theo split

| Split | Đã dùng được | Phục hồi bởi validator mới | Còn chạy Luna | Sol web hiện tại |
|---|---:|---:|---:|---:|
| train | 37.948 | 5 | 2.054 | 0 |
| valid | 2.423 | 2 | 4 | 16 |
| test | 2.643 | 0 | 321 | 0 |
| **Tổng phần chưa hoàn tất** |  | **7** | **2.379** | **16** |

## Hàng đợi Luna theo độ dài

| Bucket | Số record |
|---|---:|
| normal | 549 |
| near_tail | 229 |
| tail | 1.172 |
| high_tail | 424 |
| oversized | 5 |
| **Tổng** | **2.379** |

Trong 2.054 record train còn phải gọi model:

- 1.388 chưa từng được thử.
- 340 đã thử nhưng không có candidate hoàn chỉnh.
- 326 có candidate nhưng vẫn còn hard error; nên thử Luna Medium theo lỗi trước khi chuyển Sol.

321 record test nằm trong `exhausted_normal.jsonl`/`tail_escalation.jsonl` nhưng candidate đều null do circuit breaker cũ.
Đây là lỗi thực thi, không phải bằng chứng Luna không dịch được, nên phải đưa lại vào Luna.

## Sol web fallback

16 record đã xác nhận cần fallback được chia thành:

- Batch 01: 15 record normal, 9.966 ký tự nguồn, 29.230 ký tự full prompt.
- Batch 02: 1 record high-tail, 6.722 ký tự nguồn, 15.258 ký tự full prompt.

Trang web local: `web/sol_fallback_queue/index.html`.
