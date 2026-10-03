# Kết quả Day 17: Memory Systems for AI Agent

Chạy từ thư mục gốc repo bằng `python3 src/benchmark.py` và `python3 -m pytest src/test_agents.py -v`. Benchmark luôn dùng offline và tạo state tạm riêng cho mỗi bộ dữ liệu, nên không cần API key và chạy lại cho cùng kết quả. Mô hình live có thể cấu hình bằng `LLM_PROVIDER`, `LLM_MODEL` và API key tương ứng; các lựa chọn gồm `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`. `CUSTOM_BASE_URL` và `OLLAMA_BASE_URL` dùng cho hai provider tương ứng. Ngưỡng compact mặc định là 1200 token ước lượng, giữ 4 message gần nhất; có thể đổi bằng `COMPACT_THRESHOLD_TOKENS` và `COMPACT_KEEP_MESSAGES`.

## Số liệu offline

| Bộ dữ liệu | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | Baseline | 1071 | 12680 | 0.000 | 0.200 | 0 | 0 |
| Standard | Advanced | 1183 | 18260 | 1.000 | 1.000 | 213 | 0 |
| Stress | Baseline | 181 | 21681 | 0.000 | 0.200 | 0 | 0 |
| Stress | Advanced | 250 | 12657 | 1.000 | 1.000 | 156 | 2 |

Baseline giữ hội thoại theo `thread_id`, nên câu hỏi ở thread mới không đọc được fact cũ. Advanced lưu tên, nghề, nơi ở và preference trong `User.md`, nhờ đó recall tăng lên 1.000. Ở Standard, hội thoại ngắn không kích hoạt compact; Advanced phải mang thêm profile vào mỗi prompt, khiến prompt tokens tăng từ 12680 lên 18260. Agent tokens only cũng tăng do câu trả lời recall chứa thông tin thật, nhưng đó là lượng **đầu ra**, không phải chi phí đọc profile.

Ở Stress, hai lần compact giảm prompt tokens của Advanced xuống 12657, thấp hơn 21681 của Baseline. Đối chứng với ngưỡng compact 1000000 cho Advanced cho 0 compactions và 22572 prompt tokens, trong khi recall vẫn 1.000. Điều này cho thấy lợi ích trực tiếp của compact nằm ở lượng ngữ cảnh xử lý qua các lượt. Profile tăng 156 byte ở Stress và 213 byte ở Standard; file sẽ tiếp tục tăng nếu thêm nhiều loại fact, còn summary có thể bỏ sót chi tiết hội thoại cũ.

## Bonus: lọc fact có độ chắc chắn thấp

Bộ trích fact chỉ ghi các phát biểu tự mô tả đủ rõ và bỏ qua câu hỏi ngắn. Ngưỡng confidence là 0,8; những mẫu khẳng định trực tiếp được chấm 0,85–0,95. Nơi đi họp ở Hà Nội và câu đùa về `product manager` không được ghi vào profile. Correction rõ ràng từ Huế sang Đà Nẵng hoặc từ backend engineer sang MLOps engineer cập nhật cùng khóa, tránh giữ hai giá trị mâu thuẫn. Test bonus xác nhận các trường hợp đó.

Cách lọc theo mẫu giúp tránh một số fact sai và giữ file gọn, nhưng có thể bỏ sót lời nói mơ hồ hoặc cách diễn đạt chưa có trong các mẫu. `Response quality` là điểm heuristic từ mức khớp chuỗi mong đợi, không phải đánh giá chất lượng ngôn ngữ của judge model. Token cũng là ước lượng ký tự, không phải số token tính phí thực tế từ provider.
