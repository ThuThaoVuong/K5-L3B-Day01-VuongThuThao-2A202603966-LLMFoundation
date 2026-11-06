# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603966
- **Họ và tên:** Vương Thu Thảo
- **SEED** (in ở ô đầu notebook): 2603

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | v |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | v |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | v |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*

| # | Mô hình | F1 | F1_SUB |
|---|---|---:|---:|
| 1 | BoW + LogReg (tự code) | 0.891 | 0.947 |
| 2 | BoW + MLP (Deep Learning) | 0.921 | 0.895 |
| 3a | Encoder (MiniLM) + kNN | 0.867 | 0.950 |
| 3b | Encoder (MiniLM) + LogReg | 0.944 | 0.974 |
| 3c | Encoder (MiniLM) + MLP Head | 0.940 | 0.950 |
| 4a | LLM nhỏ Zero-shot | 0.950 | 0.950 |
| 4b | LLM lớn Zero-shot | 0.000 | 0.000 |
| 4c | LLM nhỏ Few-shot | **0.976** | **0.976** |

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

### 1. `hard_eval` (Tập đối kháng)
- **Dự đoán ban đầu:** Nhóm khó nhất đối với LLM là `prompt_injection`.
- **Kết quả thực tế:** LLM gặp nhiều khó khăn nhất ở nhóm tin giao dịch khẩn cấp (`transactional_ham`) do xuất hiện nhiều cảnh báo giả bởi các từ khóa nhạy cảm.
- **Nguyên nhân sai lệch:** Đánh giá quá cao khả năng phòng thủ trước các câu lệnh độc hại, nhưng chưa chú trọng đúng mức đến độ nhạy của mô hình đối với các từ khóa tài chính.

### 2. `llm_fewshot` (Chi phí Few-shot)
- **Dự đoán ban đầu:** Chi phí tăng khoảng 2–3 lần do bổ sung các ví dụ mẫu vào prompt.
- **Kết quả thực tế:** Chi phí thực tế tăng khoảng 5–7 lần.
- **Nguyên nhân sai lệch:** Chỉ tính đến lượng token của các ví dụ mẫu mà chưa tính đầy đủ chi phí token từ system prompt, định dạng JSON và prompt template.

### 3. `tradeoff` (Đánh đổi Pareto & Độ trễ)
- **Dự đoán ban đầu:** LLM có độ trễ ở mức hàng trăm mili giây, vượt ngưỡng SLA 50 ms.
- **Kết quả thực tế:** Độ trễ tăng lên khoảng 3–8 giây ở các trường hợp khó, đồng thời chỉ số F1 suy giảm đáng kể.
- **Nguyên nhân sai lệch:** Giả định hiệu suất F1 của LLM ổn định tương tự trên tập kiểm thử tiêu chuẩn, trong khi hiệu suất thực tế bị ảnh hưởng đáng kể khi xử lý dữ liệu đối kháng.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

- **BoW + LogReg** tụt nhiều nhất: F1 giảm từ 0.947 xuống 0.350 (giảm 0.597).
- Các mô hình BoW và Embedding truyền thống đều suy giảm ở mức độ khác nhau.
- **LLM nhỏ Zero-shot** đạt F1 cao nhất (0.985), trong khi LLM nhỏ kết hợp prompt phòng thủ duy trì hiệu suất ổn định (0.951).

### Nguyên nhân và cơ chế gây sai

- **BoW + LogReg:** Phụ thuộc vào từ khóa, dễ nhầm lẫn khi xuất hiện từ khóa gây nhiễu hoặc cách diễn đạt bất thường.
- **BoW + MLP:** Hạn chế trong việc nắm bắt ngữ cảnh và mối quan hệ ngữ nghĩa giữa các từ.
- **Embedding + LogReg/MLP:** Biểu diễn được ngữ nghĩa nhưng vẫn có thể nhầm lẫn với các trường hợp nhập nhằng hoặc ngữ cảnh phức tạp.
- **LLM:** Có khả năng phân tích ngữ cảnh tốt hơn, giúp xử lý các trường hợp khó hiệu quả hơn.


## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
### Đề #6: LLM trả kết quả JSON kèm giải thích

- **Thay đổi:** Sửa `build_classification_prompt` để LLM trả về JSON theo cấu trúc `{"label": ..., "reason": ...}` và bổ sung xử lý parse JSON.
- **Kết quả thử nghiệm:** Chạy lại trên 10 tin nhắn SUB:
  - Độ chính xác: **8/10 (80%)**.
  - 8 tin `ham` được phân loại đúng; 2 tin `spam` bị phân loại nhầm thành `ham`.
  - Tổng chi phí: **$0.00102**.
  - Tổng độ trễ: **7.37 giây**.
- **Đánh giá:** JSON giúp kết quả có cấu trúc rõ ràng, dễ tích hợp và lưu trữ; trường `reason` hỗ trợ giải thích quyết định của mô hình. Tuy nhiên, cần so sánh token và chi phí với phiên bản chỉ trả về 1 từ để đánh giá mức độ đánh đổi. 

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
- Sản phẩm thật (10 triệu SMS/ngày, p95 < 50ms) cần đạt SLA độ trễ & chất lượng thì có thể chọn LLM nhỏ zero-shot hoặc Encoder + LogReg.
- Chi phí LLM nhỏ (~$0.022/1k tin): 300 triệu tin/tháng = $6.600/tháng.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
1. **Câu hỏi:** Phân tích mô hình suy giảm hiệu năng nhiều nhất khi chuyển từ tập test thường sang 50 case khó; xác định nhóm case gây lỗi và phân tích 2 ví dụ sai cụ thể của BoW + LogReg.
2. **AI trả lời:** Xác định **BoW + LogReg** suy giảm nhiều nhất (F1 từ 0.947 xuống 0.350), chủ yếu gặp khó khăn ở nhóm `obfuscation` và `manipulation`. Phân tích 2 ví dụ cho thấy cơ chế dựa trên tần suất từ khóa dễ bị đánh lừa bởi cách viết biến đổi và ngữ cảnh thao túng.
3. **Tôi:** Xác nhận kết quả, cô đọng nội dung phân tích và hoàn thiện phần nhận xét để đưa vào báo cáo tổng kết.
