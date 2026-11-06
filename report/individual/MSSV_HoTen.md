# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603650
- **Họ và tên:** Hồ Sỹ Tùng
- **SEED** (in ở ô đầu notebook):2596

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | Đã có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | Đã có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | Đã có |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.819      0.901        0.919   
2. BoW + MLP (deep learning)         0.966   0.915      0.940        0.947   
3a. Encoder (MiniLM) + kNN           0.842   0.904      0.872        0.974   
3b. Encoder (MiniLM) + LogReg        0.926   0.936      0.931        0.923   
3c. Encoder (MiniLM) + MLP head      0.927   0.947      0.937        0.974   
4a. LLM nhỏ zero-shot                0.833   1.000      0.909        0.909   
4b. LLM lớn zero-shot                0.800   1.000      0.889        0.889   
4c. LLM nhỏ few-shot                 0.870   1.000      0.930        0.930   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            6.011      17002.548       0.060   
2. BoW + MLP (deep learning)         9.510      25596.809       0.363   
3a. Encoder (MiniLM) + kNN           3.053        156.047      17.147   
3b. Encoder (MiniLM) + LogReg        3.495       1719.752       8.101   
3c. Encoder (MiniLM) + MLP head      4.282       2232.667       8.240   
4a. LLM nhỏ zero-shot                0.000          1.182     354.722   
4b. LLM lớn zero-shot                0.000          1.186     322.195   
4c. LLM nhỏ few-shot                 0.000          0.617    2384.917   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.098           0.0  
2. BoW + MLP (deep learning)          2.426           0.0  
3a. Encoder (MiniLM) + kNN           25.804           0.0  
3b. Encoder (MiniLM) + LogReg        12.746           0.0  
3c. Encoder (MiniLM) + MLP head      12.395           0.0  
4a. LLM nhỏ zero-shot              2493.096           0.0  
4b. LLM lớn zero-shot              2533.048           0.0  
4c. LLM nhỏ few-shot               2961.304           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

hard_eval: Dự đoán Embedding + LogReg đạt 75% accuracy, thực tế 58.9% do các case đánh lừa phức tạp.

llm_models: Dự đoán LLM lớn tốt hơn, thực tế F1 = 0.889, thấp hơn LLM nhỏ (0.909).

dl_curve: Dự đoán MLP kém hoặc ngang LogReg, thực tế F1 = 0.940 so với 0.901, nhờ khả năng học phi tuyến.

## 4. Phân tích 50 case khó

BoW + LogReg tụt nhiều nhất, F1 từ 0.919 xuống 0.350, đặc biệt ở nhóm spam tự nhiên và né từ khóa.

Case #19: Spam giả dạng người quen, dễ bị BoW nhầm vì thiếu từ khóa đặc trưng.

Case #29: Spam sử dụng C0ngr4tulat10ns, w0n, khiến BoW khó nhận diện từ khóa.

## 5. Thử thách cá nhân

Viết 2 tin spam né từ khóa và 2 tin ham chứa free/win, sau đó chạy BoW và Embedding + LogReg.

Cả hai mô hình đều đúng 2/4 tin (50%). Chúng nhận diện đúng hai tin ham nhưng bỏ sót cả hai tin spam, cho thấy Embedding chưa cải thiện khả năng phát hiện spam ngụy trang trong thử nghiệm này.

## 6. Phương án cho sản phẩm thật

Chọn Embedding + MLP head: F1 = 0.937, p95 = 12.40 ms, tốc độ 2.233 tin/giây, đáp ứng mức 116 tin/giây trung bình và SLA 50 ms.

Đánh đổi: chậm hơn BoW + MLP nhưng chống chịu tốt hơn trên tập case khó (F1 = 0.607 so với 0.409). 

## 7. AI log và điều đã học

|Vì sao API báo lỗi 429 dù sử dụng gói miễn phí và cách nào khắc phục?|Giải thích giới hạn RPM và đề xuất Groq.|CHuyển đổi LLM và thử nghiệm thành công|
|Vì sao mô hình lớn hơn không nhất thiết có F1 cao hơn?|Giải thích ảnh hưởng của dữ liệu, prompt và đặc điểm mô hình.|Đối chiếu F1 của GPT-OSS-20B (0.909) và 120B (0.889)|
|Cách kết hợp ML và LLM làm sao để cân bằng?|Đề xuất kiến trúc hợp lý.|Chọn Embedding + MLP làm mô hình chính, LLM xử lý trường hợp khó.|

Điều rút ra: Mô hình càng lớn thì kết quả càng chính xác là chưa đúng, F1 cao cũng chưa phải là đủ để lựa chọn mô hình thực tế.

