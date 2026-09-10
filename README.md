# Banking77 Intent Triage

Dự án cá nhân — phân loại câu hỏi khách hàng theo **intent** để hỗ trợ định tuyến yêu cầu trong lĩnh vực ngân hàng bán lẻ.

> Mục tiêu ban đầu là tự tay thử các phương pháp từ baseline đơn giản (TF-IDF + ML) đến mạng nơ-ron (TextCNN), xem mỗi bước cải thiện được bao nhiêu và lỗi tập trung ở đâu — phục vụ việc học và chuẩn bị portfolio intern.

---

## 1. Bài toán

Cho một câu hỏi/yêu cầu của khách hàng (ví dụ: *"Why was my card declined?"*), mô hình cần dự đoán **intent** (nhóm ý định) tương ứng để hệ thống có thể tự động định tuyến hoặc gợi ý câu trả lời phù hợp.

- **Loại bài toán:** Multi-class text classification
- **Số lớp:** 77 intent (phân loại chi tiết về giao dịch, thẻ, chuyển tiền, xác minh danh tính, v.v.)

---

## 2. Dữ liệu

**Nguồn:** [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) trên Hugging Face Datasets

| Tập | Số mẫu | Ghi chú |
|-----|--------|---------|
| Train (gốc) | 10.003 | Từ dataset gốc |
| Train (thực tế) | 8.502 | Sau khi tách validation |
| Validation | 1.501 | Tách 15% từ train, stratify theo label |
| Test | 3.080 | Giữ nguyên, chỉ dùng để đánh giá cuối |

- Mỗi mẫu gồm: câu văn bản (`text`) và nhãn số (`label`, từ 0–76)
- Dữ liệu khá cân bằng (~130 mẫu/intent trong train)

---

## 3. Phương pháp

### 3.1 Baseline — Naive Bayes

- **Đặc trưng:** TF-IDF (unigram + bigram, `min_df=2`, `sublinear_tf=True`)
- **Mô hình:** `MultinomialNB`
- **Mục đích:** Xác lập ngưỡng tối thiểu cần vượt qua

### 3.2 Linear SVM

- **Đặc trưng:** TF-IDF (cùng cấu hình với Naive Bayes)
- **Mô hình:** `LinearSVC(C=1.0)`
- **Pipeline:** `sklearn.Pipeline` gồm TF-IDF → SVC
- **Kết quả tốt nhất trong 3 mô hình đã thử** → đang tiếp tục cải thiện

### 3.3 TextCNN

- **Đặc trưng:** Word embeddings (huấn luyện từ đầu)
- **Kiến trúc:** Convolutional filters với nhiều kernel size để bắt n-gram cục bộ
- **Notebook:** [`notebooks/02_textcnn.ipynb`](notebooks/02_textcnn.ipynb)

---

## 4. Cách chạy

### Yêu cầu

- Python 3.10+
- Khuyến nghị dùng virtual environment

### Cài đặt

```bash
git clone https://github.com/thanhteddyyy/banking77-triage.git
cd banking77-triage

python -m venv .venv
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### Chạy notebook

```bash
jupyter notebook notebooks/01_eda_and_baseline.ipynb  # EDA + Naive Bayes + SVM
jupyter notebook notebooks/02_textcnn.ipynb           # TextCNN
```

> **Lưu ý:** Dataset sẽ được tải tự động từ Hugging Face khi chạy lần đầu (cần kết nối internet).

---

## 5. Kết quả

Đánh giá trên **tập Validation** (1.501 mẫu):

| Mô hình | Accuracy | Macro-F1 | Thời gian train |
|---------|----------|----------|-----------------|
| Naive Bayes (TF-IDF) | 0.7975 | 0.7631 | ~0.28s |
| TextCNN | 0.8501 | 0.8439 | — |
| **Linear SVM (TF-IDF)** | **0.8767** | **0.8691** | ~0.84s |

Sau khi so sánh cả 3 mô hình, **Linear SVM đạt Accuracy và Macro-F1 cao nhất** nên được chọn để tiếp tục cải thiện.

> Kết quả trên **tập Test** (3.080 mẫu) chưa được đánh giá — sẽ cập nhật sau khi hoàn thiện pipeline SVM.

---

## 6. Phân tích lỗi

### Linear SVM làm tốt hơn đáng kể so với Naive Bayes

SVM cải thiện ~8 điểm Accuracy và ~10 điểm Macro-F1. Với dữ liệu văn bản ngắn và nhiều class, SVM với TF-IDF là lựa chọn baseline mạnh và nhanh.

### TextCNN không vượt qua được SVM trong thí nghiệm này

TextCNN đạt kết quả tốt hơn Naive Bayes nhưng vẫn kém hơn SVM ~2.5 điểm Macro-F1. Nguyên nhân có thể do embedding huấn luyện từ đầu với lượng dữ liệu hạn chế (~8.500 mẫu) chưa đủ để bắt được ngữ nghĩa tốt.

### Những intent khó nhất (F1 thấp nhất của SVM)

| Intent | F1-score | Support | Ghi chú |
|--------|----------|---------|---------|
| `virtual_card_not_working` | 0.60 | 6 | Dễ nhầm với `card_not_working` |
| `card_acceptance` | 0.63 | 9 | Số mẫu validation ít |
| `pin_blocked` | 0.69 | 17 | Hay bị nhầm thành `get_physical_card` |
| `contactless_not_working` | 0.75 | 5 | Quá ít mẫu |

### Những cặp nhầm lẫn đáng chú ý

- `card_arrival` ↔ `card_delivery_estimate` — hai intent rất gần nghĩa
- `transfer_fee_charged` → bị nhầm thành `card_payment_fee_charged`
- `top_up_reverted` → bị nhầm thành `top_up_failed` hoặc `pending_top_up`
- `direct_debit_payment_not_recognised` → nhầm thành `card_payment_not_recognised`

Phần lớn lỗi đến từ các intent **có nghĩa tương tự nhau**, không phải từ dữ liệu nhiễu.

---

## 7. Hạn chế và hướng phát triển

### Đã kiểm chứng
- [x] TF-IDF + Naive Bayes (baseline)
- [x] TF-IDF + Linear SVM (strong baseline — **đang tập trung cải thiện**)
- [x] TextCNN (word embeddings từ đầu)
- [x] Phân tích lỗi theo intent và cặp nhầm lẫn

### Đang thực hiện

- [ ] **Tinh chỉnh hyperparameter SVM** — tuning `C`, `ngram_range`, `max_features`
- [ ] **Feature engineering nâng cao** — character n-gram, subword features

### Còn cần thử

- [ ] **Pretrained embeddings** (GloVe, FastText) cho TextCNN
- [ ] **Sentence-BERT / DistilBERT fine-tune** — kỳ vọng giải quyết được các cặp nhầm lẫn nghĩa gần
- [ ] **Data augmentation** cho các intent có ít mẫu (`contactless_not_working`, `virtual_card_not_working`)
- [ ] Đánh giá cuối cùng trên tập **Test** sau khi hoàn thiện pipeline SVM

---

## Cấu trúc thư mục

```
banking77-triage/
├── notebooks/
│   ├── 01_eda_and_baseline.ipynb    # EDA, Naive Bayes, Linear SVM
│   ├── 02_textcnn.ipynb             # TextCNN
│   └── svm_error_analysis/          # CSV kết quả phân tích lỗi SVM
├── src/                             # Module Python (dự kiến)
├── data/                            # Dữ liệu local (nếu có)
├── reports/                         # Báo cáo, biểu đồ
├── requirements.txt
└── README.md
```

---

## Tác giả

Dự án side project cá nhân, thực hiện trong quá trình học và chuẩn bị cho vị trí intern ML/NLP.
