# Feature Dictionary — OULAD Data Pipeline & Warehouse

> Tài liệu mô tả toàn bộ feature/cột dữ liệu trong hệ thống — từ EDA gốc, Gold Layer (phục vụ ML), Postgres Staging, đến Mart Dashboard (phục vụ phân tích/Warehouse). Bản đầy đủ, thay thế hoàn toàn các bản trước.

---

## Phần 1 — Data Quality Findings (Bronze/Silver EDA)

Khảo sát thực hiện trên toàn bộ 7 bảng Bronze trước khi thiết kế logic Silver (Primary Key, Foreign Key/Referential Integrity, Missing Values, Duplicate, miền giá trị).

- **Referential integrity:** 0 giá trị mồ côi (orphan) ở toàn bộ 5 cặp khóa ngoại đã kiểm tra (`student_registration`/`student_assessment`/`student_vle` → `student_info`; `student_assessment` → `assessments`; `student_vle` → `vle`). → Xác nhận LEFT JOIN ở Silver an toàn, không mất dữ liệu.
- **`student_vle`:** 2,195,960 dòng trùng theo khóa `(id_student, code_module, code_presentation, id_site, date)` (~20.6% tổng số dòng) — đây là đặc điểm gốc của dữ liệu (nhiều lượt ghi nhận click trong cùng ngày), **không phải lỗi**. Không xử lý ở Silver vì Gold sẽ `SUM(sum_click)` theo nhóm nên không ảnh hưởng kết quả.
- **`student_registration`:** đã kiểm tra `duplicated(subset=["id_student","code_module","code_presentation"])` = 0 dòng trùng — khóa tự nhiên này là duy nhất trong dữ liệu gốc OULAD. Vì vậy **không tồn tại asset Silver riêng cho bảng này**.
- **`student_registration.date_unregistration`** thiếu 69.1% — có ý nghĩa nghiệp vụ (sinh viên không rút môn), không phải lỗi dữ liệu, không xử lý.
- **`student_assessment.score`:** phát hiện sai kiểu dữ liệu (`object` thay vì số) ở Bronze — đã sửa trong `standard_bronze()`. Sau khi ép kiểu bằng `pd.to_numeric(errors='coerce')`: 173/173,912 dòng (0.1%) không parse được thành số, tỷ lệ chấp nhận được, không điều tra thêm.
- **Miền giá trị:** `final_result`, `gender`, `age_band` đều đúng domain knowledge kỳ vọng, không có giá trị lạ. `score` sau khi sửa: min=0.0, max=100.0 — hợp lệ. `sum_click`: min=1, max=6977, không có giá trị âm.
- **`assessments.date` (Exam):** giá trị gốc là rỗng/NaN cho loại `Exam` (không có hạn nộp cố định) — **giữ nguyên NaN, không ép về 0** ở bất kỳ tầng nào. **Đính chính quan trọng (phát hiện muộn hơn khi làm `dim_assessment`):** không phải TOÀN BỘ bài Exam đều NULL — kiểm tra thực tế cho thấy chỉ **11/24 bài Exam** có `date = NULL` (khả năng là kỳ thi cuối kỳ thật, lịch riêng ngoài file này); **13/24 bài** vẫn có `date` cụ thể (khả năng là kiểm tra giữa kỳ, có lịch như CMA/TMA). `CMA`/`TMA` thì 100% có `date`. Nguyên tắc lọc `date <= cutoff_day` vẫn đúng bản chất cho cả 2 nhóm Exam, không cần sửa code — chỉ cần hiểu đúng là "phần lớn nhưng không phải toàn bộ" Exam bị loại khỏi cutoff sớm.
- **Giá trị ngày âm** (`date`, `date_registration`...) là **bình thường** trong OULAD — tính tương đối so với ngày khai giảng (ngày 0), âm nghĩa là trước khai giảng.
- **`vle.week_from`/`vle.week_to`:** dữ liệu gốc dùng ký tự `"?"` để biểu diễn giá trị thiếu (không phải NULL thật) — cùng quy ước với `assessments.date` (Exam). **Đã sửa** trong `standard_bronze()`: gộp vào cùng nhóm ép kiểu với các cột ngày, dùng `pd.to_numeric(errors='coerce').astype('Int64')` — chuẩn hóa `"?"` → NULL, ép kiểu số nguyên. 2 cột này hiện **chưa được sử dụng** ở Gold Layer hay mart_dashboard (metadata mô tả tuần thiết kế sử dụng tài nguyên VLE, khác với `date` là ngày tương tác thật) — có thể là hướng mở rộng phân tích sau này (so sánh sinh viên dùng tài liệu đúng/lệch tuần thiết kế).

---

## Phần 2 — Feature Dictionary: Gold Layer (phục vụ ML, có Cutoff)

### Cấu hình cutoff (`GOLD_CONFIG`)

| Asset | cutoff_day | Ý nghĩa |
|---|---|---|
| `gold_features_cutoff2` | 14 | Tuần 2 — cảnh báo sớm nhất, dữ liệu còn ít |
| `gold_features_cutoff4` | 28 | Tuần 4 — sau bài tập đầu tiên |
| `gold_features_cutoff8` | 56 | Tuần 8 — điểm giữa, còn thời gian cảnh báo |
| `gold_features_cutoff12` | 84 | Tuần 12 — đánh giá gần giữa kỳ |

*(Ghi chú đã biết: các kỳ học ngắn hơn 84 ngày sẽ khiến `cutoff12` gần tương đương toàn bộ dữ liệu kỳ đó, giảm ý nghĩa "cảnh báo sớm" — hạn chế đã chấp nhận, ghi vào phần Hạn chế của báo cáo.)*

**Grain:** 1 dòng = 1 `id_student` + 1 `code_module` + 1 `code_presentation` + 1 mốc cutoff.
**Base Population:** xuất phát từ **`bronze_student_registration`** (đọc trực tiếp, không qua asset Silver), LEFT JOIN mọi nguồn khác vào — đảm bảo giữ lại cả sinh viên không tương tác gì.
**Công cụ xử lý:** DuckDB (predicate pushdown, đọc trực tiếp Parquet trên MinIO qua `httpfs`) — thay cho pandas thuần, giải quyết lỗi Out of Memory khi xử lý `silver_student_vle_enriched` (433MB gốc) ở `cutoff8`/`cutoff12`. **Extension `httpfs` được `INSTALL` sẵn lúc build Docker image** (không phải lúc runtime) để tránh phụ thuộc mạng khi materialize asset — chỉ `LOAD httpfs` ở runtime.

### Nhóm Engagement (nguồn: `silver_student_vle_enriched`, lọc `date <= cutoff_day`)

| feature_name | description | data_type | source | calculation | cutoff_rule | leakage_note |
|---|---|---|---|---|---|---|
| `total_click` | Tổng số lượt click VLE tính đến cutoff | int | silver_student_vle_enriched | `SUM(sum_click)` nhóm theo student/module/presentation | Lọc `date <= cutoff_day` **trước** khi SUM | Nếu quên lọc trước, sẽ đếm cả click tương lai → leakage |
| `active_days` | Số ngày khác nhau có hoạt động | int | silver_student_vle_enriched | `COUNT(DISTINCT date)` | Cùng tập đã lọc cutoff | — |
| `active_site` | Số trang/tài nguyên VLE khác nhau đã truy cập | int | silver_student_vle_enriched | `COUNT(DISTINCT id_site)` | Cùng tập đã lọc cutoff | — |
| `activity_types` | Số loại hoạt động khác nhau | int | silver_student_vle_enriched | `COUNT(DISTINCT activity_type)` | Cùng tập đã lọc cutoff | — |
| `avg_click_per_day` | Trung bình click mỗi ngày hoạt động | float | phái sinh | `total_click / active_days` | Tính ngay sau khi query trả về (trước khi fillna) | Nếu điền 0 cho `active_days` trước rồi mới chia sẽ gây lỗi chia 0 |
| `last_active_day` | Ngày (tương đối) gần cutoff nhất có hoạt động | int | silver_student_vle_enriched | `MAX(date)` | Cùng tập đã lọc cutoff | Cột trung gian, không bắt buộc dùng trực tiếp làm feature ML |
| `days_since_last_activity` | Số ngày "im lặng" tính đến cutoff | int | phái sinh | `cutoff_day - last_active_day` | Tính ngay sau `last_active_day`, trước fillna | Khi NaN → điền = `cutoff_day` (im lặng tối đa quan sát được), **không điền 0** |

### Nhóm Assessment — tách 2 nguồn theo đúng grain

> **Quyết định kiến trúc quan trọng:** `num_assigned` KHÔNG được tính từ `silver_student_assessment_enriched` (chỉ chứa lượt đã nộp — sinh viên chưa từng nộp sẽ bị tính sai `num_assigned=0`). Đã sửa bằng cách tính `num_assigned` từ `bronze_assessments` (catalog), JOIN vào Base Population theo `(code_module, code_presentation)` — không theo `id_student`.

| feature_name | description | data_type | source | calculation | cutoff_rule | leakage_note |
|---|---|---|---|---|---|---|
| `num_assigned` | Số bài kiểm tra đã tới hạn nộp của môn/kỳ học | int | **bronze_assessments** | `COUNT(DISTINCT id_assessment)` nhóm theo `code_module, code_presentation` | Lọc theo `date` (hạn nộp) `<= cutoff_day`; NaN (1 phần Exam) tự động bị loại | JOIN theo 2 khóa `(code_module, code_presentation)` — không theo `id_student` |
| `num_submission` | Số bài sinh viên đã nộp trước cutoff | int | silver_student_assessment_enriched | `COUNT(date_submitted)` trong tập `submitted` | Lọc `date_submitted <= cutoff_day` | Hành vi cá nhân — khác grain với `num_assigned` |
| `avg_score` | Điểm trung bình các bài đã nộp | float | silver_student_assessment_enriched | `AVG(score)` trong tập `submitted` | Cùng tập `submitted` | Giữ NaN nếu chưa có bài nào được chấm |
| `submission_rate` | Tỷ lệ đã nộp / đã tới hạn | float | phái sinh (2 nguồn) | `num_submission / num_assigned`, sau khi cả 2 cột đã `fillna(0)` | Tính sau khi merge cả 2 nguồn vào base | `0/0` → NaN tự nhiên; `2/0` → `0.0` — phân biệt đúng nhờ tách 2 nguồn |
| `num_late_submissions` | Số bài nộp trễ hạn (trước cutoff) | int | silver_student_assessment_enriched | `COUNT(DISTINCT id_assessment)` trong tập `late`, lọc từ trong `submitted` | Tập `late` lọc từ trong `submitted` | Tránh đếm nhầm bài nộp trễ nhưng nộp **sau** cutoff |

*Ghi chú: bài Exam không có `date` (NaN) tự động bị loại khỏi `num_assigned` ở mọi cutoff — hành vi đúng mong muốn. Xem đính chính về tỷ lệ thật (11/24, không phải toàn bộ) ở Phần 1.*

### Nhóm Demographic & Registration (KHÔNG cutoff — thuộc tính tĩnh)

| feature_name | description | data_type | source | calculation | leakage_note |
|---|---|---|---|---|---|
| `age_band` | Nhóm tuổi sinh viên | category | bronze_student_info | lấy trực tiếp | — |
| `imd_band` | Chỉ số mức độ thiếu thốn khu vực | category | bronze_student_info | lấy trực tiếp | — |
| `highest_education` | Trình độ học vấn cao nhất | category | bronze_student_info | lấy trực tiếp | — |
| `region` | Vùng miền sinh sống | category | bronze_student_info | lấy trực tiếp | — |
| `num_of_prev_attempts` | Số lần đã học lại môn này trước đó | int | bronze_student_info | lấy trực tiếp | Tín hiệu rủi ro mạnh, biết được ngay từ đầu kỳ nên không leakage |
| `studied_credits` | Tổng số tín chỉ đang học trong kỳ | int | bronze_student_info | lấy trực tiếp | — |
| `disability` | Sinh viên có khuyết tật hay không (Y/N) | category | bronze_student_info | lấy trực tiếp | — |
| `date_registration` | Ngày đăng ký môn học (tương đối) | int | bronze_student_registration | lấy trực tiếp, vào Base Population ngay từ đầu | Biết trước cutoff sớm nhất |

### Quy tắc điền giá trị thiếu (`fill_missing_gold`)

| Nhóm cột | Cách xử lý khi NaN | Lý do |
|---|---|---|
| `total_click, active_days, active_site, activity_types, num_assigned, num_submission, num_late_submissions, num_of_prev_attempts, studied_credits` | Điền **0** | Không tương tác/không có bài = 0 thật |
| `age_band, imd_band, highest_education, region, disability` | Điền **"Unknown"** | Categorical thiếu |
| `avg_click_per_day, avg_score, submission_rate` | **Giữ nguyên NaN** | "Không có gì để tính" ≠ "kết quả = 0" |
| `days_since_last_activity` | Điền = **`cutoff_day`** | "Im lặng tối đa có thể quan sát được" |

**Thứ tự quan trọng:** `num_assigned`/`num_submission` phải `fillna(0)` **trước** khi tính `submission_rate`.

### Nguyên tắc chống Data Leakage (toàn bộ Gold Layer)

1. Mọi phép lọc cutoff phải thực hiện **trước** khi aggregate (predicate pushdown trong DuckDB).
2. `final_result` **tuyệt đối không** xuất hiện trong feature Gold — chỉ dùng làm Target.
3. Assessment (hành vi cá nhân): `submitted` là tập con của `assigned`, lọc lồng nhau, không lọc độc lập.
4. Assessment (catalog): `num_assigned` tính độc lập ở cấp môn học, JOIN vào mọi sinh viên bất kể có nộp hay không.
5. Mỗi mốc cutoff có 1 model ML riêng biệt — không gộp.
6. **`date_unregistration` KHÔNG được dùng làm feature ML** — đây là target leakage (không phải temporal leakage): sự tồn tại giá trị gần như đồng nghĩa `final_result = Withdrawn`, lọc cutoff không cứu được. Chỉ hợp lệ ở `mart_dashboard`.

### Nguyên tắc chung: khi nào 1 bảng KHÔNG cần asset Silver riêng

Áp dụng cho `courses`, `assessments`, `vle`, `student_info`, `student_registration` — đọc thẳng Bronze ở tầng sau. Điều kiện bỏ qua Silver: không cần dedup, không cần LEFT JOIN, không cần biến đổi/enrich cột nào.

---

## Phần 3 — Postgres Staging

### Schema `staging_ml` (database `oulad_dwh`)

4 bảng Gold có bản sao: `stg_gold_features_cutoff2/4/8/12`. Cơ chế: Truncate + Insert (Idempotent).

### Schema `staging_dashboard` (database `oulad_dwh`)

Nạp trực tiếp từ Silver/Bronze (không qua Gold, không cutoff):

| Bảng staging | Nguồn | Lý do cần |
|---|---|---|
| `stg_silver_student_vle_enriched` | silver_student_vle_enriched | Xu hướng tương tác VLE theo thời gian |
| `stg_silver_student_assessment_enriched` | silver_student_assessment_enriched | Kết quả nộp bài (hành vi cá nhân) |
| `stg_bronze_student_registration` | bronze_student_registration | Base cho `dim_enrollment` |
| `stg_bronze_student_info` | bronze_student_info | Demographic + `final_result` + `num_of_prev_attempts`/`studied_credits` |
| `stg_bronze_courses` | bronze_courses | Thông tin môn học cho `dim_course` |
| `stg_bronze_assessments` | bronze_assessments | Catalog bài kiểm tra — bắt buộc staging riêng vì `dbt-postgres` không đọc Parquet trên MinIO trực tiếp |

**Đã hoàn thành.** Dùng kỹ thuật COPY (psycopg2 `copy_expert`) cho `stg_silver_student_vle_enriched` (~10.6 triệu dòng) thay vì `to_sql(method="multi")` — giảm thời gian materialize từ 27 phút xuống 50 giây.

---

## Phần 4 — mart_dashboard (Fact Constellation: 5 Dim + 3 Fact — HOÀN THÀNH)

**Công cụ:** dbt-postgres, project `oulad_dbt/`, đọc từ `staging_dashboard` qua `source()`, ghi ra schema thật `dbt_dev_mart_dashboard` (dbt tự nối `target.schema` + custom schema — giữ nguyên mặc định, không ghi đè macro `generate_schema_name`), materialized dạng `table`.

### 4.1. Nguyên tắc chung

- Dùng dữ liệu ĐẦY ĐỦ từ Silver/Bronze, **KHÔNG cutoff** — khác biệt căn bản với Gold/`mart_ml`.
- `final_result` **CẦN** có ở đây (hiển thị Pass/Fail/Withdrawn/Distinction thật) — khác quy tắc cấm dùng làm feature ở Gold.
- `id_student`/`code_module`/`code_presentation` lưu dạng TEXT trong các bảng staging (do `df.head(0).to_sql()` đưa kiểu object về text).

### 4.2. DIMENSION TABLES

#### `dim_student` (SCD Type 2)

**Nguồn:** `stg_bronze_student_info`. **Grain:** 1 phiên bản demographic của 1 sinh viên (không phải 1 dòng/người).

| Cột | Ý nghĩa | Nguồn |
|---|---|---|
| `student_key` | Surrogate key, định danh 1 version | Sinh ra — `ROW_NUMBER() OVER (ORDER BY id_student, effective_from_presentation)` |
| `id_student` | Business key gốc | Có sẵn |
| `gender`, `highest_education`, `imd_band`, `age_band`, `region`, `disability` | Demographic | Có sẵn |
| `effective_from_presentation` | `code_presentation` bắt đầu version (TEXT, không phải DATE) | Suy ra — `MIN(code_presentation)` trong nhóm version |
| `effective_to_presentation` | `code_presentation` kết thúc version | Suy ra — `MAX(code_presentation)` trong nhóm version |
| `is_current` | Version mới nhất hay không | Suy ra — so với `MAX(effective_to_presentation)` toàn bộ của `id_student` |

**Quy trình xử lý (SCD2):**
1. Dedup về `(id_student, code_presentation)`: `ROW_NUMBER() OVER (PARTITION BY id_student, code_presentation ORDER BY code_module ASC) = 1`. Tie-breaker xử lý ca `685015`.
2. Phát hiện ranh giới version: `LAG(age_band) OVER (PARTITION BY id_student ORDER BY code_presentation)` + `IS DISTINCT FROM`.
3. Đánh số version: cumulative `SUM(is_new_version) OVER (...)` — kỹ thuật "gaps and islands".
4. Gộp theo version: `GROUP BY (id_student, version_number)`, `MIN`/`MAX(code_presentation)`.

**EDA quan trọng:** đã kiểm tra toàn bộ 6 cột demographic — **chỉ `age_band`** có ca thay đổi thật (72 sinh viên theo group thô ban đầu); 5 cột còn lại không đổi.

**Ca ngoại lệ `685015`:** 2 dòng cùng `code_presentation = 2014J` khác `code_module` (`CCC`/`DDD`), `age_band` mâu thuẫn — mâu thuẫn **trong cùng 1 kỳ**, không phải thay đổi thật. Xử lý bằng tie-breaker `code_module ASC` (chọn `CCC`).

**Kết quả xác nhận:** Tổng dòng `dim_student` = **28,856**; số `id_student` distinct = **28,785**; chênh lệch **71** sinh viên có 2 version thật; `685015` không nằm trong 71 ca này (đã xử lý dứt điểm ở bước dedup).

#### `dim_course`

**Nguồn:** `stg_bronze_courses`. **Grain:** 1 học phần × 1 kỳ học — đã xác nhận grain đúng (0 dòng trùng lặp).

| Cột | Ý nghĩa | Nguồn |
|---|---|---|
| `course_key` | Surrogate key | Sinh ra — `ROW_NUMBER() OVER (ORDER BY code_module, code_presentation)` |
| `code_module`, `code_presentation` | Mã học phần/kỳ | Có sẵn |
| `module_presentation_length` | Độ dài khóa học (ngày) — dùng sinh `dim_time` | Có sẵn |

#### `dim_assessment`

**Nguồn:** `stg_bronze_assessments`. **Grain:** 1 assessment — đã xác nhận grain đúng.

| Cột | Ý nghĩa | Nguồn |
|---|---|---|
| `assessment_key` | Surrogate key | Sinh ra — `ROW_NUMBER() OVER (ORDER BY id_assessment)` |
| `id_assessment` | ID gốc | Có sẵn |
| `assessment_type` | TMA/CMA/Exam | Có sẵn |
| `weight` | Trọng số điểm | Có sẵn |
| `date` | Hạn nộp — 11/24 bài Exam là NULL, còn lại có giá trị (xem Phần 1) | Có sẵn |
| `course_key` | FK → `dim_course` — cần để dựng "khung" đầy đủ ở `fact_assessment_submission` | Suy ra — join `stg_bronze_assessments` với `dim_course` theo `(code_module, code_presentation)` |

#### `dim_enrollment`

**Nguồn:** `stg_bronze_student_registration` + `stg_bronze_student_info`. **Grain:** 1 sinh viên × 1 học phần × 1 kỳ học. **Vai trò:** cầu nối (conformed dimension) giữa `dim_student`/`dim_course` — mọi fact chỉ lưu `enrollment_key`.

| Cột | Ý nghĩa | Nguồn |
|---|---|---|
| `enrollment_key` | Surrogate key | Sinh ra — `ROW_NUMBER() OVER (ORDER BY id_student, code_module, code_presentation)` |
| `student_key` | FK → `dim_student` | Suy ra — join `id_student` + `code_presentation BETWEEN effective_from_presentation AND effective_to_presentation` (đúng version SCD2 tại thời điểm đăng ký) |
| `course_key` | FK → `dim_course` | Suy ra — join theo `(code_module, code_presentation)` |
| `date_registration` | Ngày đăng ký | Có sẵn |
| `date_unregistration` | Ngày hủy đăng ký | Có sẵn — ⚠️ không dùng làm feature ML |
| `num_of_prev_attempts`, `studied_credits` | Có sẵn | Lấy từ `stg_bronze_student_info` (không có ở `student_registration` gốc) |

**Kết quả xác nhận:** số dòng khớp `stg_bronze_student_registration`; 0 dòng `student_key`/`course_key` NULL; ca `131222` join đúng version theo từng kỳ; ca `685015` ra 2 dòng riêng biệt (2 môn), cùng 1 `student_key`.

#### `dim_time`

**Nguồn:** sinh bằng `generate_series()` + `CROSS JOIN LATERAL`, không đọc trực tiếp 1 bảng nguồn. **Grain:** 1 course/presentation × 1 tuần tương đối.

| Cột | Ý nghĩa | Nguồn |
|---|---|---|
| `time_key` | Surrogate key | Sinh ra |
| `course_key` | FK → `dim_course` | Suy ra |
| `week_number` | Tuần thứ mấy từ khai giảng — `Week 0` = trước khai giảng | Suy ra — `generate_series(0, CEIL(module_presentation_length/7.0))` |

*(Đã bỏ cột `period` — trùng khái niệm với `week_number`.)*

**Công thức quy đổi `date → week_number`** (dùng nhất quán ở `fact_assessment_submission` và `fact_student_behavior`):
```
week_number = 0                      nếu date < 0
week_number = FLOOR(date / 7) + 1    nếu date >= 0
```

**Kết quả xác nhận:** đã kiểm tra chéo nhiều course, `tuan_cuoi_cung` luôn khớp đúng `CEIL(module_presentation_length/7.0)`.

### 4.3. FACT TABLES

#### `fact_student_course_result`

**Grain:** 1 enrollment (= khóa chính `enrollment_key`). **Nguồn:** `stg_bronze_student_info`. **Lưu ý:** đã bỏ hẳn `final_score` — OULAD không có cột điểm tổng hợp cấp course-result.

| Cột | Ý nghĩa | Nguồn |
|---|---|---|
| `enrollment_key` | FK → `dim_enrollment` | Suy ra — join qua `dim_student` (SCD2, có `BETWEEN`) + `dim_course` → `dim_enrollment` |
| `final_result` | Pass/Fail/Withdrawn/Distinction | Có sẵn |
| `is_pass`, `is_fail`, `is_withdrawn`, `is_distinction` | Cờ 1/0 | Suy ra — `CASE WHEN` |

**Kết quả xác nhận:** số dòng = số dòng `dim_enrollment`; 0 dòng `enrollment_key` NULL; mọi dòng có đúng tổng 4 cờ = 1.

#### `fact_assessment_submission`

**Grain:** 1 sinh viên × 1 assessment — **grain đầy đủ** (kể cả cặp chưa từng nộp bài). **Nguồn:** dựng "khung" đầy đủ `dim_enrollment × dim_assessment` (theo `course_key`) rồi LEFT JOIN với `stg_silver_student_assessment_enriched`.

| Cột | Ý nghĩa | Nguồn |
|---|---|---|
| `enrollment_key`, `assessment_key` | FK | Suy ra |
| `time_key` | Theo **hạn nộp** (`dim_assessment.date`); NULL nếu không có hạn | Suy ra — quy đổi `date → week_number` rồi join `dim_time` |
| `is_submitted` | 1/0 — nhờ dựng khung đầy đủ trước LEFT JOIN | Suy ra |
| `is_late` | 1 nếu `date_submitted > date`; NULL nếu không có hạn/chưa nộp | Suy ra |
| `score` | Điểm; NULL nếu chưa nộp | Có sẵn |
| `submission_date` | Ngày nộp thật; NULL nếu chưa nộp | Có sẵn |
| `days_late` | Số ngày trễ; NULL nếu không có hạn/chưa nộp | Suy ra |

**Kết quả xác nhận (đã verify đầy đủ):**
- Tổng: 323,925 dòng = 150,013 (`is_submitted=0`) + 173,912 (`is_submitted=1`).
- `is_submitted=1` khớp đúng tổng dòng gốc `student_assessment.csv` (173,912) — không mất/nhân bản dữ liệu.
- 0 dòng `is_submitted=0` mà vẫn có `score`.
- 19,328 dòng `time_key = NULL` — khớp phép tính độc lập (số enrollment × 11 bài Exam không hạn, theo đúng course).
- 150,013 dòng `is_submitted=0` đều có `is_late`/`days_late` = NULL.

#### `fact_student_behavior`

**Grain:** 1 enrollment × 1 tuần. **INCREMENTAL theo tuần, KHÔNG cộng dồn** (khác Gold). **Nguồn:** gộp Engagement (`stg_silver_student_vle_enriched`, group theo tuần) + Assessment (re-aggregate từ `fact_assessment_submission`, không tính lại từ đầu).

| Cột | Ý nghĩa | Nguồn |
|---|---|---|
| `enrollment_key`, `time_key` | Khóa chính | Suy ra — `FULL OUTER JOIN` giữa 2 nhánh, `COALESCE` lấy khóa từ bên có dữ liệu |
| `total_clicks`, `active_days`, `active_sites`, `activity_types` | Đếm theo tuần | Suy ra — điền 0 nếu tuần đó không có Engagement (biết chắc = 0) |
| `avg_clicks_per_active_day` | `total_clicks / active_days` | Suy ra — giữ NULL nếu `active_days = 0` |
| `num_assessments_assigned`, `num_assessments_submitted`, `late_submission_count` | Theo tuần | Suy ra — từ `fact_assessment_submission` (đã đúng catalog), điền 0 nếu không có |
| `avg_assessment_score` | Điểm TB bài nộp trong tuần | Suy ra — **giữ NULL**, không điền 0 |
| `submission_rate` | `num_submitted / num_assigned` | Suy ra — giữ NULL nếu mẫu số = 0 |

**Kỹ thuật quan trọng:** `FULL OUTER JOIN` (không phải LEFT/INNER) giữa Engagement và Assessment — giữ cả 2 tình huống: có Engagement mà không có bài tới hạn, và ngược lại.

**Kết quả xác nhận:** 688,120 dòng; 0 dòng `enrollment_key`/`time_key` NULL; 0 dòng mâu thuẫn (`total_clicks > 0` mà `active_days = 0`).

### 4.4. Sơ đồ quan hệ

```
dim_student ──┐
              ├──→ dim_enrollment ──┬──→ fact_student_course_result
dim_course ───┴──→ dim_assessment   ├──→ fact_student_behavior ──→ dim_time
                        │            └──→ fact_assessment_submission ──→ dim_assessment
                        └─────────────────────────────────────────────→ dim_time (theo hạn nộp)
```

### 4.5. Nguyên tắc cần nhớ khi viết SQL

1. `dim_student` nhiều dòng/người (SCD2) → luôn `COUNT(DISTINCT id_student)`, không `COUNT(*)`.
2. `num_assessments_assigned` luôn lấy từ `bronze_assessments` (catalog), không suy từ bảng nộp bài.
3. `fact_student_behavior` incremental theo tuần, không cộng dồn.
4. `time_key` trong `fact_assessment_submission` bám theo hạn nộp, không theo ngày nộp thật.
5. `date_unregistration` không dùng làm feature ML — chỉ hợp lệ ở Dashboard.
6. Thứ tự build: Dim độc lập (`dim_student`, `dim_course`) → `dim_assessment` (cần `dim_course`) → Dim phụ thuộc (`dim_enrollment`, `dim_time`) → Fact (`fact_student_course_result` → `fact_assessment_submission` → `fact_student_behavior`).
7. Không phải toàn bộ Exam có `date = NULL` — chỉ ~46% (11/24); vẫn lọc đúng theo `date <= cutoff_day`/`date` như bình thường.

---

## Phần 5 — dbt Tests

### Generic tests (file `_marts.yml`)

Áp dụng cho toàn bộ 8 model:
- `unique`/`not_null` cho mọi surrogate key (`student_key`, `course_key`, `assessment_key`, `enrollment_key`, `time_key`).
- `relationships` cho mọi FK trỏ đúng dimension tương ứng.
- `accepted_values` cho `final_result` (4 giá trị: Pass/Fail/Withdrawn/Distinction) và `is_submitted` (0/1).

### Singular test — canh gác demographic

File `tests/assert_no_student_demographic_conflicts.sql` — loại trừ sẵn ca đã biết (`685015`), tự động fail nếu phát hiện thêm sinh viên có mâu thuẫn demographic mới (ở bất kỳ cột nào trong 6 cột) trong cùng 1 `code_presentation`, phòng khi dữ liệu nguồn thay đổi sau này.

### Kết quả chạy `dbt test` (toàn bộ, không `--select`)

**PASS 100%** — xác nhận `mart_dashboard` đạt chất lượng, sẵn sàng cho tầng ứng dụng (Streamlit).

---

## Phần 6 — Hạ tầng: fix lỗi đã gặp

- **DuckDB `httpfs` — `IOException: Extension not found`:** do `INSTALL httpfs` chạy ở runtime, phụ thuộc mạng lúc container khởi động. Đã sửa: `INSTALL httpfs` chuyển vào `Dockerfile` (build-time, chạy 1 lần, cache sẵn trong image); `gold.py` chỉ còn `LOAD httpfs` (runtime, không cần mạng).
- **`from scratch_eda import week_from` trong `bronze.py`:** dòng import sót lại từ lúc test logic rời, khiến Dagster load definitions lỗi (`scratch_eda.py` tự chạy code kết nối MinIO bằng `localhost:9000` — sai địa chỉ khi chạy trong container). Đã xóa dòng import này — nhắc lại nguyên tắc: code asset chính thức không bao giờ import từ file `scratch_*.py` (không tồn tại trong image build thật).