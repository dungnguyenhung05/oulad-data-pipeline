# Feature Dictionary — OULAD Dropout Prediction Pipeline

## Phần 1 — Data Quality Findings (Bronze/Silver EDA)

Khảo sát thực hiện trên toàn bộ 7 bảng Bronze trước khi thiết kế logic Silver (Primary Key, Foreign Key/Referential Integrity, Missing Values, Duplicate, miền giá trị).

- **Referential integrity:** 0 giá trị mồ côi (orphan) ở toàn bộ 5 cặp khóa ngoại đã kiểm tra (`student_registration`/`student_assessment`/`student_vle` → `student_info`; `student_assessment` → `assessments`; `student_vle` → `vle`). → Xác nhận LEFT JOIN ở Silver an toàn, không mất dữ liệu.
- **`student_vle`:** 2,195,960 dòng trùng theo khóa `(id_student, code_module, code_presentation, id_site, date)` (~20.6% tổng số dòng) — đây là đặc điểm gốc của dữ liệu (nhiều lượt ghi nhận click trong cùng ngày), **không phải lỗi**. Không xử lý ở Silver vì Gold sẽ `SUM(sum_click)` theo nhóm nên không ảnh hưởng kết quả.
- **`student_registration`:** đã kiểm tra `duplicated(subset=["id_student","code_module","code_presentation"])` = 0 dòng trùng — khóa tự nhiên này là duy nhất trong dữ liệu gốc OULAD. Vì vậy **không tồn tại asset Silver riêng cho bảng này** (xem ghi chú ở mục Base Population bên dưới) — tương tự lý do đã bỏ asset Silver demographic.
- **`student_registration.date_unregistration`** thiếu 69.1% — có ý nghĩa nghiệp vụ (sinh viên không rút môn), không phải lỗi dữ liệu, không xử lý.
- **`student_assessment.score`:** phát hiện sai kiểu dữ liệu (`object` thay vì số) ở Bronze — đã sửa trong `standard_bronze()`. Sau khi ép kiểu bằng `pd.to_numeric(errors='coerce')`: 173/173,912 dòng (0.1%) không parse được thành số, tỷ lệ chấp nhận được, không điều tra thêm.
- **Miền giá trị:** `final_result`, `gender`, `age_band` đều đúng domain knowledge kỳ vọng, không có giá trị lạ. `score` sau khi sửa: min=0.0, max=100.0 — hợp lệ. `sum_click`: min=1, max=6977, không có giá trị âm.
- **`assessments.date` (Exam):** giá trị gốc là rỗng/NaN cho loại `Exam` (không có hạn nộp cố định) — **giữ nguyên NaN, không ép về 0** ở bất kỳ tầng nào. Ép về 0 sẽ khiến Exam bị hiểu nhầm là "hạn nộp ngày 0", lọt vào mọi cutoff kể cả sớm nhất — đã phát hiện lỗi này khi debug bằng script test độc lập (dùng `.replace('?', 0)` không cẩn thận), không phải lỗi trong pipeline chính.
- **Giá trị ngày âm** (`date`, `date_registration`...) là **bình thường** trong OULAD — tính tương đối so với ngày khai giảng (ngày 0), âm nghĩa là trước khai giảng (ví dụ đăng ký sớm, truy cập hệ thống làm quen trước khi kỳ học bắt đầu).

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
**Base Population:** xuất phát từ **`bronze_student_registration`** (đọc trực tiếp, không qua asset Silver — xem lý do ở Phần 1), LEFT JOIN mọi nguồn khác vào — đảm bảo giữ lại cả sinh viên không tương tác gì.
**Công cụ xử lý:** DuckDB (predicate pushdown, đọc trực tiếp Parquet trên MinIO qua `httpfs`) — thay cho pandas thuần, giải quyết lỗi Out of Memory khi xử lý `silver_student_vle_enriched` (433MB gốc) ở `cutoff8`/`cutoff12`.

### Nhóm Engagement (nguồn: `silver_student_vle_enriched`, lọc `date <= cutoff_day`)

| feature_name | description | data_type | source | calculation | cutoff_rule | leakage_note |
|---|---|---|---|---|---|---|
| `total_click` | Tổng số lượt click VLE tính đến cutoff | int | silver_student_vle_enriched | `SUM(sum_click)` nhóm theo student/module/presentation | Lọc `date <= cutoff_day` **trước** khi SUM (predicate pushdown trong DuckDB) | Nếu quên lọc trước, sẽ đếm cả click tương lai → leakage |
| `active_days` | Số ngày khác nhau có hoạt động | int | silver_student_vle_enriched | `COUNT(DISTINCT date)` | Cùng tập đã lọc cutoff | — |
| `active_site` | Số trang/tài nguyên VLE khác nhau đã truy cập | int | silver_student_vle_enriched | `COUNT(DISTINCT id_site)` | Cùng tập đã lọc cutoff | — |
| `activity_types` | Số loại hoạt động khác nhau (forumng, resource, quiz...) | int | silver_student_vle_enriched | `COUNT(DISTINCT activity_type)` | Cùng tập đã lọc cutoff | — |
| `avg_click_per_day` | Trung bình click mỗi ngày hoạt động | float | phái sinh | `total_click / active_days` | Tính bằng pandas ngay sau khi query trả về (trước khi fillna) | Nếu điền 0 cho `active_days` trước rồi mới chia sẽ gây lỗi chia 0 |
| `last_active_day` | Ngày (tương đối) gần cutoff nhất có hoạt động | int | silver_student_vle_enriched | `MAX(date)` | Cùng tập đã lọc cutoff | Cột trung gian, không bắt buộc dùng trực tiếp làm feature ML |
| `days_since_last_activity` | Số ngày "im lặng" tính đến cutoff | int | phái sinh | `cutoff_day - last_active_day` | Tính ngay sau `last_active_day`, trước fillna | Khi NaN (không hoạt động gì) → điền = `cutoff_day` (im lặng tối đa quan sát được), **không điền 0** |

### Nhóm Assessment — tách 2 nguồn theo đúng grain (nguồn: `bronze_assessments` cấp môn học + `silver_student_assessment_enriched` cấp sinh viên)

> **Quyết định kiến trúc quan trọng (phát hiện qua debug thực tế):** `num_assigned` KHÔNG được tính từ `silver_student_assessment_enriched`, vì bảng đó chỉ chứa các lượt **đã nộp** (grain cấp sự kiện nộp bài). Nếu tính `num_assigned` từ đó, sinh viên **chưa từng nộp bài nào** sẽ bị tính sai thành `num_assigned = 0` (dù môn học thực sự có bài tới hạn) — khiến `submission_rate` bị `NaN` thay vì `0.0`, làm mất tín hiệu "tỷ lệ nộp bài = 0%" quan trọng nhất cho nhóm sinh viên rủi ro cao. Đã sửa bằng cách tính `num_assigned` từ `bronze_assessments` (catalog bài kiểm tra của môn học, không phụ thuộc hành vi từng sinh viên), JOIN vào Base Population theo `(code_module, code_presentation)` — không theo `id_student`.

| feature_name | description | data_type | source | calculation | cutoff_rule | leakage_note |
|---|---|---|---|---|---|---|
| `num_assigned` | Số bài kiểm tra đã tới hạn nộp **của môn/kỳ học** (không phụ thuộc sinh viên có nộp hay không) | int | **bronze_assessments** | `COUNT(DISTINCT id_assessment)` nhóm theo `code_module, code_presentation` | Lọc theo **`date`** (hạn nộp) `<= cutoff_day`; NaN (Exam) tự động bị loại | JOIN theo 2 khóa `(code_module, code_presentation)` vào Base Population — **không** JOIN theo `id_student` |
| `num_submission` | Số bài sinh viên này đã thực sự nộp trước cutoff | int | silver_student_assessment_enriched | `COUNT(date_submitted)` trong tập `submitted` | Lọc `date_submitted <= cutoff_day` | Đây là hành vi cá nhân — khác grain với `num_assigned` |
| `avg_score` | Điểm trung bình các bài đã nộp và có điểm | float | silver_student_assessment_enriched | `AVG(score)` trong tập `submitted` | Cùng tập `submitted` | Giữ NaN nếu chưa có bài nào được chấm — không ép về 0 |
| `submission_rate` | Tỷ lệ đã nộp / đã tới hạn | float | phái sinh (2 nguồn) | `num_submission / num_assigned`, tính **sau khi cả 2 cột đã `fillna(0)`** trong `fill_missing_gold` | Tính sau khi merge cả `assignment_catalog` và `assessment` vào base | `0/0` (môn chưa có bài nào tới hạn) tự nhiên ra NaN; `2/0` (có bài nhưng chưa nộp gì) ra `0.0` — đúng ý nghĩa nghiệp vụ khác nhau, phân biệt được nhờ tách 2 nguồn |
| `num_late_submissions` | Số bài nộp trễ hạn (trước cutoff) | int | silver_student_assessment_enriched | `COUNT(DISTINCT id_assessment)` trong tập `late` (`date_submitted > date`), lọc từ trong `submitted` | Tập `late` lọc **từ trong `submitted`** | Tránh đếm nhầm bài nộp trễ nhưng nộp **sau** cutoff (chưa biết được ở thời điểm dự đoán) |

*Ghi chú riêng: bài Exam không có `date` (hạn nộp) cố định (NaN trong `assessments`) → tự động bị loại khỏi `num_assigned` ở mọi cutoff. Đây là hành vi **đúng mong muốn** (Exam luôn diễn ra sau mọi mốc cutoff sớm), không phải lỗi cần sửa.*

### Nhóm Demographic & Registration (nguồn: `bronze_student_info` + `bronze_student_registration`, KHÔNG cutoff — thuộc tính tĩnh)

> Cả 2 nguồn đọc **trực tiếp từ tầng Bronze** qua DuckDB, không có asset Silver riêng cho chúng — vì cả hai đều không có logic clean/join/dedup thực sự cần thiết (đã kiểm tra `bronze_student_info` không có dòng trùng, `bronze_student_registration` không trùng theo khóa nghiệp vụ ở Phần 1), không thỏa lý do để tồn tại 1 tầng Silver riêng biệt. Cùng nguyên tắc đã áp dụng cho `courses`/`assessments`/`vle` (không có Silver asset riêng cho các bảng danh mục/thuộc tính tĩnh không cần biến đổi thật sự).

| feature_name | description | data_type | source | calculation | cutoff_rule | leakage_note |
|---|---|---|---|---|---|---|
| `age_band` | Nhóm tuổi sinh viên | category | bronze_student_info | lấy trực tiếp | Không áp dụng (thuộc tính tĩnh) | — |
| `imd_band` | Chỉ số mức độ thiếu thốn theo khu vực sinh sống | category | bronze_student_info | lấy trực tiếp | Không áp dụng | — |
| `highest_education` | Trình độ học vấn cao nhất | category | bronze_student_info | lấy trực tiếp | Không áp dụng | — |
| `region` | Vùng miền sinh sống | category | bronze_student_info | lấy trực tiếp | Không áp dụng | — |
| `num_of_prev_attempts` | Số lần đã học lại môn này trước đó | int | bronze_student_info | lấy trực tiếp | Không áp dụng | Tín hiệu rủi ro mạnh, biết được ngay từ đầu kỳ nên không leakage |
| `studied_credits` | Tổng số tín chỉ đang học trong kỳ | int | bronze_student_info | lấy trực tiếp | Không áp dụng | — |
| `disability` | Sinh viên có khuyết tật hay không (Y/N) | category | bronze_student_info | lấy trực tiếp | Không áp dụng | — |
| `date_registration` | Ngày đăng ký môn học (tương đối so với ngày khai giảng) | int | **bronze_student_registration** | lấy trực tiếp, đưa vào Base Population ngay từ đầu (`get_base_population`) | Không áp dụng | Biết trước cutoff sớm nhất (đăng ký luôn diễn ra trước/đầu kỳ) |

### Quy tắc điền giá trị thiếu (áp dụng thống nhất trong `fill_missing_gold`, sau khi đã merge toàn bộ nguồn vào base)

| Nhóm cột | Cách xử lý khi NaN | Lý do |
|---|---|---|
| `total_click, active_days, active_site, activity_types, num_assigned, num_submission, num_late_submissions, num_of_prev_attempts, studied_credits` | Điền **0** | Không tương tác/không có bài = 0 thật, không phải thiếu dữ liệu |
| `age_band, imd_band, highest_education, region, disability` | Điền **"Unknown"** | Categorical thiếu, không có giá trị 0 hợp lý |
| `avg_click_per_day, avg_score, submission_rate` | **Giữ nguyên NaN** | "Không có gì để tính" ≠ "kết quả = 0" — ý nghĩa nghiệp vụ khác nhau |
| `days_since_last_activity` | Điền = **`cutoff_day`** | Nghĩa là "im lặng tối đa có thể quan sát được"; điền 0 sẽ sai thành "vừa mới hoạt động" |

**Thứ tự xử lý quan trọng trong `fill_missing_gold`:** `num_assigned`/`num_submission` phải được `fillna(0)` **trước** khi tính `submission_rate` (nếu không sẽ chia cho NaN); `submission_rate` phải tính **trước** khi coi là xong bước fillna (vì bản thân nó không nằm trong danh sách điền 0).

### Nguyên tắc chống Data Leakage (áp dụng cho toàn bộ Gold Layer)

1. Mọi phép lọc theo cutoff phải thực hiện **trước** khi tổng hợp (aggregate) — với DuckDB, đặt `WHERE` trước `GROUP BY` trong cùng câu SQL để tận dụng predicate pushdown (không tải toàn bộ file vào RAM trước khi lọc).
2. `final_result` **tuyệt đối không** xuất hiện trong bất kỳ cột feature nào của Gold — chỉ dùng làm Target ở tầng ML, lấy riêng từ `bronze_student_info` khi cần.
3. Assessment (hành vi cá nhân): `submitted` phải là tập con của `assigned` (lọc lồng nhau theo `date` rồi `date_submitted`), không lọc độc lập từ bảng gốc.
4. Assessment (catalog): `num_assigned` tính độc lập ở cấp môn học từ `bronze_assessments`, JOIN vào mọi sinh viên cùng môn/kỳ học bất kể họ có nộp bài hay không — tránh nhầm "không nộp gì" thành "không có gì được giao".
5. Mỗi mốc cutoff có 1 model ML riêng biệt — không gộp dữ liệu nhiều mốc để train chung 1 model (cùng giá trị feature nhưng khác ý nghĩa rủi ro tùy mốc thời gian).

### Nguyên tắc chung: khi nào 1 bảng KHÔNG cần asset Silver riêng

Áp dụng nhất quán cho `courses`, `assessments`, `vle`, `student_info` (demographic), và `student_registration` — tất cả đọc thẳng từ Bronze ở các tầng sau, **không có asset Silver trung gian**. Điều kiện để bỏ qua Silver: bảng không cần dedup (đã xác nhận qua EDA), không cần LEFT JOIN với bảng nào khác, không cần biến đổi/enrich cột nào. Nếu chỉ "chọn cột" (column pruning) hoặc "copy nguyên vẹn", việc đó có thể thực hiện trực tiếp ở tầng đọc dữ liệu (DuckDB query trong Gold, hoặc asset Staging) mà không cần thêm 1 tầng trung gian không mang giá trị biến đổi thực sự.

---

## Phần 3 — Postgres Staging

### Schema `staging_ml` (database `oulad_dwh`) — phục vụ nhánh ML

4 bảng Gold có bản sao trong `oulad_dwh.staging_ml`, dùng làm nguồn cho dbt (`mart_ml`): `stg_gold_features_cutoff2/4/8/12`. Cơ chế ghi: Truncate + Insert (Idempotent), tự tạo bảng ở lần chạy đầu tiên nếu chưa tồn tại.

### Schema `staging_dashboard` (database `oulad_dwh`) — phục vụ nhánh Dashboard (đang triển khai)

Nạp trực tiếp từ Silver/Bronze (không qua Gold, không cutoff) để phục vụ `mart_dashboard`:

| Bảng staging | Nguồn | Lý do cần |
|---|---|---|
| `stg_silver_student_vle_enriched` | silver_student_vle_enriched | Xu hướng tương tác VLE theo thời gian (full-period) |
| `stg_silver_student_assessment_enriched` | silver_student_assessment_enriched | Kết quả nộp bài (hành vi cá nhân, full-period) |
| `stg_bronze_student_registration` | bronze_student_registration | Base population + `date_registration` (đổi từ `silver_student_registration_clean` sau khi asset đó bị loại bỏ — xem Phần 1) |
| `stg_bronze_student_info` | bronze_student_info | Demographic + **`final_result`** (Dashboard cần hiển thị Pass/Fail/Withdrawn, khác Gold nghiêm cấm dùng làm feature) |
| `stg_bronze_courses` | bronze_courses | Thông tin môn học cho `dim_course` |
| `stg_bronze_assessments` | bronze_assessments | Catalog bài kiểm tra — **bắt buộc phải staging riêng** vì `dbt-postgres` không đọc được Parquet trên MinIO trực tiếp (khác Gold dùng DuckDB); cần cho `fact_student_performance.num_assigned`, áp dụng đúng nguyên tắc tách nguồn đã học ở Gold (không tính từ `student_assessment`) |

staging_dashboard đã hoàn thành. Dùng kỹ thuật COPY (qua psycopg2 copy_expert) cho bảng stg_silver_student_vle_enriched thay vì to_sql(method="multi") mặc định, giúp giảm thời gian materialize từ 27 phút xuống 50s (do bảng có ~10.6 triệu dòng, INSERT nhiều lô tốn quá nhiều round-trip qua mạng Docker).