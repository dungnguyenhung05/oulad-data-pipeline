## Data Quality Findings (Bronze/Silver EDA)

- Referential integrity: 0 giá trị mồ côi ở toàn bộ 5 cặp khóa ngoại đã kiểm tra
  (student_registration/student_assessment/student_vle -> student_info,
  student_assessment -> assessments, student_vle -> vle).
  => Xác nhận LEFT JOIN ở Silver an toàn, không mất dữ liệu.
- student_vle: 2,195,960 dòng trùng theo khóa (student, module, presentation,
  site, date) - đây là đặc điểm gốc của dữ liệu (nhiều lượt ghi nhận click
  trong cùng ngày), không phải lỗi. Không xử lý ở Silver vì Gold sẽ SUM
  sum_click nên không ảnh hưởng kết quả.
- student_registration.date_unregistration thiếu 69.1% - có ý nghĩa nghiệp vụ
  (sinh viên không rút môn), không phải lỗi dữ liệu.
- student_assessment.score: phát hiện sai kiểu dữ liệu (object thay vì số)
  ở Bronze, đã sửa trong standard_bronze(). Sau khi sửa: 173/173,912 dòng
  (0.1%) không parse được thành số, chấp nhận được.