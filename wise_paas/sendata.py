import MyWisepaas as toDatahub
import time

# Gửi dữ liệu 60 lần với các giá trị mẫu
for i in range(2):
    timestamp = time.time()
    violations = 10
    density = 20
    toDatahub.sendData("Timestamp", timestamp, "TagViolation", violations, "TagDensity", density)

    time.sleep(1)  # Thời gian nghỉ 1 giây giữa các lần gửi
