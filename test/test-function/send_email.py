import smtplib

# Thông tin tài khoản gửi email
from_email_nickname = "Traffic Management System (TMS) <vhp08072004@gmail.com>"
from_email = "vhp08072004@gmail.com"
password = "wayp qcui muhs ietx"
to_email = "vhp08071974@gmail.com"
subject = "Violation Alert"
message_body = "In the past 20 seconds, 1 motorbike have encroached the lane."

# Kết nối đến máy chủ SMTP của Gmail
server = smtplib.SMTP("smtp.gmail.com", 587)
server.starttls()  # Kích hoạt chế độ bảo mật TLS

try:
    # Đăng nhập vào tài khoản Gmail
    server.login(from_email, password)
    
    # Tạo nội dung email với tên hiển thị
    message = f"From: {from_email_nickname}\nTo: {to_email}\nSubject: {subject}\n\n{message_body}"
    
    # Gửi email
    server.sendmail(from_email, to_email, message)
    print("Email đã được gửi thành công!")
except Exception as e:
    print(f"Không thể gửi email: {e}")
finally:
    # Đóng kết nối với server
    server.quit()
