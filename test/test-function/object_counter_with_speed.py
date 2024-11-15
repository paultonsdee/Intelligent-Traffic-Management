from object_counter import ObjectCounter
from speed_estimation import SpeedEstimator

class ObjectCounterWithSpeed(ObjectCounter):
    def __init__(self, count_mode='in', speed_threshold=0, fps=30, **kwargs):
        super().__init__(count_mode, **kwargs)
        self.speed_estimator = SpeedEstimator(**kwargs)
        self.speed_threshold = speed_threshold  # Ngưỡng tốc độ (ví dụ: 0 cho xe dừng)
        self.frame_threshold = fps * 5  # Thiết lập khoảng thời gian t (5 giây) theo số khung hình

    def count_objects_with_speed(self, im0, track_id, cls):
        """
        Kiểm tra và đếm các đối tượng dựa trên vận tốc của chúng.

        Args:
            im0 (np.ndarray): Ảnh đầu vào để xử lý.
            track_id (int): ID của đối tượng theo dõi.
            cls (int): Chỉ mục lớp của đối tượng.
        """
        speed = self.speed_estimator.spd.get(track_id, 0)
        is_stationary = speed <= self.speed_threshold

        # Đếm nếu đối tượng dừng trong khoảng thời gian frame_threshold
        if is_stationary:
            self.object_frame_counters[track_id]['stationary'] += 1
            if self.object_frame_counters[track_id]['stationary'] >= self.frame_threshold:
                self.classwise_counts[self.names[cls]]["STATIONARY"] += 1
        else:
            self.object_frame_counters[track_id]['stationary'] = 0  # Reset nếu đối tượng di chuyển

    def count(self, im0, fps=0, t=0):
        super().count(im0, fps, t)
        self.speed_estimator.estimate_speed(im0)
        
        # Đếm thêm dựa trên vận tốc
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.count_objects_with_speed(im0, track_id, cls)

        return im0