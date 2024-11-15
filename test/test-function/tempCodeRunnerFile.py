timate_speed(self, im0):
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)

            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = self.track_line[-1]

            # Kiểm tra vị trí
            line = LineString([self.trk_pp[track_id], self.track_line[-1]])
            if line.intersects(self.r_s):
                direction = "known"
            else:
                direction = "unknown"

            if direction in ["known", "unknown"] and track_id not in self.trkd_ids:
                distance = np.linalg.norm(
                    np.array(self.track_line[-1]) - np.array(self.trk_pp[track_id])
                )
                speed = distance  # Cần chuyển đổi đơn vị

                if speed <= 2:
                    speed = 0
                self.spd[track_id] = speed

                if speed <= 2:
                    self.zero_speed_frames[track_id] += 1

                    if (
                        self.zero_speed_frames[track_id] >= self.frame_threshold
                        and track_id not in self.counted_ids
                    ):
                        self.count_zero += 1
                        self.counted_ids.add(track_id)
                        print(f"Vehicle with track ID {track_id} has zero velocity. Total zero count: {self.count_zero}")
                else:
                    self.zero_speed_frames[track_id] = 0

                    self.positive_speed_frames[track_id] += 1

                    if (
                        self.positive_speed_frames[track_id] >= self.frame_threshold
                        and track_id not in self.counted_positive_ids
                    ):
                        self.count_positive += 1
                        self.counted_positive_ids.add(track_id)
                        print(f"Vehicle with track ID {track_id} has positive velocity. Total positive count: {self.count_positive}")

                self.trk_pp[track_id] = self.track_line[-1]

            # Đánh dấu và nhãn
            if direction == "unknown":
                color = (0, 0, 255)  # Màu đỏ cho ngoài vùng
                label = f"{int(self.spd[track_id])} km/h (Out)"
            else:
                color = colors(track_id, True)
                label = f"{int(self.spd[track_id])} km/h"

            self.annotator.box_label(box, label=label, color=color)

        self.display_output(im0)

        return im0