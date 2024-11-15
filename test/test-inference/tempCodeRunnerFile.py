    def on_canvas_click(self, event):
        x = event.pos().x()
        y = event.pos().y()
        if len(self.current_polygon) < self.max_points:
            self.current_polygon.append((x, y))
            painter = QPainter(self.canvas.pixmap())
            painter.setPen(QPen(Qt.red, 5))
            painter.drawEllipse(x - 3, y - 3, 6, 6)
            painter.end()
            self.canvas.update()

            self.canvas.setPixmap(self.pixmap.copy())
            self.draw_polygon(self.current_polygon)
            self.logger.info(f"Selected {len(self.current_polygon)} points. Point selected: ({x}, {y})")
            self.logger.info(f"Region selected: {self.current_polygon}")