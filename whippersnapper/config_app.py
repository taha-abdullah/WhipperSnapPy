#!/usr/bin/python3

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, \
                            QMessageBox, QSlider, QLineEdit, QGroupBox

from PyQt5.QtCore import Qt


class ConfigWindow(QWidget):
    def __init__(self, parent=None, screen_dims=None, initial_fthresh_value=2.0,
                 initial_fmax_value=4.0):
        super(ConfigWindow, self).__init__(parent)

        self.current_fthresh_value = initial_fthresh_value
        self.current_fmax_value = initial_fmax_value
        self.screen_dims = screen_dims
        self.window_size = (400, 200)

        layout = QVBoxLayout()

        self.slider_tick_limits = (0.0, 1000.0)
        self.slider_value_limits = (0.0, 10.0)
        self.fthresh_slider = QSlider(Qt.Horizontal)
        self.fthresh_slider.setMinimum(int(self.slider_tick_limits[0]))
        self.fthresh_slider.setMaximum(int(self.slider_tick_limits[1]))
        self.fthresh_slider.setValue(int(self.convert_value_to_range(self.current_fthresh_value, 
                                                                  self.slider_value_limits, 
                                                                  self.slider_tick_limits)))
        self.fthresh_slider.setTickPosition(QSlider.TicksBelow)
        self.fthresh_slider.setTickInterval(int((self.slider_tick_limits[1] - self.slider_tick_limits[0]) / 20.))
        self.fthresh_slider.valueChanged.connect(self.fthresh_slider_value_cb)

        self.fthresh_value_box = QLineEdit('Threshold')
        self.fthresh_value_box.setText(str(self.current_fthresh_value))
        self.fthresh_value_box.textChanged.connect(self.fthresh_value_cb)

        self.fmax_slider_tick_limits = (0.0, 1000.0)
        self.fmax_slider_value_limits = (0.0, 10.0)
        self.fmax_slider = QSlider(Qt.Horizontal)
        self.fmax_slider.setMinimum(int(self.fmax_slider_tick_limits[0]))
        self.fmax_slider.setMaximum(int(self.fmax_slider_tick_limits[1]))
        self.fmax_slider.setValue(int(self.convert_value_to_range(self.current_fmax_value, 
                                                                  self.fmax_slider_value_limits, 
                                                                  self.fmax_slider_tick_limits)))
        self.fmax_slider.setTickPosition(QSlider.TicksBelow)
        self.fmax_slider.setTickInterval(int((self.fmax_slider_tick_limits[1] - self.fmax_slider_tick_limits[0]) / 20.))
        self.fmax_slider.valueChanged.connect(self.fmax_slider_value_cb)

        self.fmax_value_box = QLineEdit('Threshold')
        self.fmax_value_box.setText(str(self.current_fmax_value))
        self.fmax_value_box.textChanged.connect(self.fmax_value_cb)

        ## Group boxes layout:
        fthresh_box = QGroupBox('Thresh')
        fthresh_box_layout = QVBoxLayout()
        fthresh_box_layout.addWidget(self.fthresh_slider)
        fthresh_box_layout.addWidget(self.fthresh_value_box)
        fthresh_box.setLayout(fthresh_box_layout)
        fthresh_box.setStyleSheet('QGroupBox  {color: blue;}')

        fmax_box = QGroupBox('Max')
        fmax_box_layout = QVBoxLayout()
        fmax_box_layout.addWidget(self.fmax_slider)
        fmax_box_layout.addWidget(self.fmax_value_box)
        fmax_box.setLayout(fmax_box_layout)
        fmax_box.setStyleSheet('QGroupBox  {color: green;}')

        thresholds_box = QGroupBox('Thresholds')
        thresholds_box_layout = QHBoxLayout()
        thresholds_box_layout.addWidget(fthresh_box)
        thresholds_box_layout.addWidget(fmax_box)
        thresholds_box.setLayout(thresholds_box_layout)

        layout.addWidget(thresholds_box)

        self.setWindowTitle('WhipperSnapper Configuration')
        self.setLayout(layout)
        if self.screen_dims is not None:
            self.setGeometry(screen_dims[0] - int(screen_dims[0] / 5), 
                             int(screen_dims[1] / 8), self.window_size[0], 
                             self.window_size[1])
        else:
            self.setGeometry(0, 0, self.window_size[0], self.window_size[1])

    def fthresh_slider_value_cb(self):
        self.current_fthresh_value = self.convert_value_to_range(self.fthresh_slider.value(),
                                                                 self.slider_tick_limits, 
                                                                 self.slider_value_limits)
        self.fthresh_value_box.setText(str(self.current_fthresh_value))

    def fthresh_value_cb(self, new_value):
        # Do not react to invalid values:
        try:
            new_value = float(new_value)
        except ValueError:
            return

        self.current_fthresh_value = float(new_value)

        slider_fthresh_value = self.convert_value_to_range(self.current_fthresh_value,
                                                           self.slider_value_limits, 
                                                           self.slider_tick_limits)
        self.fthresh_slider.setValue(int(slider_fthresh_value))

    def fmax_slider_value_cb(self):
        self.current_fmax_value = self.convert_value_to_range(self.fmax_slider.value(),
                                                              self.slider_tick_limits, 
                                                              self.slider_value_limits)
        self.fmax_value_box.setText(str(self.current_fmax_value))

    def fmax_value_cb(self, new_value):
        # Do not react to invalid values:
        try:
            new_value = float(new_value)
        except ValueError:
            return

        self.current_fmax_value = float(new_value)

        slider_fmax_value = self.convert_value_to_range(self.current_fmax_value,
                                                        self.slider_value_limits, 
                                                        self.slider_tick_limits)
        self.fmax_slider.setValue(int(slider_fmax_value))

    def convert_value_to_range(self, value, old_limits, new_limits):
        old_range = old_limits[1] - old_limits[0]
        new_range = new_limits[1] - new_limits[0]
        new_value = (((value - old_limits[0]) * new_range) / old_range) + new_limits[0]

        return new_value

    def get_fthresh_value(self):
        return self.current_fthresh_value

    def get_fmax_value(self):
        return self.current_fmax_value
