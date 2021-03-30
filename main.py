# -*- coding: utf-8 -*-
import sys
import os
import csv
import warnings
# ignore the warnings
# warnings.filterwarnings('ignore')

import cv2

from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QHeaderView, QProgressBar
from PyQt5.QtWidgets import QMessageBox, QAbstractItemView, QTableWidgetItem
from PyQt5 import QtCore
from PyQt5.QtCore import QThread, pyqtSignal
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')

from cell_analysis import InferenceConfig
from cell_analysis import cell_analysis
from mrcnn import model as modellib

QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

class PyQtMainEntry(QMainWindow):

    def __init__(self):
        super().__init__()
        # load the ui from the file
        self.ui = uic.loadUi("ui/main.ui")
        self.result_window = uic.loadUi("ui/result.ui")
        # initialize the signal/slot
        self.signal_init()
        # initialize the model
        self.model_init()

    def signal_init(self):
        # initialize the signal/slot function
        # mainWindow signal initialized
        self.ui.browser_btn.clicked.connect(self.file_Browser_Clicked)
        self.ui.analysis_btn.clicked.connect(self.startAnalysis_Clicked)
        self.ui.export_btn.clicked.connect(self.export_Clicked)
        self.ui.add_btn.clicked['bool'].connect(self.add_item_Clicked)
        self.ui.sub_btn.clicked.connect(self.sub_item_Clicked)
        self.ui.clear_btn.clicked.connect(self.clear_item_Clicked)
        self.ui.filelist.itemClicked['QListWidgetItem*'].connect(self.item_Clicked)

        # result window signal initialized
        self.result_window.check_btn.clicked.connect(self.check_Clicked)
        self.result_window.export_table_btn.clicked.connect(self.export_table_Clicked)

    def model_init(self):
        # load the maskrcnn model
        config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode='inference', config=config, model_dir='logs')
        weights_path = 'logs/cell/mask_rcnn_cell_0060.h5'
        print("Loading weights ", weights_path)
        self.model.load_weights(weights_path, by_name=True)
        print('Load the model sucessfully')
        # self.model.detect()
        self.model.keras_model._make_predict_function()

    def file_Browser_Clicked(self):
        print('file_Browser_Clicked clicked')
        self.files, _ = QFileDialog.getOpenFileNames(self, '打开图片')
        self.ui.lineEdit.setText(os.path.dirname(self.files[0]))
        files_basename = [os.path.basename(x) for x in self.files]
        self.ui.filelist.addItems(files_basename)

        # show the first image in the ImageViewer
        self.img = cv2.imread(self.files[0])
        self.ui.graphWidget.setImage(self.img)

    def startAnalysis_Clicked(self):
        # start analysis the select img

        print('分析中...')
        self.ui.progressBar.setRange(0,0)
        self.ui.analysis_btn.setText('分析中...')
        # ********************************** single thread function ******************************************
        # self.cell_obj = cell_analysis(self.model, self.img, calibration=self.ui.doubleSpinBox.text())
        # # show img
        # self.result_window.widget.setImage(self.cell_obj.detect_img)
        # # show cell count result table
        # self.set_result_table(self.cell_obj.merge_measurements())

        # *********************************** multi thread function ******************************************
        # start a thread to analysis
        self.analysis_thread = WorkThread(self.img, self.model, self.ui.doubleSpinBox.text())
        self.analysis_thread.trigger.connect(self.show_results)
        self.analysis_thread.finished.connect(self.onFinished)
        self.analysis_thread.start()

        # **************************************************************************************************

    def onFinished(self):
        # while complete the cell analysis, clear progressbar
        self.ui.progressBar.setRange(0,1)
        self.ui.analysis_btn.setText('分析')

    def show_results(self, cell_obj):
        '''
        Args:
            cell_obj: the thread emit results

        Returns:
            set Image in the result widget and show the detect paramater in the table of result widget
        '''
        self.cell_obj = cell_obj
        # show img
        self.result_window.widget.setImage(cell_obj.detect_img)
        # show cell count result table
        self.set_result_table(cell_obj.merge_measurements())
        self.result_window.show()

    def set_result_table(self, data):
        # set the result table in the result window widget
        cow_label = list(data[0].keys())
        cow_name = ['细胞ID', '长度(um)','宽度(um)', '线粒体个数', '总长度(um)', '各段长度(um)', '节点个数']
        cow_number = len(data[0])
        row_number = len(data)

        self.result_window.tableWidget.setColumnCount(cow_number)
        self.result_window.tableWidget.setRowCount(row_number)

        # se the header label in the horizontal direction and the header label in the vertical direction
        self.result_window.tableWidget.setHorizontalHeaderLabels(cow_name)
        # set the horizontal diretion table to the adaptive stretch mode
        self.result_window.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # set the entire row of the table to be selected
        self.result_window.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        # set full of the widget
        self.result_window.tableWidget.horizontalHeader().setStretchLastSection(True)

        # add data into the table
        for row in range(row_number):
            for cow in range(cow_number):
                cur_cell = data[row]
                print(cur_cell[cow_label[cow]])
                newItem = QTableWidgetItem(str(cur_cell[cow_label[cow]]))
                self.result_window.tableWidget.setItem(row, cow, newItem)

    def export_Clicked(self):
        # batching anlaysis the cell imgs
        print('batching analysis btn clicked')
        output_dir = QFileDialog.getExistingDirectory(self, 'Choose a Directory to save the result')
        self.ui.progressBar.setRange(0, 0)
        # batchAnalysis_Thread = batchAnalysis_Thread(self.files, self.model,
        #                                             self.ui.doubleSpinBox.text(), output_dir)
        batch_Thread = batchAnalysis_Thread(self.files, self.model,
                                 self.ui.doubleSpinBox.text(), output_dir)
        batch_Thread.trigger.connect(self.batch_Thread_slots)
        batch_Thread.flag.connect(self.batch_Thread_flag)
        batch_Thread.start()
        batch_Thread.exec_()

    def batch_Thread_slots(self, value):
        self.ui.progressBar.setRange(0, 100)
        self.ui.progressBar.setValue(value)

    def batch_Thread_flag(self, flag):
        self.ui.progressBar.reset()
        if flag:
            QMessageBox.information(self, '提示', '保存成功！')

    def add_item_Clicked(self):
        # while the add button "+" pressed
        # open the file dialog and append new img into filelistWidget
        files, _ = QFileDialog.getOpenFileNames(self, '选择图片')
        self.ui.lineEdit.setText(os.path.dirname(files[0]))
        file_basename = [os.path.basename(x) for x in files]
        self.ui.filelist.addItems(file_basename)

    def sub_item_Clicked(self):
        # while the add button "-" pressed
        # delete current selected items
        for item in self.ui.filelist.selectedItems():
            self.ui.filelist.takeItem(self.ui.filelist.row(item))


    def clear_item_Clicked(self):
        print('clear_item_Clicked clicked')
        # while "clear" button is cliked, clear the list
        self.ui.filelist.clear()

    def item_Clicked(self):
        # if item clicked, change the ImageView content to current select image
        print('item_Clicked clicked')
        cur_filename = self.ui.filelist.selectedItems()[0].text()
        print(cur_filename)
        cur_filename = os.path.join(self.ui.lineEdit.text(), cur_filename)
        print(cur_filename)
        self.img = cv2.imread(cur_filename)
        self.ui.graphWidget.setImage(self.img)

    def check_Clicked(self):
        # print('check_Clicked clicked')
        # plot the current row responding cell process
        cur_cell_id = self.result_window.tableWidget.currentRow()
        self.cell_obj.show_specific_cell(cell_id=cur_cell_id)

    def export_table_Clicked(self):
        # print('export table clicked')
        # export the result table and img
        output_dir = QFileDialog.getExistingDirectory(self, 'Open Directory to save the result')
        # result save
        if output_dir:
            try:
                # 1. origin img save
                origin_img_name = os.path.join(output_dir, 'origin.png')
                cv2.imwrite(origin_img_name, img=self.img)
                # 2. detect img save
                detect_img_name = os.path.join(output_dir, 'detect.png')
                cv2.imwrite(detect_img_name, img=self.cell_obj.detect_img)
                # 3. cell morphological and mitochondrial infos save
                cell_infos = self.cell_obj.merge_measurements()
                cell_infos_name = os.path.join(output_dir, 'cell_infos.csv')
                with open(cell_infos_name, 'w', newline='') as csvfile:
                    fieldnames = list(cell_infos[0].keys())
                    fieldnames = ['Cell_Id', 'Length(um)', 'Width(um)', 'mitochondrial_numbers',
                                  'mitochondrial_overall_length(um)', 'mitochondrial_each_segment_len(um)',
                                  'mitochondrial_each_seg_intersection_num']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(cell_infos)
                print('save sucessfully')
                QMessageBox.information(self, '提示', 'Save sucessfully!')

            except:
                QMessageBox.information(self, '提示', 'Save Failure, Please check your save directory!')

class WorkThread(QThread):
    trigger = pyqtSignal(object)
    def __init__(self, img, model, calibration):
        super(WorkThread, self).__init__()
        self.img = img
        self.model = model
        self.calibration = calibration

    def run(self):
        print('start analysis the cell')
        self.cell_obj = cell_analysis(self.model, self.img, calibration=self.calibration)
        self.trigger.emit(self.cell_obj)

class batchAnalysis_Thread(QThread):
    trigger = pyqtSignal(int)
    flag = pyqtSignal(bool)
    def __init__(self, imgs_dir, model, calibration, output_dir):
        super(batchAnalysis_Thread, self).__init__()
        self.files = imgs_dir
        self.model = model
        self.calibration = calibration
        self.output_dir = output_dir

    def run(self):
        print('start batching analysis...')
        if self.output_dir:
            # batch process
            print(self.files)
            nums_file = len(self.files)
            progress = 0
            for file in self.files:
                img = cv2.imread(file)
                cell_obj = cell_analysis(self.model, img, calibration=self.calibration)

                # 1. mkdir
                dir_path = os.path.join(self.output_dir, os.path.basename(file).split('.')[0])
                print('current path dir', dir_path)
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                # 1. origin img save
                origin_img_name = os.path.join(dir_path, 'origin.png')
                cv2.imwrite(origin_img_name, img=img)
                # 2. detect img save
                detect_img_name = os.path.join(dir_path, 'detect.png')
                cv2.imwrite(detect_img_name, img=cell_obj.detect_img)
                # 3. cell morphological and mitochondrial infos save
                cell_infos = cell_obj.merge_measurements()
                cell_infos_name = os.path.join(dir_path, 'cell_infos.csv')
                with open(cell_infos_name, 'w', newline='') as csvfile:
                    fieldnames = list(cell_infos[0].keys())
                    fieldnames = ['Cell_Id', 'Length(um)', 'Width(um)', 'mitochondrial_numbers',
                                  'mitochondrial_overall_length(um)', 'mitochondrial_each_segment_len(um)',
                                  'mitochondrial_each_seg_intersection_num']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(cell_infos)
                progress += 100 // nums_file
                self.trigger.emit(progress)
            self.flag.emit(True)
        else:
            print('Not choose a output dir, batch analysis cancelled')



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PyQtMainEntry()
    window.ui.show()
    sys.exit(app.exec_())