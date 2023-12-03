from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
import pickle
import functions

class Ui_DepressionDetect(object):
    def setupUi(self, DepressionDetect):
        DepressionDetect.setObjectName("DepressionDetect")
        DepressionDetect.resize(950, 400)

        self.centralwidget = QtWidgets.QWidget(DepressionDetect)
        self.centralwidget.setObjectName("centralwidget")

        # Title
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(250, 50, 700, 100))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(32)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")

        # File Section
        self.fileLayout = QtWidgets.QHBoxLayout()

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.fileLayout.addWidget(self.label_2)

        self.plainTextEdit0 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit0.setObjectName("plainTextEdit")
        self.plainTextEdit0.setMaximumHeight(30)
        self.fileLayout.addWidget(self.plainTextEdit0)

        self.browseButton = QtWidgets.QPushButton("Browse", self.centralwidget)
        self.browseButton.setObjectName("Browse")
        self.browseButton.clicked.connect(self.BrowseHandler)
        self.browseButton.setStyleSheet("QPushButton { background-color: #3498DB; color: white; border: none; padding: 15px; border-radius: 10px; font-size: 14px; }")
        self.fileLayout.addWidget(self.browseButton)

        # Feature Selection and Detection Section
        self.detectLayout = QtWidgets.QHBoxLayout()

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.detectLayout.addWidget(self.label_4)

        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("Eyes Open")
        self.comboBox.addItem("Eyes Closed")
        self.comboBox.addItem("Task")
        self.comboBox.setMaximumHeight(30)
        self.detectLayout.addWidget(self.comboBox)

        self.detectButton = QtWidgets.QPushButton("Predict", self.centralwidget)
        self.detectButton.setObjectName("Detect")
        self.detectButton.clicked.connect(self.DetectHandler)
        self.detectButton.setStyleSheet("QPushButton { background-color: #3498DB; color: white; border: none; padding: 15px; border-radius: 10px; font-size: 14px; }")
        self.detectLayout.addWidget(self.detectButton)

        # Results Section
        self.resultsLayout = QtWidgets.QVBoxLayout()

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.label_3.setAlignment(QtCore.Qt.AlignLeft)
        self.label_3.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.resultsLayout.addWidget(self.label_3)

        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.plainTextEdit.setMaximumHeight(80)  # Larger plain text
        self.resultsLayout.addWidget(self.plainTextEdit)

        # Set up main vertical layout
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.addWidget(self.label)  
        self.verticalLayout.addLayout(self.fileLayout)
        self.verticalLayout.addLayout(self.detectLayout)

        # Create a horizontal layout for results and resultsLayout
        self.resultsDisplayLayout = QtWidgets.QHBoxLayout()
        self.resultsDisplayLayout.addWidget(self.label_3)
        self.resultsDisplayLayout.addLayout(self.resultsLayout)
        self.verticalLayout.addLayout(self.resultsDisplayLayout)

        DepressionDetect.setCentralWidget(self.centralwidget)

        self.retranslateUi(DepressionDetect)
        QtCore.QMetaObject.connectSlotsByName(DepressionDetect)

        self.diagnosis_written = False  # Flag to check if diagnosis is already written

    def retranslateUi(self, DepressionDetect):
        _translate = QtCore.QCoreApplication.translate
        DepressionDetect.setWindowTitle(_translate("DepressionDetect", "Depression Detection"))
        self.label.setText(_translate("DepressionDetect", "Depression Detection"))
        self.label_2.setText(_translate("DepressionDetect", "File:"))
        self.label_4.setText(_translate("DepressionDetect", "State:"))
        self.label_3.setText(_translate("DepressionDetect", "Results:"))

    def BrowseHandler(self):
        self.open_dialog_box()
        self.plainTextEdit.clear()
        self.diagnosis_written = False  # Reset the flag

    def open_dialog_box(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName()
        file = filename
        self.plainTextEdit0.setPlainText(file)
        self.pom = pd.DataFrame()

        try:
            # Try to read features from the file
            self.pom = functions.get_features(file)
            self.pom = self.pom.mean()
        except Exception as e:
            # Handle the error if file type or content is not suitable
            self.plainTextEdit.appendPlainText("Error: Invalid File Format")

    def DetectHandler(self):
        if self.diagnosis_written:
            return  # Avoid redundant diagnosis

        if not hasattr(self, 'pom') or self.pom.empty:
            self.plainTextEdit.appendPlainText("Error: No valid file selected")
            return

        if self.comboBox.currentText() == "Eyes Closed":
            df = pd.DataFrame([1, 0, 0])
        elif self.comboBox.currentText() == "Eyes Open":
            df = pd.DataFrame([0, 1, 0])
        elif self.comboBox.currentText() == "Task":
            df = pd.DataFrame([0, 0, 1])

        df = functions.depression_predict(df, self.pom)
        model = pickle.load(open('bestRFmodel2.sav', 'rb'))
        prediction = model.predict(df)

        if prediction == 0:
            self.plainTextEdit.appendPlainText("NOT DEPRESSED")
        elif prediction == 1:
            self.plainTextEdit.appendPlainText("DEPRESSED")

        self.diagnosis_written = True


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    DepressionDetect = QtWidgets.QMainWindow()
    ui = Ui_DepressionDetect()
    ui.setupUi(DepressionDetect)
    DepressionDetect.show()
    sys.exit(app.exec_())
