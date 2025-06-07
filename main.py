from image_preprocessing import determine_tiling_dimensions, compute_tile_size
from PIL import Image

# from tiling import TileInator
import os  # Import the os module

# from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QFontDatabase, QPixmap, QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QWidget,
    QSplashScreen,
)
from PyQt5.QtCore import QObject, QEvent, Qt
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal

TileInator = None


class ModelLoader(QThread):
    finished = pyqtSignal()
    status_update = pyqtSignal(str)

    def run(self):
        global TileInator

        try:
            # Import tiling module with full model loading
            from tiling import TileInator as TI, load_models

            # Load all models directly during splash screen
            self.status_update.emit("Loading model components...")
            load_models(self.status_update.emit)

            # Make TileInator available globally
            TileInator = TI

            self.finished.emit()
        except Exception as e:
            self.status_update.emit(f"Error loading models: {str(e)}")


class SplashScreen(QSplashScreen):
    def __init__(self):
        # Create splash screen with logo
        logo_path = os.path.join("resources", "gui", "logo.png")
        pixmap = QtGui.QPixmap(600, 300)
        pixmap.fill(QtGui.QColor("#1E1E20"))
        super().__init__(pixmap)

        # Setup layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Add logo
        self.logo = QtWidgets.QLabel()
        self.logo.setPixmap(
            QtGui.QPixmap(logo_path).scaled(100, 100, QtCore.Qt.KeepAspectRatio)
        )
        self.logo.setAlignment(QtCore.Qt.AlignHCenter)
        layout.addWidget(self.logo)

        # Add title
        self.title = QtWidgets.QLabel("AI-No-Swiping")
        font = QtGui.QFont()
        font.setFamily("Georama Black")  # Uses custom Georama Black font
        font.setPointSize(24)
        font.setBold(True)
        self.title.setFont(font)
        self.title.setStyleSheet("color: #27c9bb; font-size: 24px; font-weight: bold;")
        self.title.setAlignment(QtCore.Qt.AlignHCenter)
        layout.addWidget(self.title)

        # Add status
        self.status = QtWidgets.QLabel(
            "Wohoi teka lang! The models are still being loaded. Please wait..."
        )
        self.status.setStyleSheet("color: white;")
        self.status.setAlignment(QtCore.Qt.AlignHCenter)
        layout.addWidget(self.status)

        # Add progress bar
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(
            """
                QProgressBar{
                    background-color: #232323;
                    border-radius: 10px;
                    text-align: center;
                    padding: 1px;
                    height: 12px;
                }
                QProgressBar::chunk {
                    background-color: #27c9bb;
                    border-radius: 9px;
                    margin: 1px;
                }
            """
        )
        layout.addWidget(self.progress)

        self.setLayout(layout)

        # Keep on top and make modal
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.Tool
        )

    def update_status(self, message):
        self.status.setText(message)
        QApplication.processEvents()


# Class for processing the perturbation. Needs to be separate from GUI to prevent GUI from not responding.
class ProcessingThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # Success flag and message
    image_processed = pyqtSignal(str)  # signal for each processed image path

    def __init__(self, image_paths, output_path, intensity, segmentation, prompt):
        super().__init__()
        self.image_paths = image_paths
        self.output_path = output_path
        self.intensity = intensity
        self.segmentation = segmentation
        self.prompt = prompt

    def run(self):
        try:
            # Process each selected image
            for image_path in self.image_paths:
                try:
                    # Extract filename for display
                    filename = os.path.basename(image_path)
                    self.progress.emit(f"Processing image: {filename}")

                    # Load the image
                    image = Image.open(image_path)

                    # Get tiling dimensions
                    rows, cols = determine_tiling_dimensions(image, self.segmentation)
                    tile_width, tile_height = compute_tile_size(image, rows, cols)
                    overlap_size = int(tile_height * 0.1)

                    # Process with TileInator
                    global TileInator
                    input_image_processor = TileInator(
                        overlap_size=overlap_size,
                        image=image,
                        tile_width=tile_width,
                        tile_height=tile_height,
                        num_cols=cols,
                        num_rows=rows,
                        intensity=self.intensity,
                        prompt=self.prompt,
                        filename=filename,
                        output_path=self.output_path,
                        progress_callback=self.progress.emit,
                    )

                    # Process image
                    output_path = input_image_processor.process_image()
                    self.image_processed.emit(output_path)
                    self.progress.emit(f"✓ Finished processing {filename}")

                except Exception as e:
                    self.progress.emit(f"❌ Error processing {filename}: {str(e)}")

            self.finished.emit(
                True, f"Successfully processed all {len(self.image_paths)} image(s)"
            )

        except Exception as e:
            self.finished.emit(False, f"Processing failed: {str(e)}")


class Ui_MainWindow(QObject):  # Make it inherit from QObject for event handling
    def __init__(self):
        super().__init__()  # Initialize the QObject
        self.selected_image_paths = []
        self.perturbed_image_paths = []
        self.intensity = 2
        self.segmentation = 2
        self.ai_description = ""
        self.output_path = ""
        self.progress_lines = (
            []
        )  # all strings to be shown in the progress screen will be stored here

    def setupUi(self, MainWindow):
        icon_path = self.resource_path("resources/gui/logo.png")
        app_icon = QtGui.QIcon(icon_path)
        MainWindow.setWindowIcon(app_icon)

        MainWindow.setWindowFlags(
            QtCore.Qt.Window
            | QtCore.Qt.CustomizeWindowHint
            | QtCore.Qt.WindowCloseButtonHint  # Allow closing
            | QtCore.Qt.WindowMinimizeButtonHint  # Allow minimizing
        )
        self.main_window = MainWindow  # Store reference to the main window
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1205, 800)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setStyleSheet("background-color: #1E1E20;")
        MainWindow.setDocumentMode(False)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        MainWindow.setDockNestingEnabled(False)

        # === CENTRAL WIDGET ===
        # This is the main container for all UI elements
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # === HEADER SECTION ===
        # Contains the title and subtitle in a vertical layout
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(100, 20, 241, 54))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")

        # Vertical layout for heading and subheading
        self.heading_subheading = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.heading_subheading.setSizeConstraint(
            QtWidgets.QLayout.SetDefaultConstraint
        )
        self.heading_subheading.setContentsMargins(0, 0, 0, 0)
        self.heading_subheading.setSpacing(
            0
        )  # No spacing between heading and subheading
        self.heading_subheading.setObjectName("heading_subheading")

        # Main application title - "AI-No-Swiping" in teal color
        self.ains_heading = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Georama Black")  # Uses custom Georama Black font
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.ains_heading.setFont(font)
        self.ains_heading.setScaledContents(True)
        self.ains_heading.setObjectName("ains_heading")
        self.heading_subheading.addWidget(self.ains_heading)

        # Subtitle text - "Protect your art from AI misuse!"
        self.ains_subheading = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Inter")  # Uses custom Inter font
        self.ains_subheading.setFont(font)
        self.ains_subheading.setObjectName("ains_subheading")
        self.heading_subheading.addWidget(self.ains_subheading)

        # === MAIN TAB WIDGET ===
        # Contains all three tabs of the application workflow
        self.perturb_tabs = QtWidgets.QTabWidget(self.centralwidget)
        self.perturb_tabs.setGeometry(QtCore.QRect(20, 80, 650, 671))
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum
        )
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.perturb_tabs.sizePolicy().hasHeightForWidth())
        self.perturb_tabs.setSizePolicy(sizePolicy)
        self.perturb_tabs.setMaximumSize(QtCore.QSize(1161, 16777215))
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.perturb_tabs.setFont(font)
        self.perturb_tabs.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.perturb_tabs.setMouseTracking(False)
        self.perturb_tabs.setAcceptDrops(False)
        self.perturb_tabs.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.perturb_tabs.setAutoFillBackground(False)
        self.perturb_tabs.setStyleSheet("")
        self.perturb_tabs.setTabPosition(QtWidgets.QTabWidget.West)
        self.perturb_tabs.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.perturb_tabs.setUsesScrollButtons(True)
        self.perturb_tabs.setDocumentMode(True)
        self.perturb_tabs.setTabsClosable(False)
        self.perturb_tabs.setMovable(False)
        self.perturb_tabs.setTabBarAutoHide(False)
        self.perturb_tabs.setObjectName("perturb_tabs")

        # Hide tab bar completely
        self.perturb_tabs.tabBar().hide()

        # Set tab bar stylesheet - adjust these values to customize the look
        self.tab_stylesheet = """
            /* Main tab widget */
            QTabWidget::pane {
                border: none; 
                background-color: transparent;
            }
        """
        self.perturb_tabs.setStyleSheet(self.tab_stylesheet)

        # === TAB 1: ATTACH IMAGES ===
        # First tab where users upload their artwork
        self.attach = QtWidgets.QWidget()
        self.attach.setObjectName("attach")

        # Layout widget for the heading and subheading in the Attach tab
        self.layoutWidget = QtWidgets.QWidget(self.attach)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 20, 591, 75))
        self.layoutWidget.setObjectName("layoutWidget")

        # Horizontal layout containing the title section
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        # Vertical layout for tab title and subtitle
        self.attach_title = QtWidgets.QVBoxLayout()
        self.attach_title.setSpacing(0)  # No spacing between heading and subheading
        self.attach_title.setObjectName("attach_title")

        # Tab heading - "Upload your art."
        self.attach_heading = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(28)
        font.setBold(True)
        font.setWeight(75)
        self.attach_heading.setFont(font)
        self.attach_heading.setScaledContents(False)
        self.attach_heading.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop
        )
        self.attach_heading.setWordWrap(False)
        self.attach_heading.setIndent(0)
        self.attach_heading.setObjectName("attach_heading")
        self.attach_title.addWidget(self.attach_heading)

        # Tab subheading - description text about accepted file formats
        self.attach_subheading = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(10)
        font.setKerning(True)
        self.attach_subheading.setFont(font)
        self.attach_subheading.setObjectName("attach_subheading")
        self.attach_title.addWidget(self.attach_subheading)

        # Add the title vertical layout to the horizontal layout
        self.horizontalLayout.addLayout(self.attach_title)

        # === IMAGE UPLOAD AREA ===
        # Create a drop area for images
        self.image_drop_area = QtWidgets.QFrame(self.attach)
        self.image_drop_area.setGeometry(QtCore.QRect(20, 120, 591, 200))
        self.image_drop_area.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.image_drop_area.setFrameShadow(QtWidgets.QFrame.Raised)
        self.image_drop_area.setStyleSheet(
            """
            QFrame {
                border: none;
                border-radius: 12px;
                background-color: #2F2F2F;
            }
            QFrame:hover {
                border: 2px solid #27c9bb;
            }
        """
        )
        # Sets the cursor to a pointing hand when hovering over the drop area
        self.image_drop_area.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.image_drop_area.setAcceptDrops(True)
        self.image_drop_area.installEventFilter(self)
        self.image_drop_area.setObjectName("image_drop_area")

        # Vertical layout for drop area content
        self.drop_area_layout = QtWidgets.QVBoxLayout(self.image_drop_area)

        # Icon for the drop area
        self.drop_icon = QtWidgets.QLabel()
        self.drop_icon.setAlignment(QtCore.Qt.AlignCenter)
        self.drop_icon.setText("📁")  # Simple folder icon using emoji
        font = QtGui.QFont()
        font.setPointSize(32)
        self.drop_icon.setFont(font)
        self.drop_icon.setStyleSheet("border: none;")
        self.drop_area_layout.addWidget(self.drop_icon)

        # Text instructions
        self.drop_text = QtWidgets.QLabel()
        self.drop_text.setAlignment(QtCore.Qt.AlignCenter)
        self.drop_text.setText("Drag and drop image files here\nor click to browse")
        self.drop_text.setStyleSheet("border: none; color: #999999; font-size: 12px;")
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(12)
        self.drop_text.setFont(font)
        self.drop_area_layout.addWidget(self.drop_text)

        # Selected files list
        self.selected_files_label = QtWidgets.QLabel(self.attach)
        self.selected_files_label.setGeometry(QtCore.QRect(20, 330, 591, 30))
        self.selected_files_label.setText("Selected Files: None")
        self.selected_files_label.setStyleSheet("color: #ffffff;")
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(10)
        self.selected_files_label.setFont(font)

        self.attach_reset_btn = QtWidgets.QPushButton(self.attach)
        self.attach_reset_btn.setGeometry(QtCore.QRect(280, 360, 160, 40))
        self.attach_reset_btn.setText("RESET")
        self.attach_reset_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #FFFFFF;
                color: #000000;
                border-radius: 4px;
                font-weight: bold;
                font-family: "Inter";
                font-size: 10px;
                padding: 8px 16px;
            }
            QPushButton:pressed {
                background-color: #d7dbd8;
            }
            QPushButton:disabled {
                background-color: #424242;
                color: #777777;
            }
        """
        )
        self.attach_reset_btn.setEnabled(False)
        self.attach_reset_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.attach_reset_btn.clicked.connect(self.reset_attached_images)

        # "Next" button
        self.attach_next_btn = QtWidgets.QPushButton(self.attach)
        self.attach_next_btn.setGeometry(QtCore.QRect(450, 360, 160, 40))
        self.attach_next_btn.setText("NEXT →")
        self.attach_next_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #D1F35D;
                color: #000000;
                border-radius: 4px;
                font-weight: bold;
                font-family: "Inter";
                font-size: 10px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #D9FF58;
            }
            QPushButton:pressed {
                background-color: #1f9e93;
            }
            QPushButton:disabled {
                background-color: #424242;
                color: #777777;
            }
        """
        )
        self.attach_next_btn.setEnabled(False)
        self.attach_next_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # Keep your existing pak_input_field but hide it
        if hasattr(self, "pak_input_field"):
            self.pak_input_field.hide()

        # Spacer to push the title to the left
        spacerItem = QtWidgets.QSpacerItem(
            198, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem)

        # Add the first tab to the tab widget
        self.perturb_tabs.addTab(self.attach, "")

        # === TAB 2: TWEAK PARAMETERS ===
        # Second tab where users adjust perturbation parameters
        self.tweak = QtWidgets.QWidget()
        self.tweak.setObjectName("tweak")
        self.perturb_tabs.addTab(self.tweak, "")

        # Layout widget for the heading and subheading in the Attach tab
        self.layoutTweakWidget = QtWidgets.QWidget(self.tweak)
        self.layoutTweakWidget.setGeometry(QtCore.QRect(20, 20, 591, 75))
        self.layoutTweakWidget.setObjectName("layoutTweakWidget")

        # Horizontal layout containing the title section
        self.horizontalTweakLayout = QtWidgets.QHBoxLayout(self.layoutTweakWidget)
        self.horizontalTweakLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalTweakLayout.setObjectName("horizontalTweakLayout")

        # Vertical layout for tab title and subtitle
        self.tweak_title = QtWidgets.QVBoxLayout()
        self.tweak_title.setSpacing(0)  # No spacing between heading and subheading
        self.tweak_title.setObjectName("tweak_title")

        # Tab heading - "Tweak the parameters."
        self.tweak_heading = QtWidgets.QLabel(self.layoutTweakWidget)
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(28)
        font.setBold(True)
        font.setWeight(75)
        self.tweak_heading.setFont(font)
        self.tweak_heading.setScaledContents(False)
        self.tweak_heading.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop
        )
        self.tweak_heading.setWordWrap(False)
        self.tweak_heading.setIndent(0)
        self.tweak_heading.setObjectName("tweak_heading")
        self.tweak_title.addWidget(self.tweak_heading)

        # Tab subheading - description text about accepted file formats
        self.tweak_subheading = QtWidgets.QLabel(self.layoutTweakWidget)
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(10)
        font.setKerning(True)
        self.tweak_subheading.setFont(font)
        self.tweak_subheading.setObjectName("tweak_subheading")
        self.tweak_title.addWidget(self.tweak_subheading)

        # Add the title vertical layout to the horizontal layout
        self.horizontalTweakLayout.addLayout(self.tweak_title)

        # Watermark Intensity Area
        self.watermark_intensity_area = QtWidgets.QFrame(self.tweak)
        self.watermark_intensity_area.setGeometry(QtCore.QRect(20, 120, 591, 200))
        self.watermark_intensity_area.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.watermark_intensity_area.setFrameShadow(QtWidgets.QFrame.Raised)
        self.watermark_intensity_area.setStyleSheet(
            """
            QFrame {
                border: none;
                border-radius: 12px;
                background-color: #2F2F2F;
                padding: 5px;
            }
            """
        )
        self.watermark_intensity_area.setObjectName("watermark_intensity_area")

        self.watermark_vertical_layout = QtWidgets.QVBoxLayout(
            self.watermark_intensity_area
        )
        self.watermark_vertical_layout.setSpacing(0)
        self.watermark_vertical_layout.setContentsMargins(10, 10, 10, 10)

        self.intensity_title = QtWidgets.QVBoxLayout()
        self.intensity_title.setSpacing(0)  # No spacing between heading and subheading
        self.intensity_title.setContentsMargins(0, 0, 0, 0)
        self.intensity_title.setObjectName("intensity_title")

        # Move the heading and subheading into this layout
        self.intensity_heading_text = QtWidgets.QLabel(self.watermark_intensity_area)
        self.intensity_heading_text.setAlignment(
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop
        )
        self.intensity_heading_text.setText("Watermark Intensity")
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(12)
        font.setBold(True)
        self.intensity_heading_text.setFont(font)
        self.intensity_heading_text.setObjectName("intensity_heading_text")
        self.intensity_heading_text.setStyleSheet("color: #ffffff;")
        self.intensity_title.addWidget(self.intensity_heading_text)

        # Add the subheading with no spacing
        self.intensity_subheading_text = QtWidgets.QLabel(self.watermark_intensity_area)
        self.intensity_subheading_text.setAlignment(
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop
        )
        self.intensity_subheading_text.setText(
            "Select the intensity of the noise (watermark) you want applied to your image. Higher intensity settings introduce more noise but offer stronger protection against AI."
        )
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(10)
        font.setBold(False)
        self.intensity_subheading_text.setFont(font)
        self.intensity_subheading_text.setWordWrap(True)
        self.intensity_subheading_text.setStyleSheet("color: #ffffff;")
        self.intensity_subheading_text.setObjectName("intensity_subheading_text")
        self.intensity_subheading_text.setMinimumWidth(550)
        self.intensity_title.addWidget(self.intensity_subheading_text)

        # Add the title layout to the main vertical layout
        self.watermark_vertical_layout.addLayout(self.intensity_title)

        # Now add spacing before the slider
        self.watermark_vertical_layout.addSpacing(15)

        # === INTENSITY SLIDER TAB BAR ===
        # Create a container frame for the sliding tab bar with 3 options
        self.intensity_slider_container = QtWidgets.QFrame(
            self.watermark_intensity_area
        )
        self.intensity_slider_container.setMinimumHeight(49)
        self.intensity_slider_container.setMaximumHeight(49)
        self.intensity_slider_container.setStyleSheet(
            """
            QFrame {
                background-color: #232323;
                border-radius: 22px;
                border: none;
            }
        """
        )
        self.watermark_vertical_layout.addWidget(self.intensity_slider_container)
        self.watermark_vertical_layout.addSpacing(0)

        # Create horizontal layout for the tab buttons
        self.intensity_slider_layout = QtWidgets.QHBoxLayout(
            self.intensity_slider_container
        )
        self.intensity_slider_layout.setContentsMargins(0, 0, 0, 0)
        self.intensity_slider_layout.setSpacing(0)

        # Create the three tab buttons
        # Subtle button (1)
        self.subtle_tab = QtWidgets.QPushButton("Subtle")
        self.subtle_tab.setCheckable(True)
        self.subtle_tab.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # Moderate button (2)
        self.noticeable_tab = QtWidgets.QPushButton("Noticeable")
        self.noticeable_tab.setCheckable(True)
        self.noticeable_tab.setChecked(True)  # Default selected
        self.noticeable_tab.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # High button (3)
        self.obvious_tab = QtWidgets.QPushButton("Obvious")
        self.obvious_tab.setCheckable(True)
        self.obvious_tab.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # Add buttons to button group for exclusive selection
        self.intensity_button_group = QtWidgets.QButtonGroup(self)
        self.intensity_button_group.addButton(self.subtle_tab, 1)
        self.intensity_button_group.addButton(self.noticeable_tab, 2)
        self.intensity_button_group.addButton(self.obvious_tab, 3)
        self.intensity_button_group.setExclusive(True)

        # Apply common style to all buttons
        for button in [self.subtle_tab, self.noticeable_tab, self.obvious_tab]:
            button.setFixedHeight(39)
            button.setFont(QtGui.QFont("Inter", 10))

        # Add the buttons to the layout
        self.intensity_slider_layout.addWidget(self.subtle_tab)
        self.intensity_slider_layout.addWidget(self.noticeable_tab)
        self.intensity_slider_layout.addWidget(self.obvious_tab)

        # Set initial styling for all buttons
        self.update_intensity_tabs(2)  # Start with Subtle selected

        # Connect signal for button clicks
        self.intensity_button_group.buttonClicked.connect(self.on_intensity_changed)

        """ Segmentation Area """
        self.segmentation_area = QtWidgets.QFrame(self.tweak)
        self.segmentation_area.setGeometry(QtCore.QRect(20, 350, 591, 200))
        self.segmentation_area.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.segmentation_area.setFrameShadow(QtWidgets.QFrame.Raised)
        self.segmentation_area.setStyleSheet(
            """
            QFrame {
                border: none;
                border-radius: 12px;
                background-color: #2F2F2F;
                padding: 5px;
            }
            """
        )
        self.segmentation_area.setObjectName("segmentation_area")

        self.segmentation_vertical_layout = QtWidgets.QVBoxLayout(
            self.segmentation_area
        )
        self.segmentation_vertical_layout.setSpacing(0)
        self.segmentation_vertical_layout.setContentsMargins(10, 10, 10, 10)

        # Create a vertical layout for the intensity heading and subheading
        # similar to what's used for the tab title/subtitle
        self.segmentation_title = QtWidgets.QVBoxLayout()
        self.segmentation_title.setSpacing(
            0
        )  # No spacing between heading and subheading
        self.segmentation_title.setContentsMargins(0, 0, 0, 0)
        self.segmentation_title.setObjectName("segmentation_title")

        # Move the heading and subheading into this layout
        self.segmentation_heading_text = QtWidgets.QLabel(self.segmentation_area)
        self.segmentation_heading_text.setAlignment(
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop
        )
        self.segmentation_heading_text.setText("Number of Image Segments")
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(12)
        font.setBold(True)
        self.segmentation_heading_text.setFont(font)
        self.segmentation_heading_text.setObjectName("segmentation_heading_text")
        self.segmentation_heading_text.setStyleSheet("color: #ffffff;")
        self.segmentation_title.addWidget(self.segmentation_heading_text)

        # Add the subheading with no spacing
        self.segmentation_subheading_text = QtWidgets.QLabel(self.segmentation_area)
        self.segmentation_subheading_text.setAlignment(
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop
        )
        self.segmentation_subheading_text.setText(
            "Set how much segmentation you want your image to undergo. Before processing, AINS will segment your image into smaller parts in order to efficiently apply the protection to it."
        )
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(10)
        font.setBold(False)
        self.segmentation_subheading_text.setFont(font)
        self.segmentation_subheading_text.setWordWrap(True)
        self.segmentation_subheading_text.setStyleSheet("color: #ffffff;")
        self.segmentation_subheading_text.setObjectName("segmentation_subheading_text")
        self.segmentation_subheading_text.setMinimumWidth(550)
        self.segmentation_title.addWidget(self.segmentation_subheading_text)

        # Add the title layout to the main vertical layout
        self.segmentation_vertical_layout.addLayout(self.segmentation_title)

        # Now add spacing before the slider
        self.segmentation_vertical_layout.addSpacing(15)

        # === INTENSITY SLIDER TAB BAR ===
        # Create a container frame for the sliding tab bar with 3 options
        self.segmentation_slider_container = QtWidgets.QFrame(self.segmentation_area)
        self.segmentation_slider_container.setMinimumHeight(49)
        self.segmentation_slider_container.setMaximumHeight(49)
        self.segmentation_slider_container.setStyleSheet(
            """
            QFrame {
                background-color: #232323;
                border-radius: 22px;
                border: none;
            }
        """
        )
        self.segmentation_vertical_layout.addWidget(self.segmentation_slider_container)
        self.segmentation_vertical_layout.addSpacing(0)

        # Create horizontal layout for the tab buttons
        self.segmentation_slider_layout = QtWidgets.QHBoxLayout(
            self.segmentation_slider_container
        )
        self.segmentation_slider_layout.setContentsMargins(0, 0, 0, 0)
        self.segmentation_slider_layout.setSpacing(0)

        # Create the three tab buttons
        # Subtle button (1)
        self.low_tab = QtWidgets.QPushButton("Low")
        self.low_tab.setCheckable(True)
        self.low_tab.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # Moderate button (2)
        self.medium_tab = QtWidgets.QPushButton("Medium")
        self.medium_tab.setCheckable(True)
        self.medium_tab.setChecked(True)  # Default selected
        self.medium_tab.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # High button (3)
        self.high_tab = QtWidgets.QPushButton("High")
        self.high_tab.setCheckable(True)
        self.high_tab.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # Add buttons to button group for exclusive selection
        self.segmentation_button_group = QtWidgets.QButtonGroup(self)
        self.segmentation_button_group.addButton(self.low_tab, 1)
        self.segmentation_button_group.addButton(self.medium_tab, 2)
        self.segmentation_button_group.addButton(self.high_tab, 3)
        self.segmentation_button_group.setExclusive(True)

        # Apply common style to all buttons
        for button in [self.low_tab, self.medium_tab, self.high_tab]:
            button.setFixedHeight(39)
            button.setFont(QtGui.QFont("Inter", 10))

        # Add the buttons to the layout
        self.segmentation_slider_layout.addWidget(self.low_tab)
        self.segmentation_slider_layout.addWidget(self.medium_tab)
        self.segmentation_slider_layout.addWidget(self.high_tab)

        # Set initial styling for all buttons
        self.update_segmentation_tabs(2)  # Start with Subtle selected

        # Connect signal for button clicks
        self.segmentation_button_group.buttonClicked.connect(
            self.on_segmentation_changed
        )

        # "Back" button
        self.tweak_back_btn = QtWidgets.QPushButton(self.tweak)
        self.tweak_back_btn.setGeometry(QtCore.QRect(280, 580, 160, 40))
        self.tweak_back_btn.setText("BACK")
        self.tweak_back_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #FFFFFF;
                color: #000000;
                border-radius: 4px;
                font-weight: bold;
                font-family: "Inter";
                font-size: 10px;
                padding: 8px 16px;
            }
            QPushButton:pressed {
                background-color: #d7dbd8;
            }
            QPushButton:disabled {
                background-color: #424242;
                color: #777777;
            }
        """
        )
        self.tweak_back_btn.setEnabled(True)
        self.tweak_back_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # Connects signal for back button
        self.tweak_back_btn.clicked.connect(self.on_back_clicked)

        # "Next" button
        self.tweak_next_btn = QtWidgets.QPushButton(self.tweak)
        self.tweak_next_btn.setGeometry(QtCore.QRect(450, 580, 160, 40))
        self.tweak_next_btn.setText("NEXT →")
        self.tweak_next_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #D1F35D;
                color: #000000;
                border-radius: 4px;
                font-weight: bold;
                font-family: "Inter";
                font-size: 10px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #D9FF58;
            }
            QPushButton:pressed {
                background-color: #1f9e93;
            }
            QPushButton:disabled {
                background-color: #424242;
                color: #777777;
            }
        """
        )
        self.tweak_next_btn.setEnabled(True)
        self.tweak_next_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # Connect signals for next button
        self.tweak_next_btn.clicked.connect(self.on_next_clicked)

        # === TAB 3: RUN PERTURBATION ===
        # Third tab where users start the process and see results
        self.run = QtWidgets.QWidget()
        self.run.setObjectName("run")
        self.perturb_tabs.addTab(self.run, "")

        # Layout widget for the heading and subheading in the Run tab.
        self.layoutRunWidget = QtWidgets.QWidget(self.run)
        self.layoutRunWidget.setGeometry(QtCore.QRect(20, 20, 591, 110))
        self.layoutRunWidget.setObjectName("layoutRunWidget")

        # Horinzontal layout
        self.horizontalRunLayout = QtWidgets.QHBoxLayout(self.layoutRunWidget)
        self.horizontalRunLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalRunLayout.setObjectName("horizontalRunLayout")

        # Vertical layout
        self.run_title = QtWidgets.QVBoxLayout()
        self.run_title.setSpacing(0)
        self.run_title.setObjectName("run_title")

        # Tab heading
        self.run_heading = QtWidgets.QLabel(self.layoutRunWidget)
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(28)
        font.setBold(True)
        font.setWeight(75)
        self.run_heading.setFont(font)
        self.run_heading.setScaledContents(False)
        self.run_heading.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop
        )
        self.run_heading.setWordWrap(False)
        self.run_heading.setIndent(0)
        self.run_heading.setObjectName("run_heading")
        self.run_title.addWidget(self.run_heading)

        # Subheading
        self.run_subheading = QtWidgets.QLabel(self.layoutRunWidget)
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(10)
        font.setKerning(True)
        self.run_subheading.setFont(font)
        self.run_subheading.setObjectName("run_subheading")
        self.run_subheading.setWordWrap(True)
        self.run_title.addWidget(self.run_subheading)

        self.horizontalRunLayout.addLayout(self.run_title)

        # AI description area
        self.ai_description_area = QtWidgets.QFrame(self.run)
        self.ai_description_area.setGeometry(QtCore.QRect(20, 145, 591, 180))
        self.ai_description_area.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.ai_description_area.setFrameShadow(QtWidgets.QFrame.Raised)
        self.ai_description_area.setStyleSheet(
            """
            QFrame {
                border: none;
                border-radius: 12px;
                background-color: #2F2F2F;
                padding: 5px;
            }
            """
        )
        self.ai_description_area.setObjectName("ai_description_area")

        # AI description vertical layout
        self.description_vertical_layout = QtWidgets.QVBoxLayout(
            self.ai_description_area
        )
        self.description_vertical_layout.setSpacing(0)
        self.description_vertical_layout.setContentsMargins(10, 10, 10, 18)
        self.description_vertical_layout.setObjectName("description_vertical_layout")

        # Description heading
        self.ai_description_heading = QtWidgets.QLabel(self.ai_description_area)
        self.ai_description_heading.setText("Target AI Description")
        self.ai_description_heading.setAlignment(
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop
        )
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(12)
        font.setBold(True)
        self.ai_description_heading.setFont(font)
        self.ai_description_heading.setObjectName("intensity_heading_text")
        self.ai_description_heading.setStyleSheet("color: #ffffff")
        self.ai_description_heading.setMaximumHeight(35)
        self.description_vertical_layout.addWidget(self.ai_description_heading)

        # Description subheading
        self.ai_description_subheading = QtWidgets.QLabel(self.ai_description_area)
        self.ai_description_subheading.setText(
            "Enter how AI models might describe your artwork, including a unique identifier (like 'xyz_style' or 'abc_art'). This unique token helps focus protection against specific AI training. Leave blank for general protection."
        )
        self.ai_description_subheading.setAlignment(
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop
        )
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(10)
        font.setBold(False)
        self.ai_description_subheading.setFont(font)
        self.ai_description_subheading.setWordWrap(True)
        self.ai_description_subheading.setStyleSheet("color: #ffffff")
        self.ai_description_subheading.setObjectName("ai_description_subheading")
        self.ai_description_subheading.setMinimumWidth(550)
        self.description_vertical_layout.addWidget(self.ai_description_subheading)

        # Text input area
        self.ai_description_input = QtWidgets.QTextEdit(self.ai_description_area)
        self.ai_description_input.setPlaceholderText("a painting in xyz_abc style")
        self.ai_description_input.setMaximumHeight(40)
        self.ai_description_input.setStyleSheet(
            """
            QTextEdit {
                background-color: #232323;
                color: #ffffff;
                border-radius: 8px;
                border: none;
                padding: 8px;
                font-family: 'Inter';
                font-size: 12px;
                qproperty-verticalScrollBarPolicy: ScrollBarAlwaysOff;
            }
            QTextEdit:focus {
                border: 1px solid #27c9bb;
            }
        """
        )
        self.description_vertical_layout.addWidget(self.ai_description_input)

        # Output location area
        self.output_location_area = QtWidgets.QFrame(self.run)
        self.output_location_area.setGeometry(QtCore.QRect(20, 345, 591, 80))
        self.output_location_area.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.output_location_area.setFrameShadow(QtWidgets.QFrame.Raised)
        self.output_location_area.setStyleSheet(
            """
            QFrame {
                border: none;
                border-radius: 12px;
                background-color: #2F2F2F;
                padding: 5px;
            }
            """
        )
        self.output_location_area.setObjectName("output_location_area")

        # Set output location horizontal layout
        self.output_location_horizontal_layout = QtWidgets.QHBoxLayout(
            self.output_location_area
        )
        self.output_location_horizontal_layout.setSpacing(5)
        self.output_location_horizontal_layout.setContentsMargins(10, 10, 10, 10)
        self.output_location_horizontal_layout.setObjectName(
            "output_location_horizontal_layout"
        )

        # Output location button for directory selection
        self.browse_dir_button = QtWidgets.QPushButton("📁", self.output_location_area)
        # self.browse_dir_button.setGeometry(QtCore.QRect(10, 70, 571, 40))
        self.browse_dir_button.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,  # Horizontal policy - use preferred width
            QtWidgets.QSizePolicy.Fixed,  # Vertical policy - fixed height
        )
        self.browse_dir_button.setMaximumWidth(38)
        self.browse_dir_button.setStyleSheet(
            """
            QPushButton {
                background-color: #232323;
                color: #ffffff;
                border-radius: 8px;
                border: none;
                padding: 8px;
                font-family: 'Inter';
                font-size: 16px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #353535;
                border: 1px solid #27c9bb;
            }
            QPushButton:pressed {
                background-color: #1a1a1a;
            }
        """
        )
        self.browse_dir_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.output_location_horizontal_layout.addWidget(self.browse_dir_button)

        # Create vertical layout for output location text
        self.browse_vertical_layout = QtWidgets.QVBoxLayout()
        self.browse_vertical_layout.setSpacing(0)

        # Add heading label
        self.output_location_heading = QtWidgets.QLabel("Set output location")
        self.output_location_heading.setStyleSheet("color: #FFFFFF; font-weight: bold;")
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(12)
        font.setBold(True)
        self.output_location_heading.setFont(font)
        self.browse_vertical_layout.addWidget(self.output_location_heading)

        # Add path subheading that will show the selected path
        self.output_location_path = QtWidgets.QLabel("Not set")
        self.output_location_path.setStyleSheet("color: #5E5D5D;")
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(9)
        self.output_location_path.setFont(font)
        # Allow the label to show ellipsis when text is too long
        self.output_location_path.setMinimumWidth(200)
        self.output_location_path.setMaximumWidth(450)
        self.output_location_path.setWordWrap(False)
        self.output_location_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.browse_vertical_layout.addWidget(self.output_location_path)
        # Make path clickable
        self.output_location_path.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.output_location_path.setStyleSheet("color: #5E5D5D;")

        # Add the vertical layout to the horizontal layout (after the button)
        self.output_location_horizontal_layout.addLayout(self.browse_vertical_layout)

        # Add a spacer at the end to push everything to the left
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.output_location_horizontal_layout.addItem(spacerItem)
        # event listener
        self.output_location_path.mousePressEvent = self.open_output_directory
        self.browse_dir_button.clicked.connect(self.select_output_directory)

        # event filter to handle hover events
        self.output_location_path.installEventFilter(self)

        # Run button
        self.run_button = QtWidgets.QPushButton("RUN", self.run)
        self.run_button.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        self.run_button.setStyleSheet(
            """
            QPushButton {
                background-color: #D1F35D;
                color: black;
                border-radius: 4px;
                border: none;
                padding: 8px;
                font-family: 'Inter';
                font-size: 10px;
                font-weight: bold;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #D9FF58;
                border: 1px solid #27c9bb;
            }
            QPushButton:pressed {
                background-color: #A7C848;
            }
            QPushButton:disabled {
                background-color: #424242;
                color: #777777;
            }
        """
        )
        self.run_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.run_button.setGeometry(
            QtCore.QRect(20, 445, 591, 35)
        )  # Positioned below the output location area
        self.run_button.setEnabled(False)
        self.run_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.run_button.clicked.connect(self.run_perturbation)

        # back button
        self.run_back_btn = QtWidgets.QPushButton("BACK", self.run)
        self.run_back_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        self.run_back_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #FFFFFF;
                color: black;
                border-radius: 4px;
                border: none;
                padding: 8px;
                font-family: 'Inter';
                font-size: 10px;
                font-weight: bold;
                text-align: center;
            }
            QPushButton:pressed {
                background-color: #d7dbd8;
            }
            QPushButton:disabled {
                background-color: #424242;
                color: #777777;
            }
        """
        )
        self.run_back_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.run_back_btn.setGeometry(QtCore.QRect(20, 490, 591, 35))
        self.run_back_btn.setEnabled(True)
        self.run_back_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.run_back_btn.clicked.connect(self.on_back_clicked)

        # === APPLICATION LOGO ===
        # Logo displayed in the top-left corner
        self.logo = QtWidgets.QLabel(self.centralwidget)
        self.logo.setGeometry(QtCore.QRect(35, 20, 60, 60))
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.logo.sizePolicy().hasHeightForWidth())
        self.logo.setSizePolicy(sizePolicy)
        self.logo.setText("")
        logo_path = self.resource_path("resources/gui/logo.png")
        self.logo.setPixmap(QtGui.QPixmap(logo_path))
        self.logo.setScaledContents(True)
        self.logo.setObjectName("logo")

        # === IMAGE PREVIEW AREA ===
        # Right side area where uploaded/processed images are displayed
        self.display_attached_image = QtWidgets.QLabel(self.centralwidget)
        self.display_attached_image.setGeometry(QtCore.QRect(680, 102, 485, 470))
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(28)
        font.setBold(False)
        font.setWeight(50)
        self.display_attached_image.setFont(font)
        self.display_attached_image.setScaledContents(False)
        self.display_attached_image.setAlignment(QtCore.Qt.AlignCenter)
        self.display_attached_image.setWordWrap(False)
        self.display_attached_image.setIndent(0)
        self.display_attached_image.setObjectName("display_attached_image")
        self.display_attached_image.setStyleSheet(
            """ 
            QLabel {
            background-color: #2f2f2f; 
            border-radius: 12px;
            color: #ffffff;
            }
            """
        )

        # Files list widget
        self.files_list = QtWidgets.QListWidget(self.centralwidget)
        self.files_list.setGeometry(QtCore.QRect(680, 600, 485, 150))
        self.files_list.setStyleSheet(
            """
            QListWidget {
                background-color: transparent; 
                border : 2px solid #3a3a3d;
                border-radius: 12px;
                color: #ffffff;
                qproperty-verticalScrollBarPolicy: ScrollBarAlwaysOff;
            }
            QListWidget::item {
                padding: 6px;
                border-bottom: 1px solid #3a3a3d;
            }
            QListWidget::item:selected {
                background-color: #27c9bb;
                color: #000000;
            }
        """
        )
        self.files_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        # Set the central widget of the main window
        MainWindow.setCentralWidget(self.centralwidget)

        # === MENU BAR ===
        # Top menu bar (currently empty)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1205, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        # === STATUS BAR ===
        # Bottom status bar for displaying messages
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # === SET UP UI TEXT AND CONNECTIONS ===
        # Set all text elements and connect signals
        self.retranslateUi(MainWindow)
        self.perturb_tabs.setCurrentIndex(0)  # Start on the first tab

        # Connect signals
        self.image_drop_area.mousePressEvent = self.browse_for_images
        self.files_list.itemSelectionChanged.connect(self.on_file_selection_changed)
        self.attach_next_btn.clicked.connect(self.on_next_clicked)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.installEventFilter(self)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AI-No-Swiping"))
        self.ains_heading.setText(
            _translate(
                "MainWindow",
                '<html><head/><body><p><span style=" font-size:16pt; color:#27c9bb;">AI-No-Swiping</span></p></body></html>',
            )
        )
        self.ains_subheading.setText(
            _translate(
                "MainWindow",
                '<html><head/><body><p><span style=" font-size:10pt; color:#ffffff;">Protect your art from AI misuse!</span></p></body></html>',
            )
        )
        self.attach_heading.setText(
            _translate(
                "MainWindow",
                '<html><head/><body><p><span style=" color:#d1f35d;">Upload your art.</span></p></body></html>',
            )
        )
        self.attach_subheading.setText(
            _translate(
                "MainWindow",
                '<html><head/><body><p><span style=" color:#5e5d5d;">Attach the artwork that you want to protect. It must be in .png, or .jpeg format.</span></p></body></html>',
            )
        )
        self.tweak_heading.setText(
            _translate(
                "MainWindow",
                '<html><head/><body><p><span style=" color:#d1f35d;">Tweak protection.</span></p></body></html>',
            )
        )
        self.tweak_subheading.setText(
            _translate(
                "MainWindow",
                '<html><head/><body><p><span style=" color:#5e5d5d;">Adjust the protection settings for your image.</span></p></body></html>',
            )
        )
        self.run_heading.setText(
            _translate(
                "MainWindow",
                '<html><head/><body><p><span style=" color:#d1f35d;">Run AINS.</span></p></body></html>',
            )
        )
        self.run_subheading.setText(
            _translate(
                "MainWindow",
                '<html><head/><body><p><span style=" color:#5e5d5d;">Enter an AI description and set the output location of your image. Click the “run” to apply protection to your image. Once done, you can click the “save output” button below the image preview on the right to download your successfully protected image.</span></p></body></html>',
            )
        )
        # Change tab names to be more descriptive
        self.perturb_tabs.setTabText(
            self.perturb_tabs.indexOf(self.attach),
            _translate("MainWindow", "1"),
        )
        self.perturb_tabs.setTabText(
            self.perturb_tabs.indexOf(self.tweak),
            _translate("MainWindow", "2"),
        )
        self.perturb_tabs.setTabText(
            self.perturb_tabs.indexOf(self.run), _translate("MainWindow", "3")
        )
        self.display_attached_image.setText(
            _translate(
                "MainWindow",
                '<html><head/><body><p><span style=" font-size:10pt; color:#5e5d5d;">Your image will be displayed here.</span></p></body></html>',
            )
        )

    def resource_path(self, relative_path):
        """Get absolute path to resource, works for dev and for PyInstaller"""
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def load_fonts(self):
        """Load custom fonts into the Qt application"""
        print("Loading custom fonts...")
        # Create a database of fonts
        font_db = QFontDatabase()

        # Define all font paths we want to load
        font_paths = [
            # Inter fonts
            "resources/gui/Inter/static/Inter_18pt-Regular.ttf",
            "resources/gui/Inter/static/Inter_18pt-Bold.ttf",
            "resources/gui/Inter/static/Inter_18pt-Medium.ttf",
            # Georama fonts - including Black which is used in the heading
            "resources/gui/Georama/static/Georama-Regular.ttf",
            "resources/gui/Georama/static/Georama-Bold.ttf",
            "resources/gui/Georama/static/Georama-Black.ttf",  # Important for main heading
        ]

        # Add fonts to database
        font_ids = []
        loaded_families = []

        for relative_path in font_paths:
            font_path = self.resource_path(relative_path)
            if os.path.exists(font_path):
                font_id = font_db.addApplicationFont(font_path)
                if font_id != -1:
                    families = font_db.applicationFontFamilies(font_id)
                    if families:
                        loaded_families.extend(families)
                        print(
                            f"Successfully loaded: {os.path.basename(font_path)} with family: {families[0]}"
                        )
                    font_ids.append(font_id)
                else:
                    print(f"Failed to load font: {font_path}")
            else:
                print(f"Font file not found: {font_path}")

        # Print available font families for debugging
        if loaded_families:
            print("Loaded font families:", list(set(loaded_families)))
        else:
            print("WARNING: No fonts were loaded successfully.")

        return font_ids

    def eventFilter(self, source, event):
        """Handle drag-and-drop events for the image drop area."""
        if source == self.image_drop_area:
            if event.type() == QEvent.DragEnter:
                if event.mimeData().hasUrls():
                    event.accept()
                else:
                    event.ignore()
                return True
            elif event.type() == QEvent.Drop:
                if event.mimeData().hasUrls():
                    self.handle_dropped_files(event.mimeData().urls())
                    event.accept()
                else:
                    event.ignore()
                return True

        # Check if the attribute exists first
        if (
            hasattr(self, "output_location_path")
            and source == self.output_location_path
        ):
            if event.type() == QEvent.Enter:
                # Mouse enter? show underline
                self.output_location_path.setStyleSheet(
                    "color: #5E5D5D; text-decoration: underline; text-decoration-color: white;"
                )
                return True
            elif event.type() == QEvent.Leave:
                # Mouse exits
                self.output_location_path.setStyleSheet(
                    "color: #5E5D5D; text-decoration: none;"
                )
                return True

        if event.type() == QEvent.MouseButtonPress:
            if self.ai_description_input.hasFocus():
                pos = event.globalPos()
                local_pos = self.ai_description_input.mapFromGlobal(pos)
                if not self.ai_description_input.rect().contains(local_pos):
                    new_description = self.ai_description_input.toPlainText().strip()
                    self.ai_description = new_description
                    self.ai_description_input.clearFocus()

                    if new_description:
                        print(f"AI description set to: {self.ai_description}")

                    return True

        return super().eventFilter(source, event)

    def handle_dropped_files(self, urls):
        """Process dropped files and update the UI."""
        for url in urls:
            file_path = url.toLocalFile()
            if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                if file_path not in self.selected_image_paths:
                    self.selected_image_paths.append(file_path)
                    self.files_list.addItem(os.path.basename(file_path))

        if self.selected_image_paths:
            self.update_selected_files_label()
            self.display_image(self.selected_image_paths[0])
            self.attach_next_btn.setEnabled(True)
            self.attach_reset_btn.setEnabled(True)

    def reset_attached_images(self):
        """Clear all selected images and reset the UI."""
        # Clear selected image paths
        self.selected_image_paths = []
        self.perturbed_image_paths = []
        self.showing_perturbed_images = False

        # Clear the files list
        self.files_list.clear()

        # Update the label
        self.update_selected_files_label()

        # Reset image preview to default message
        self.display_attached_image.setText(
            '<html><head/><body><p><span style=" font-size:10pt; color:#5e5d5d;">Your image will be displayed here.</span></p></body></html>'
        )
        self.display_attached_image.setPixmap(
            QtGui.QPixmap()
        )  # Clear any existing pixmap

        # Disable next and reset buttons
        self.attach_next_btn.setEnabled(False)
        self.attach_reset_btn.setEnabled(False)

        # Log action
        print("All attached images have been reset.")

        # Show confirmation message
        self.show_alert("All attached images have been cleared.", "confirmation")

    def display_image(self, image_path):
        """Display the selected image in the preview area."""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # Scale the pixmap to fit the label while preserving aspect ratio
            pixmap = pixmap.scaled(
                self.display_attached_image.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.display_attached_image.setPixmap(pixmap)
        else:
            self.display_attached_image.setText("Failed to load image")

    def update_selected_files_label(self):
        """Update the label showing the number of selected files."""
        count = len(self.selected_image_paths)
        if count == 0:
            self.selected_files_label.setText("Selected Files: None")
        elif count == 1:
            self.selected_files_label.setText("Selected Files: 1 image")
        else:
            self.selected_files_label.setText(f"Selected Files: {count} images")

    def browse_for_images(self, event):
        """Open a file dialog to browse for images."""
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(
            self.main_window,
            "Select Image Files",
            "",
            "Image Files (*.png *.jpg *.jpeg)",
        )

        if file_paths:
            for file_path in file_paths:
                if file_path not in self.selected_image_paths:
                    self.selected_image_paths.append(file_path)
                    self.files_list.addItem(os.path.basename(file_path))

            self.update_selected_files_label()
            self.display_image(self.selected_image_paths[0])
            self.attach_next_btn.setEnabled(True)
            self.attach_reset_btn.setEnabled(True)

        # Reset perturbed images state when new images are added
        self.showing_perturbed_images = False
        self.perturbed_image_paths = []

    def on_file_selection_changed(self):
        """Update the preview when the user selects a file from the list."""
        selected_items = self.files_list.selectedItems()

        if not selected_items:
            return

        selected_index = self.files_list.row(selected_items[0])

        # Determine which list to use based on current display state
        if getattr(self, "showing_perturbed_images", False):
            # We're showing perturbed images
            if 0 <= selected_index < len(self.perturbed_image_paths):
                self.display_image(self.perturbed_image_paths[selected_index])
        else:
            # We're showing original images
            if 0 <= selected_index < len(self.selected_image_paths):
                self.display_image(self.selected_image_paths[selected_index])

    def on_next_clicked(self):
        """Move to the next tab (+1)"""
        # Move to the next tab
        # Get current index of the tab widget
        current_index = self.perturb_tabs.currentIndex()
        self.perturb_tabs.setCurrentIndex(current_index + 1)

    def on_back_clicked(self):
        """Moves back the current tab index by 1"""
        current_index = self.perturb_tabs.currentIndex()
        self.perturb_tabs.setCurrentIndex(current_index - 1)

    def update_intensity_tabs(self, selected_intensity):
        """Update the styling of the intensity tabs based on the selected intensity."""
        for button in [self.subtle_tab, self.noticeable_tab, self.obvious_tab]:
            if self.intensity_button_group.id(button) == selected_intensity:
                button.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #27c9bb;
                        color: #ffffff;
                        border-radius: 18px;
                        font-weight: bold;
                    }
                """
                )
            else:
                button.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #232323;
                        color: #999999;
                        border-radius: 18px;
                        font-weight: normal;
                    }
                """
                )

    def on_intensity_changed(self, button):
        """Handle intensity tab selection changes."""
        self.intensity = self.intensity_button_group.id(button)
        self.update_intensity_tabs(self.intensity)
        print(f"Intensity changed to: {self.intensity}")

    def update_segmentation_tabs(self, selected_segmentation):
        """Update the styling of the segmentation tabs based on the selected segmentation."""
        for button in [self.low_tab, self.medium_tab, self.high_tab]:
            if self.segmentation_button_group.id(button) == selected_segmentation:
                button.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #27c9bb;
                        color: #ffffff;
                        border-radius: 18px;
                        font-weight: bold;
                    }
                """
                )
            else:
                button.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #232323;
                        color: #999999;
                        border-radius: 18px;
                        font-weight: normal;
                    }
                """
                )

    def on_segmentation_changed(self, button):
        """Handle segmentation tab selection changes."""
        self.segmentation = self.segmentation_button_group.id(button)
        self.update_segmentation_tabs(self.segmentation)
        print(f"Segmentation changed to: {self.segmentation}")

    def select_output_directory(self):
        """Open a directory selection dialog"""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self.main_window,
            "Select Output Directory",
            "",
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )

        if directory:
            self.output_path = directory
            # update path label with selected label
            #  if the path is too long, show only rhe end
            display_path = directory
            if len(display_path) > 40:
                display_path = "..." + display_path[-37:]

            self.output_location_path.setText(display_path)
            self.output_location_path.setStyleSheet(
                "color: #5E5D5D;"
            )  # Change color when set
            # Enable the Run button
            self.run_button.setEnabled(True)
            print(f"Output directory set to: {directory}")

            print("The global path selected is: ", self.output_path)

    def show_alert(self, message, messageType):
        """Creates an alert widget at the bottom for pop up alerts/messages"""
        # Create the alert widget
        self.alert_widget = QtWidgets.QFrame(self.centralwidget)

        # Calculate center position (adjust width based on message length)
        message_width = min(
            max(len(message) * 7, 300), 500
        )  # Estimate width based on message length
        x_position = (self.centralwidget.width() - message_width) // 2

        # Position it at the bottom fo the window, initially hidden
        self.alert_widget.setGeometry(QtCore.QRect(x_position, 708, message_width, 40))

        # set z-order for the alert message to appear on top of everything else
        self.alert_widget.raise_()

        # Different style sheets depending on the type of messaage
        if messageType == "confirmation":
            self.alert_widget.setStyleSheet(
                """
                QFrame {
                    background-color: #27c9bb;
                    border-radius: 17px;
                }
            """
            )
        elif messageType == "error":
            self.alert_widget.setStyleSheet(
                """
                QFrame {
                    background-color: #E54848;
                    border-radius: 17px;
                }
            """
            )

        # Horizontal layout for the alert content
        alert_layout = QtWidgets.QHBoxLayout(self.alert_widget)
        alert_layout.setContentsMargins(15, 5, 15, 5)

        # Alert message
        self.alert_message = QtWidgets.QLabel(message)
        self.alert_message.setStyleSheet(
            """ 
            background-color: transparent;
            color: #ffffff;
            font-family: 'Inter';
            font-weight: bold;
            font-size: 12px;
            """
        )
        self.alert_message.setAlignment(QtCore.Qt.AlignCenter)
        alert_layout.addWidget(self.alert_message)

        # enable visibility
        self.alert_widget.setVisible(True)

        # Auto-dismiss after 4 seconds
        QtCore.QTimer.singleShot(6000, self.hide_alert)

        # Make the alert clickable to dismiss it
        self.alert_widget.mousePressEvent = lambda e: self.hide_alert()

    def hide_alert(self):
        """Function to hide the alert message"""
        self.alert_widget.setVisible(False)

    def update_progress_screen(self, message):
        """Function for updating the progress screen"""
        # Add message to list
        self.progress_lines.append(message)

        # Clear current list
        self.files_list.clear()

        self.files_list.setStyleSheet(
            """
            QListWidget {
                background-color: transparent; 
                border : 2px solid #3a3a3d;
                border-radius: 12px;
                color: #ffffff;
                qproperty-verticalScrollBarPolicy: ScrollBarAlwaysOff;
                padding: 5px;
            }
            QListWidget::item {
                padding: 0px;
                border-bottom: none;
            }
        """
        )

        # Only show last 8 lines to avoid overwhelming the display
        visible_lines = (
            self.progress_lines[-32:]
            if len(self.progress_lines) > 32
            else self.progress_lines
        )

        # Add each line to the list widget
        for line in visible_lines:
            self.files_list.addItem(line)

        # Scroll to the bottom
        self.files_list.scrollToBottom()

        # Process events to update UI immediately
        QtWidgets.QApplication.processEvents()

    def run_perturbation(self):
        # Check if the directory exists
        if not os.path.isdir(self.output_path):
            print(f"Error: Directory not found - {self.output_path}")
            self.show_alert("Invalid file path chosen.", "error")
            return

        # Check if we have any images to process
        if not self.selected_image_paths:
            print("Error: No images to be perturbed!")
            self.show_alert(
                "Please select at least one image from the Attach Image tab.", "error"
            )
            return

        # Clear previous perturbed images
        self.perturbed_image_paths = []

        # Clear previous progress lines
        self.progress_lines = []
        self.files_list.clear()
        self.update_progress_screen("Starting processing...")

        # Use a generic prompt if user did not provide any
        prompt = (
            self.ai_description if self.ai_description else "an image in xyz_abc style"
        )
        (
            self.show_alert(
                f"Processing {len(self.selected_image_paths)} images", "confirmation"
            )
            if (len(self.selected_image_paths) > 1)
            else self.show_alert(
                f"Processing {len(self.selected_image_paths)} image", "confirmation"
            )
        )

        # Disable run button while processing to avoid blowing up your computer
        self.run_button.setEnabled(False)
        self.run_back_btn.setEnabled(False)

        # Create and start the processing thread
        self.processing_thread = ProcessingThread(
            self.selected_image_paths,
            self.output_path,
            self.intensity,
            self.segmentation,
            prompt,
        )

        # Connect signals
        self.processing_thread.progress.connect(self.update_progress_screen)
        self.processing_thread.image_processed.connect(self.add_perturbed_image)
        self.processing_thread.finished.connect(self.on_processing_finished)

        # Start the thread
        self.processing_thread.start()

        # # Process each selected image
        # for image_path in self.selected_image_paths:
        #     try:
        #         # Extract filename for display
        #         filename = os.path.basename(image_path)
        #         print(f"\n--- Processing image: {filename} ---")

        #         # Load the image
        #         image = Image.open(image_path)

        #         # Get tiling dimensions based on segmentation level (1=low, 2=medium, 3=high)
        #         rows, cols = determine_tiling_dimensions(image, self.segmentation)
        #         tile_width, tile_height = compute_tile_size(image, rows, cols)

        #         # Ensure overlap_size is an integer
        #         overlap_size = int(tile_height * 0.1)

        #         print(
        #             f"  Tiling with {rows} rows, {cols} columns. Overlap: {overlap_size}px"
        #         )

        #         # Create TileInator instance with the selected parameters
        #         input_image_processor = TileInator(
        #             overlap_size=overlap_size,
        #             image=image,
        #             tile_width=tile_width,
        #             tile_height=tile_height,
        #             num_cols=cols,
        #             num_rows=rows,
        #             intensity=self.intensity,  # Pass the intensity level (1, 2, or 3)
        #             prompt=prompt,  # Pass the AI description
        #             filename=filename,
        #             progress_callback=self.update_progress_screen,  # pass the function to tileinator
        #         )

        #         # Process the image
        #         input_image_processor.process_image()
        #         self.update_progress_screen(f"✓ Finished processing {filename}")
        #         print(f"  Finished processing {filename}.")

        #         # Calculate output path
        #         # output_filename = f"{filename}"
        #         # output_path = os.path.join(self.output_path, output_filename)

        #         # Update UI and show success message
        #         # self.statusbar.showMessage(f"Successfully processed: {filename}", 3000)

        #         # You might want to display the processed image if it's the last one
        #         # self.display_image(output_path)

        #     except Exception as e:
        #         error_msg = f"Error processing {os.path.basename(image_path)}: {str(e)}"
        #         self.update_progress_screen(f"❌ {error_msg}")
        #         print(f"Error processing image {image_path}: {e}")
        #         self.show_alert(
        #             f"Error processing {os.path.basename(image_path)}", "error"
        #         )

    def on_processing_finished(self, success, message):
        # Re-enable the run button
        self.run_button.setEnabled(True)
        self.run_back_btn.setEnabled(True)

        # Show completion message
        if success:
            self.show_alert(message, "confirmation")
            if self.perturbed_image_paths:
                self.switch_to_perturbed_images()
        else:
            self.show_alert(message, "error")

        # # Show completion message
        # self.update_progress_screen(
        #     f"Successfully processed all {len(self.selected_image_paths)} image(s)."
        # )
        # self.show_alert(
        #     f"Successfully processed {len(self.selected_image_paths)} image(s)!",
        #     "confirmation",
        # )
        # print("\n--- Finished processing all images ---")

    def open_output_directory(self, event):
        """Open output directory in the file manager"""

        if hasattr(self, "output_path") and self.output_path:

            # check if it exists
            if os.path.isdir(self.output_path):
                os.startfile(self.output_path)  # only works on Windows at this moment
            else:
                # Directory no longer exists or some other error
                self.show_alert("The specified directory doesn't exist.", "error")

    def add_perturbed_image(self, image_path):
        """Add a perturbed image path to list and update the UI"""
        self.perturbed_image_paths.append(image_path)

    def switch_to_perturbed_images(self):
        """Switch UI files list back to normal files list"""
        # Clear the files list
        self.files_list.clear()
        # Bring back old stylesheet
        self.files_list.setStyleSheet(
            """
            QListWidget {
                background-color: transparent; 
                border : 2px solid #3a3a3d;
                border-radius: 12px;
                color: #ffffff;
                qproperty-verticalScrollBarPolicy: ScrollBarAlwaysOff;
            }
            QListWidget::item {
                padding: 6px;
                border-bottom: 1px solid #3a3a3d;
            }
            QListWidget::item:selected {
                background-color: #27c9bb;
                color: #000000;
            }
        """
        )

        # add perturbed image filenames to the list
        for image_path in self.perturbed_image_paths:
            self.files_list.addItem(os.path.basename(image_path))

        # update label
        self.update_selected_files_label()

        # display the first image in the list
        if self.perturbed_image_paths:
            self.display_image(self.perturbed_image_paths[0])

        # update the file seledction handler to handle perturbed images
        self.showing_perturbed_images = True


if __name__ == "__main__":
    # Initialize application
    app = QtWidgets.QApplication(sys.argv)

    # Create and show splash screen
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    # Create loader thread
    loader = ModelLoader()

    def on_models_loaded():
        # Create main window when models are loaded
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.load_fonts()
        ui.setupUi(MainWindow)

        # Show main window and close splash
        MainWindow.show()
        splash.finish(MainWindow)

    # Connect signals
    loader.status_update.connect(splash.update_status)
    loader.finished.connect(on_models_loaded)

    # Start loading models
    loader.start()

    sys.exit(app.exec_())
