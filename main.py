import os
import numpy as np
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm  # Import tqdm for the progress bar
from PyQt5.QtWidgets import QApplication,QListWidgetItem,QAction, QWidget, \
    QVBoxLayout, QPushButton, QFileDialog, QLabel,QProgressBar,QListWidget, \
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,QMainWindow
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from qt_material import apply_stylesheet

image_extensions = ['.jpeg', '.png', '.jpg', '.bmp']
def get_image_list(dir):   
    return [file for file in os.listdir(dir) if os.path.splitext(file)[1].lower() in image_extensions]

# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove the last fully connected layer
model.eval()  # Set the model to evaluation mode
model.to('cuda')

# Preprocess input images
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((512, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Calculate image features
def get_image_features(image_path):
    input_image = preprocess_image(image_path)
    with torch.no_grad():
        features = model(input_image.to('cuda')).cpu()
    return features.squeeze().numpy()

# Function to calculate cosine similarity
def cosine_similarity(feature1, feature2):
    dot_product = np.dot(feature1, feature2)
    norm1 = np.linalg.norm(feature1)
    norm2 = np.linalg.norm(feature2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

class SimilarityCalculator(QThread):
    update_progress = pyqtSignal(int, str)  # Signal to update progress bar
    update_list = pyqtSignal(list)  # Signal to update list widget
    def __init__(self, reference_path, search_folder_path):
        super().__init__()
        self.reference_path = reference_path
        self.search_folder_path = search_folder_path
    def run(self):
        # Path to the folder containing all images
        image_folder_path = self.search_folder_path

        # Path to the reference image
        reference_image_path = self.reference_path

        # Get features for the reference image
        reference_features = get_image_features(reference_image_path)

        # Calculate similarity for all images in the folder
        image_similarity_list = []
        count = 0
        images = get_image_list(image_folder_path)
        total  = len(images)
        for image_name in tqdm(images,desc="Calculating Similarity"):
            image_path = os.path.join(image_folder_path, image_name)
            image_features = get_image_features(image_path)
            similarity_score = cosine_similarity(reference_features, image_features)
            image_similarity_list.append((image_path, similarity_score))
            self.update_progress.emit(count*100/total, f"Calculating {count}/{total}")
            count = count+1
        sorted_images = sorted(image_similarity_list, key=lambda x: x[1], reverse=True)
        # Sort images based on similarity in descending order
        # # Print the sorted list
        # for idx, (image_path, similarity_score) in enumerate(sorted_images):
        #     print(f"Rank {idx + 1}: {image_path}, Similarity: {similarity_score}")
        self.update_progress.emit(100, "Calculation complete.")
        self.update_list.emit(sorted_images)
class ImageWindow(QMainWindow):
    def __init__(self, image_path):
        super().__init__()

        self.image_path = image_path

        self.scene = QGraphicsScene()
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.view = QGraphicsView(self.scene)
        self.setCentralWidget(self.view)

        self.create_toolbar()
        self.load_image(image_path)

        self.setWindowTitle('Image Viewer')
        self.setGeometry(300, 300, 800, 600)
    def load_image(self,image_path):
        pixmap = QPixmap(image_path)
        self.pixmap_item.setPixmap(pixmap)

    def change_image(self,image_path):
        self.load_image(image_path)

    def create_toolbar(self):
        zoom_in_action = QAction('Zoom In', self)
        zoom_out_action = QAction('Zoom Out', self)
        fit_to_window_action = QAction('Fit to Window', self)

        zoom_in_action.triggered.connect(self.zoom_in)
        zoom_out_action.triggered.connect(self.zoom_out)
        fit_to_window_action.triggered.connect(self.fit_to_window)

        toolbar = self.addToolBar('Image Toolbar')
        toolbar.addAction(zoom_in_action)
        toolbar.addAction(zoom_out_action)
        toolbar.addAction(fit_to_window_action)


    def zoom_in(self):
        self.view.scale(1.2, 1.2)

    def zoom_out(self):
        self.view.scale(1 / 1.2, 1 / 1.2)

    def fit_to_window(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
class ImageSimilarityApp(QWidget):
    def __init__(self):
        super().__init__()

        self.reference_image_path = ""
        self.search_folder_path = ""

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.reference_label = QLabel("Reference Image: Not selected")
        self.search_label = QLabel("Search Folder: Not selected")

        reference_button = QPushButton("Select Reference Image", self)
        reference_button.clicked.connect(self.show_reference_dialog)

        search_button = QPushButton("Select Search Folder", self)
        search_button.clicked.connect(self.show_search_folder_dialog)

        self.progress_bar = QProgressBar(self)
        self.progress_label = QLabel("Progress: 0%")

        calculate_button = QPushButton("Calculate Similarity", self)
        calculate_button.clicked.connect(self.calculate_similarity)

        self.list_widget = QListWidget()
        self.list_widget.itemSelectionChanged.connect(self.display_selected_image)

        layout.addWidget(self.reference_label)
        layout.addWidget(reference_button)
        layout.addWidget(self.search_label)
        layout.addWidget(search_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        layout.addWidget(calculate_button)
        layout.addWidget(self.list_widget)

        self.setLayout(layout)

        self.setWindowTitle('Image Similarity App')
        self.setGeometry(300, 300, 500, 400)
        self.image_window = None
    def display_selected_image(self):
        item = self.list_widget.selectedItems()[0]
        image_path = item.data(Qt.UserRole)
        if image_path:
            self.display_image(image_path)
    def display_image(self, image_path):
        if(not self.image_window):
            self.image_window = ImageWindow(image_path)
            self.image_window.show()
        else:
            if (self.image_window.isVisible()):
                self.image_window.change_image(image_path)
            else:
                self.image_window = ImageWindow(image_path)
                self.image_window.show()
    def show_reference_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setViewMode(QFileDialog.Detail)
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.reference_image_path = file_paths[0]
                self.reference_label.setText(f"Reference Image: {self.reference_image_path}")

    def show_search_folder_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folder_dialog = QFileDialog()
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setViewMode(QFileDialog.Detail)
        if folder_dialog.exec_():
            folder_paths = folder_dialog.selectedFiles()
            if folder_paths:
                self.search_folder_path = folder_paths[0]
                self.search_label.setText(f"Search Folder: {self.search_folder_path}")

    def calculate_similarity(self):
        if not self.reference_image_path or not self.search_folder_path:
            print("Please select both a reference image and a search folder.")
            return
        self.progress_bar.setValue(0)
        self.progress_label.setText("Progress: 0%")
        # Create and start a worker thread for similarity calculation
        self.similarity_calculator = SimilarityCalculator(self.reference_image_path, self.search_folder_path)
        self.similarity_calculator.update_progress.connect(self.update_progress)
        self.similarity_calculator.update_list.connect(self.update_list)
        self.similarity_calculator.start()
    def update_progress(self, value, text):
        self.progress_bar.setValue(value)
        self.progress_label.setText(f"Progress: {value}% - {text}")

    def update_list(self, sorted_images):
        self.list_widget.clear()
        for idx, (image_path, similarity_score) in enumerate(sorted_images):
            item_text = f"Rank {idx + 1}: {image_path}, Similarity: {similarity_score:.4f}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, image_path)  # Store image path as item data
            self.list_widget.addItem(item)
if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='light_blue.xml')
    ex = ImageSimilarityApp()
    ex.show()

    # window = MainWindow(dbconn)
    app.exec_()
    sys.exit(0)