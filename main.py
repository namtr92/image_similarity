import os
import numpy as np
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms,datasets
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from tqdm import tqdm  # Import tqdm for the progress bar
from PyQt5.QtWidgets import QApplication,QListWidgetItem,QAction, QWidget, \
    QVBoxLayout, QPushButton, QFileDialog, QLabel,QProgressBar,QListWidget, \
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,QMainWindow
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from qt_material import apply_stylesheet
import onnxruntime as ort
import cv2

image_extensions = ['.jpeg', '.png', '.jpg', '.bmp']
def get_image_list(dir):   
    return [file for file in os.listdir(dir) if os.path.splitext(file)[1].lower() in image_extensions]
preprocess = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, cache_data = {},transform=None):
        self.root_dir = root_dir
        self.cache_data = cache_data
        self.transform = transform
        self.image_list = get_image_list(root_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.image_list[idx] not in self.cache_data:
            img_name = os.path.join(self.root_dir, self.image_list[idx])
            image = Image.open(img_name).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return {'image_process': image, 'name': self.image_list[idx],'type':0}
        else:
            return {'image_process': self.cache_data[self.image_list[idx]], 'name': self.image_list[idx],'type':1}
# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove the last fully connected layer
model.eval()  # Set the model to evaluation mode
model.to('cuda')

providers = ['CUDAExecutionProvider','CPUExecutionProvider']
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
detection_model_path = 'model/model.onnx'
# if not os.path.exists(detection_model_path):
ONNXSession = ort.InferenceSession(detection_model_path,options=options,providers=providers)
input_name = ONNXSession.get_inputs()[0].name
output_name = ONNXSession.get_outputs()[0].name
# Preprocess input images
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Calculate image features
def get_image_features(image_path):
    input_image = preprocess_image(image_path)
    segmentation_result = ONNXSession.run([output_name], {input_name: np.asarray(input_image)})[0][0]
    _,product_mask = cv2.threshold(segmentation_result[0],0.5,255,cv2.THRESH_BINARY)
    product_mask = product_mask.astype(np.uint8)
    contours, _ = cv2.findContours(product_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    x = x*2+150
    y =y*2+100
    w = 230
    h =256
    crop_img = input_image[:,:,y:y+h, x:x+w]
    crop_img = transforms.Resize((256,256))(crop_img)
    # test_img= transforms.ToPILImage()(crop_img[0])
    # test_img.save('test.png')
    #crop_img = cv2.resize(crop_img,(256,256))
    with torch.no_grad():
        features = model(crop_img.to('cuda')).cpu()
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
        cache_path = '.cache'
        if(not os.path.isdir(cache_path)):
            os.mkdir(cache_path)
        cache_file_path = os.path.join(cache_path,'data.npz')
        if (os.path.exists(cache_file_path)):
            cache_data = dict(np.load(cache_file_path))
        else:
            cache_data = {}
        
        # Path to the folder containing all images
        image_folder_path = self.search_folder_path

        # Path to the reference image
        reference_image_path = self.reference_path

        # Get features for the reference image
        reference_features = get_image_features(reference_image_path)

        # Calculate similarity for all images in the folder
        image_similarity_list = []
        count = 0
        dataset = CustomImageDataset(root_dir=image_folder_path,cache_data=dict(cache_data), transform=preprocess)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
        #images = get_image_list(image_folder_path)
        total  = len(dataset)
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(data_loader),desc="Calculating Similarity"):
                image = batch['image_process']
                image_name = batch['name'][0]
                cache_type = batch['type'][0]
                image_path = os.path.join(image_folder_path, image_name)
                if (cache_type == 0):
                    segmentation_result = ONNXSession.run([output_name], {input_name: np.asarray(image)})[0][0]
                    _,product_mask = cv2.threshold(segmentation_result[0],0.5,255,cv2.THRESH_BINARY)
                    product_mask = product_mask.astype(np.uint8)
                    contours, _ = cv2.findContours(product_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if (len(contours)>0):
                        c = max(contours, key = cv2.contourArea)
                        x,y,w,h = cv2.boundingRect(c)
                        x = x*2+150
                        y =y*2+100
                        w = 230
                        h =256
                        crop_img = image[:,:,y:y+h, x:x+w]
                        crop_img = transforms.Resize((256,256))(crop_img)
                        features = model(crop_img.to('cuda')).cpu()
                        image_features = features.squeeze().numpy()
                        cache_data[image_name] = image_features                    
                        similarity_score = cosine_similarity(reference_features, image_features)
                        image_similarity_list.append((image_path, similarity_score))
                        self.update_progress.emit(count*100/total, f"Calculating {count}/{total}")                        
                else:
                    similarity_score = cosine_similarity(reference_features, image[0])
                    image_similarity_list.append((image_path, similarity_score))
                    self.update_progress.emit(count*100/total, f"Calculating {count}/{total}")
                count = count+1
        np.savez(cache_file_path, **cache_data)
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
        try:
            item = self.list_widget.selectedItems()[0]
            image_path = item.data(Qt.UserRole)
            if image_path:
                self.display_image(image_path)
        except:
            pass
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
    #reference_features = get_image_features(r'E:\img_fuji\images\06-50-58-893.png')
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='light_blue.xml')
    ex = ImageSimilarityApp()
    ex.show()

    # window = MainWindow(dbconn)
    app.exec_()
    sys.exit(0)