import os
from model import build_cnn_model
from data_loader import load_data

train_dir = r"D:\Project\Project data set\chest_xray\train"
val_dir = r"D:\Project\Project data set\chest_xray\val"
model_save_path = '../saved_models/cnn_chest_xray.h5'

train_data = load_data(train_dir)
val_data  = load_data(val_dir)

model = build_cnn_model()

histrory = model.fit(
    train_data,
    epochs = 10,
    validation_data = val_data

)

os.makedirs('../saved_models', exist_ok=True)
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
