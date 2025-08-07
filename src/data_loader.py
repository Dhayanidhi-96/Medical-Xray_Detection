from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, target_size = (150, 150), batch_size = 32):
    datagen = ImageDataGenerator(rescale=1./255)

    data = datagen.flow_from_directory(
        data_dir,
        target_size = target_size,
        batch_size = batch_size,
        class_mode = 'binary',
        shuffle = True
    )

    return data