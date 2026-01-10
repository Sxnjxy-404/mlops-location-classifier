from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

data = datagen.flow_from_directory(
    "data/train",
    target_size=(224,224),
    batch_size=16
)

print("Classes:", data.class_indices)
print("Samples:", data.samples)
