from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def build_model(input_shape=(150, 150, 3), num_classes=6):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'
    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode='categorical')
    val_gen = datagen.flow_from_directory(test_dir, target_size=(150,150), batch_size=32, class_mode='categorical')

    model = build_model(num_classes=train_gen.num_classes)
    print("trenowanie")
    model.fit(train_gen, validation_data=val_gen, epochs=5)
    model.save('model/image_model.h5')
    return "Model trained and saved."

def test_model():
    from tensorflow.keras.models import load_model
    model = load_model('model/image_model.h5')
    datagen = ImageDataGenerator(rescale=1./255)
    test_dir = 'dataset/test'
    test_gen = datagen.flow_from_directory(test_dir, target_size=(150,150), batch_size=32, class_mode='categorical')
    loss, acc = model.evaluate(test_gen)
    return {"loss": float(loss), "accuracy": float(acc)}
