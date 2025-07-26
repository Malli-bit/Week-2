from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

data_path = "resized/Tree_Species_Dataset"

datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255
)

train_data = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(train_data.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint("tree_classifier_model.h5", monitor="val_accuracy", save_best_only=True, mode="max")
model.fit(train_data, validation_data=val_data, epochs=5, callbacks=[checkpoint])



# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.optimizers import Adam

# # Load base model
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

# # Freeze base layers
# for layer in base_model.layers:
#     layer.trainable = False

# # Add custom head
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(128, activation='relu')(x)
# predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)

# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# # Load data
# train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# train_gen = train_datagen.flow_from_directory(
#     'dataset_path',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',
#     subset='training'
# )

# val_gen = train_datagen.flow_from_directory(
#     'dataset_path',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation'
# )

# model.fit(train_gen, validation_data=val_gen, epochs=10)
# model.save('tree_classifier.h5')
   