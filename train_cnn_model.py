from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.metrics import Precision, Recall
from keras.models import save_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import config as cfg
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
import json


def save_model_ext(model, filepath, meta_data=None):
    """
    save_model_ext() is saving keras model and the classes of the model's prediction
    :param model: keras model to be saved
    :param filepath: string with the path to save the model
    :param meta_data: meta_data: classes that the model predicts
    """
    save_model(model, filepath, overwrite=True)
    if meta_data is not None:
        f = h5py.File(filepath, mode='a')
        f.attrs['my_meta_data'] = meta_data
        f.close()


def create_image_generators(df_4_train, df_4_test):
    """
    create_image_generators() create ImageDataGenerator for the train, validation and test sets
    :param df_4_train: Dataframe with the images' information (train set)
    :param df_4_test: Dataframe with the images' information (test set)
    :return: train_gen, valid_gen, test_gen: ImageDataGenerator of the train, validation and test sets
    """
    if cfg.AUGMENTATION:
        datagen = ImageDataGenerator(rescale=cfg.RESCALE, validation_split=cfg.VALIDATION_SPLIT,
                                     rotation_range=cfg.ROTATE, height_shift_range=cfg.SHIFT,
                                     width_shift_range=cfg.SHIFT, horizontal_flip=True)
    else:
        datagen = ImageDataGenerator(rescale=cfg.RESCALE, validation_split=cfg.VALIDATION_SPLIT)

    train_gen = datagen.flow_from_dataframe(
        dataframe=df_4_train,
        directory=cfg.IMAGES_FOLDER,
        x_col="image_name",
        y_col="label",
        subset="training",
        batch_size=cfg.BATCH_SIZE,
        seed=cfg.SEED,
        class_mode="categorical",
        target_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))

    valid_gen = datagen.flow_from_dataframe(
        dataframe=df_4_train,
        directory=cfg.IMAGES_FOLDER,
        x_col="image_name",
        y_col="label",
        subset="validation",
        batch_size=cfg.BATCH_SIZE,
        seed=cfg.SEED,
        class_mode="categorical",
        target_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))

    test_datagen = ImageDataGenerator(rescale=cfg.RESCALE)
    test_gen = test_datagen.flow_from_dataframe(
        dataframe=df_4_test,
        directory=cfg.IMAGES_FOLDER,
        x_col="image_name",
        y_col="label",
        batch_size=1,
        seed=cfg.SEED,
        shuffle=False,
        class_mode="categorical",
        target_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))

    return train_gen, valid_gen, test_gen


def create_and_fit_model(train_generator, valid_generator):
    """
    create_and_fit_model() creates the cnn model architecture and use the hyperparameters
    It's also fitting the model and returns the trained model
    :param train_generator: ImageDataGenerator of the train set images
    :param valid_generator: ImageDataGenerator of the validation set images
    :return: keras model
    """
    pre_trained_model = cfg.PRETRAIN_MODEL(include_top=False, input_shape=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3))
    pre_trained_model.trainable = cfg.TRAIN_ALL

    model = tf.keras.models.Sequential([pre_trained_model,
                                        Flatten(),
                                        Dense(cfg.HEAD_UNITS, activation='relu', trainable=True),
                                        Dense(len(train_generator.class_indices), trainable=True, activation='sigmoid')
                                        ])

    opt = Adam(learning_rate=cfg.LEARNING_RATE)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[Precision(), Recall(), 'accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=cfg.PATIENCE, restore_best_weights=True)
    model.fit(train_generator, validation_data=valid_generator, epochs=cfg.EPOCHS, callbacks=es)

    return model


def print_classification_report(model, train_generator, test_generator):
    """
    print_classification_report() print classification report of the model
    label predictions using the probability threshold "cfg.PROBA_THRESH"
    :param model: keras model
    :param train_generator: ImageDataGenerator of the train set images
    :param test_generator: ImageDataGenerator of the test set images
    """

    IMAGE_LABELS_LIST = list(train_generator.class_indices.keys())
    n_images = len(test_generator.classes)
    n_labels = len(IMAGE_LABELS_LIST)
    encode_labels = np.zeros((n_images, n_labels)).astype(int)
    IMAGE_LABELS_LIST = list(train_generator.class_indices.keys())
    for i, labels in enumerate(test_generator.classes):
        for label_index in labels:
            encode_labels[i, label_index] = 1

    y_prob = model.predict(test_generator)
    y_pred = np.array([[1 if i > cfg.PROBA_THRESH else 0 for i in j] for j in y_prob])
    print('Threshold:', cfg.PROBA_THRESH)
    print(classification_report(encode_labels, y_pred, target_names=IMAGE_LABELS_LIST))


def create_df():
    """
    create_df() creates train and test Dataframes with the labels of the images
    it creates them from the dataframe of images comments
    :return: Dataframes: df_train and df_test
    """
    df_image_comments = pd.read_csv(cfg.IMAGES_FOLDER + cfg.COMMENTS)
    df_image_comments['label'] = df_image_comments[
        ['comment_0', 'comment_1', 'comment_2', 'comment_3', 'comment_4']].apply(find_labels, axis=1)
    df_image_comments = df_image_comments.dropna(axis=0)
    df_train, df_test = train_test_split(df_image_comments, test_size=cfg.TEST_SPLIT, random_state=cfg.SEED)
    return df_train, df_test


def invert_dict(d):
    """
    invert_dict() invert the order of the values and keys in the dictionary
    :param d: a dictionary
    :return: revert dictionary
    """

    inverse = dict()
    for key in d:
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse:
                # If not create a new list
                inverse[item] = [key]
            else:
                inverse[item].append(key)
    return inverse


def find_labels(comments):
    """
    find_labels() finds the desire labels in each image captions
    The desire labels are defined in cfg.IMAGE_LABELS
    :param comments: a series of image comments
    :return: list of labels
    """
    labels = list()
    singles = list()
    for comment in comments:
        stemmer = PorterStemmer()
        singles += [stemmer.stem(plural) for plural in word_tokenize(comment)]

    for key, value in cfg.IMAGE_LABELS.items():
        labels += list(set(value) & set(singles))

    IMAGE_LABELS_REV = invert_dict(cfg.IMAGE_LABELS)
    labels_out = list()
    for label in labels:
        labels_out += IMAGE_LABELS_REV[label]

    return list(set(labels_out))


def fit():
    """
    fit() is splitting the images' data to train, validation and test and use them with image generators
    then it trains the model, prints classification report and allows to save the trained model
    """
    answer = input('Warning- It is recommended to continue only if you have GPU\n Continue? (y,n)\n')
    if answer.lower() == 'y':
        # preprocess and train model
        df_train, df_test = create_df()
        train_g, valid_g, test_g = create_image_generators(df_train, df_test)
        model = create_and_fit_model(train_g, valid_g)
        print_classification_report(model, train_g, test_g)

        answer = input('Overwrite model? (y,n)\n')
        if answer.lower() == 'y':
            # save model and labels
            labels = list(train_g.class_indices.keys())
            labels_string = json.dumps(labels)
            save_model_ext(model, cfg.CNN_MODEL_FILE, meta_data=labels_string)
