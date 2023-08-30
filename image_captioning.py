import h5py
import json
import cv2
import argparse
import sys
import train_cnn_model
import train_transformer
import gradio as gr
import numpy as np
import pandas as pd
import config as cfg
import tensorflow as tf
from collections import defaultdict
from pyinflect import getInflection
from argparse import RawTextHelpFormatter
from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer


def load_model_ext(filepath):
    """
    load_model_ext() is loading keras model and the classes of the model's prediction
    :param filepath: string with the path of the model
    :return: model: keras model
    :return: meta_data: classes that the model predicts
    """
    loaded_model = tf.keras.models.load_model(filepath)
    f = h5py.File(filepath, mode='r')
    meta_data = None
    if 'my_meta_data' in f.attrs:
        meta_data = f.attrs.get('my_meta_data')
    f.close()
    return loaded_model, meta_data


def create_caption_cnn(img):
    """
    create_caption_cnn() create a caption for an image using a cnn model that was trained on 'Flickr image dataset'
    :param img: a numpy array of the image
    :return: a string of the image caption
    """

    # preprocess image
    img = cv2.resize(img, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))[np.newaxis, :]
    img = img * cfg.RESCALE

    # predict probabilities of the labels
    pred = cnn_model.predict(img)

    # create labels prediction list by the probabilities threshold
    y_pred = np.array([[1 if i > cfg.PROBA_THRESH else 0 for i in j] for j in pred])
    pred_labels = cnn_labels[y_pred.astype(bool)[0]]

    # organize labels by their pos tags
    labels_pos_tag = {pred_label: cfg.IMAGE_LABELS[pred_label][-1] for pred_label in pred_labels}
    mydict = dict(labels_pos_tag)
    pos_tag_labels = defaultdict(list)
    for key, value in mydict.items():
        pos_tag_labels[value].append(key)

    # create caption
    caption = ['A']
    if 'NOUN1' in pos_tag_labels:
        for n, noun in enumerate(pos_tag_labels['NOUN1']):
            if n > 0:
                caption += ['and a']
            caption += [noun]
    if 'VERB' in pos_tag_labels:
        for n, verb in enumerate(pos_tag_labels['VERB']):
            if n > 0:
                caption += ['and']
            verb = getInflection(verb, tag='VBG')[0]
            caption += [verb]
    if 'NOUN2' in pos_tag_labels:
        for n, noun in enumerate(pos_tag_labels['NOUN2']):
            if n > 0:
                caption += ['and']
            caption += ['on the', noun]

    return ' '.join(caption)


def create_caption_transformer(img):
    """
    create_caption_transformer() create a caption for an image using a transformer model
    that was trained on 'Flickr image dataset'
    :param img: a numpy array of the image
    :return: a string of the image caption
    """

    sample = feature_extractor(img, return_tensors="pt").pixel_values.to('cpu')
    caption_ids = model.generate(sample)[0]
    caption_text = tokenizer.decode(caption_ids, skip_special_tokens=True)
    caption_text = caption_text.split('.')[0]
    return caption_text


def create_captions_in_csv():
    """
    create_captions_in_csv() create captions for all the images in the folder cfg.IMAGES_EXAMPLES_FOLDER
    and save the captions in the csv file cfg.CSV_FILE =
    """
    if args.model == 'cnn':
        fnc = create_caption_cnn
    else:  # 'transformer'
        fnc = create_caption_transformer

    captions = list()
    images = list()
    n_images = len(cfg.IMAGES_EXAMPLES)
    for n, file in enumerate(cfg.IMAGES_EXAMPLES):
        print(f'progress: {100 * (n + 1) / n_images:.2f}%')
        # load file and preprocess
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # create caption and print the result
        pred_caption = fnc(img)
        captions.append(pred_caption)
        images.append(file)
    # print and save the captions
    df_captions = pd.DataFrame()
    df_captions['images'] = images
    df_captions['captions'] = captions
    print(df_captions)
    df_captions.to_csv(cfg.CSV_FILE)
    print(f"Captions were saved to the file '{cfg.CSV_FILE}'")


def create_gui_link():
    """
    create_gui_link() creates a link to the gui in the local machine and in an external url link
    """
    # create GUI object and print a link to the GUI
    if args.model == 'cnn':
        fnc = create_caption_cnn
    else:  # 'transformer'
        fnc = create_caption_transformer

    gr.Interface(fn=fnc,
                 inputs="image",
                 outputs='text',
                 examples=cfg.IMAGES_EXAMPLES
                 ).launch(share=True)


def parse_args(args_string_list):
    """
    parse_args() is parsing the py file input arguments into the struct args
    :param args_string_list: a list of the input arguments of the py file
    :return a Struct with all the input arguments of the py file
    """

    # Interface definition
    parser = argparse.ArgumentParser(description="This program can generate caption to a given.\n"
                                                 "It allows to open a GUI or to caption all the images in the folder\n"
                                                 "cfg.IMAGES_EXAMPLES_FOLDER.\n"
                                                 "It also allows to retrain the models using the hyperparameters"
                                                 "in the config.py.",
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('-p', '--program_mode', type=str, help='Options: csv, gui, train', required=True)
    parser.add_argument('-m', '--model', type=str, help='Options: transformer, cnn', default='transformer')

    return parser.parse_args(args_string_list)


def main():
    """
    main() getting the input arguments of the py file
    main() also catches exceptions
    """

    if args.program_mode == 'csv':
        try:
            create_captions_in_csv()
        except cv2.error as e:
            print(f'{e}Warning: All the files in the folder "{cfg.IMAGES_EXAMPLES_FOLDER}" must be images!')

    elif args.program_mode == 'train':
        if args.model == 'cnn':
            train_cnn_model.fit()
        else:  # 'transformer'
            train_transformer.fit()

    else:  # 'gui'
        create_gui_link()


# ------- load model ------- #
args = parse_args(sys.argv[1:])
if args.model == 'cnn':
    cnn_model, loaded_labels_string = load_model_ext(cfg.CNN_MODEL_FILE)
    cnn_labels = np.array(json.loads(loaded_labels_string))
else:  # 'transformer'
    model = VisionEncoderDecoderModel.from_pretrained(cfg.TRANSFORMER_PATH).to('cpu')
    feature_extractor = AutoFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

if __name__ == '__main__':
    main()
