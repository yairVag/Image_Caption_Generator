from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from torch.utils.data import Dataset
from PIL import Image
import config as cfg
import pandas as pd
import torch


class MyDataset(Dataset):
    """
    MyDataset is the class that helps to load the images to the transformer
    """

    def __init__(self, root_dir, df, feature_extractor, decoder_tokenizer, max_target_length=cfg.MAX_LENGTH):
        self.root_dir = root_dir
        self.df = df
        self.max_target_length = max_target_length
        self.feature_extractor = feature_extractor
        self.decoder_tokenizer = decoder_tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['image'][idx]
        caption = self.df['caption'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB").resize((cfg.IMAGE_SIZE_VIT, cfg.IMAGE_SIZE_VIT))
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values

        # add labels (input_ids) by encoding the text
        labels = self.decoder_tokenizer(caption, padding="max_length",
                                        max_length=self.max_target_length,
                                        truncation=True).input_ids

        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.decoder_tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


def create_df():
    """
    create_df() creates train and test Dataframes with the captions of the images
    :return: Dataframes: df_train and df_test
    """
    df = pd.read_csv(cfg.CAPTIONS)
    df_train = df[:cfg.TRAIN_SIZE]
    df_test = df[cfg.TRAIN_SIZE:]

    # we reset the indices to start from zero
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    return df_train, df_test


def set_model_configuration(model, decoder_tokenizer):
    """
    set_train_configuration() is setting the configuration for the training
    :param model: VisionEncoderDecoderModel, the full pipeline of the transformer for image captioning
    :param decoder_tokenizer: AutoTokenizer, the tokens decoder.
    """

    # setting special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = decoder_tokenizer.bos_token_id
    model.config.pad_token_id = decoder_tokenizer.pad_token_id
    model.config.decoder.pad_token_id = decoder_tokenizer.pad_token_id

    # making sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # setting beam search parameter
    model.config.eos_token_id = decoder_tokenizer.sep_token_id
    model.config.max_length = cfg.MAX_LENGTH
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = cfg.NO_REPEAT_NGRAM_SIZE
    model.config.length_penalty = cfg.LENGTH_PENALTY
    model.config.num_beams = cfg.NUM_BEAMS
    model.decoder.resize_token_embeddings(len(decoder_tokenizer))

    # freezing the encoder
    for param in model.encoder.parameters():
        param.requires_grad = False


def set_train_configuration(model, feature_extractor, train_dataset, eval_dataset):
    """
    set_train_configuration() is setting the configuration for the training
    :param model: VisionEncoderDecoderModel, the full pipeline of the transformer for image captioning
    :param feature_extractor: AutoFeatureExtractor, the image features extractor
    :param train_dataset: train_dataset of the images and captions
    :param eval_dataset: eval_dataset of the images and captions
    :return: training_args, trainer
    """
    training_args = Seq2SeqTrainingArguments(
        "Image-Captioning",
        predict_with_generate=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        per_device_train_batch_size=cfg.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=cfg.EVAL_BATCH_SIZE,
        overwrite_output_dir=False,
        fp16=False,
        learning_rate=cfg.LR,
        num_train_epochs=cfg.NUM_TRAIN_EPOCHS,
        load_best_model_at_end=True,
        logging_steps=cfg.LOGGING_STEPS,
        eval_steps=cfg.EVAL_STEPS,
        save_steps=cfg.SAVE_STEPS,
        report_to="none",
        disable_tqdm=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    return training_args, trainer


def fit():
    """
    fit() is splitting the images' data to train, validation and test and use them with Datasets
    then it trains the model, and saves it
    """

    answer = input('Warning- It is recommended to continue only if you have GPU\n Continue? (y,n)\n')
    if answer.lower() == 'y':
        # initialize a bert2gpt2 from a pretrained BERT and GPT2 models.
        # Note that the cross-attention layers will be randomly initialized
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained("google/vit-base-patch16-224-in21k", "gpt2")

        # feature extractor from image
        feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

        # decoder tokens
        decoder_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

        # create df_test and Dataframe, and use them to create Datasets
        df_train, df_test = create_df()
        train_dataset = MyDataset(cfg.IMAGES_FOLDER, df_train, feature_extractor, decoder_tokenizer)
        eval_dataset = MyDataset(cfg.IMAGES_FOLDER, df_test, feature_extractor, decoder_tokenizer)

        set_model_configuration(model, decoder_tokenizer)

        training_args, trainer = set_train_configuration(model, feature_extractor, train_dataset, eval_dataset)

        trainer.train()

        answer = input('Overwrite model? (y,n)\n')
        if answer.lower() == 'y':
            # save model
            model.save_pretrained(cfg.TRANSFORMER_PATH)
