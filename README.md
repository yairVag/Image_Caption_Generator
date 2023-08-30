# Image Captioning

- This program generates a caption to an image.  
It uses a Vision Encoder Decoder Model (transformer)
that was train on kaggle data set of 8k images.  
You are welcome to check it in the following web app interface: https://huggingface.co/spaces/yairVag/Image_Captioning

- The program can either open a gui for uploading an image and generate the caption,  
or create a csv file with captions for images that are locally in a folder.


- The program has an option to train the model using the kaggle data set.   
you are encouraged to tune the hyperparametrs and try the training yourself (use it only with GPU)


- Last note about the program- It also has the baseline that we used in this project:  
 a pretrained cnn model for multi-label classification. We use the labels to generate the caption.   
the program allows all the options with the cnn model as well- open gui, generate captions locally, train the model.
 


## Steps: 

### Clone the project

```bash
  git clone https://github.com/yairVag/Image_Caption_Generator.git
```

### Go to the project directory

```bash
  cd Image_Caption_Generator
```

### Install dependencies

```bash
  pip install -r requirements.txt
```

### Help flag

```bash
   python .\image_captioning.py -h
   
   optional arguments:
  -h, --help            show this help message and exit
  -p PROGRAM_MODE, --program_mode PROGRAM_MODE
                        Options: csv, gui, train
  -m MODEL, --model MODEL
                        Options: transformer, cnn
```

### Create GUI link
* Note: by default, the program use the transformer model (You can use the cnn model defining the flag '-m cnn' in the CLI)
* You can upload one image at a time and generate a caption
```bash
  python .\image_captioning.py -p gui
  
  Running on local URL:  http://127.0.0.1:7860/
  Running on public URL: https://18192.gradio.app
```
  ![Output in Pycharm](https://user-images.githubusercontent.com/100131903/177012162-bac9cd0e-88cb-429f-bb3b-735a64ee193d.png)


### Create csv of images captions
* Note: by default, the program use the transformer model (You can use the cnn model defining the flag '-m cnn' in the CLI)
* You can upload many images to a local folder and save the folder path  
The path to the folder is defined in the file config.py under the variable IMAGES_EXAMPLES_FOLDER  
The csv file name and path is defined in the file config.py under the variable CSV_FILE
```bash
  python .\image_captioning.py -p csv
```
  ![Output in Pycharm](https://user-images.githubusercontent.com/100131903/177012497-3dde4aa3-d4a7-42fa-aedf-845fd939168d.png)


### Train model
* Note: by default, the program use the transformer model (You can use the cnn model defining the flag '-m cnn' in the CLI)
* The model configuration and hyperparameters are defined in the file config.py. you are encouraged to tune them and try the training yourself (use it only with GPU)
```bash
  python .\image_captioning.py -p train
```
