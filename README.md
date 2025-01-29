# text to image

```bash
python3 -m venv .venv
source .venv/bin/activate

source .venv/Scripts/activate # for windows


#-----------------------------------
pip install torch diffusers transformers accelerate huggingface-hub datasets pillow


# To train the model:
python script.py --mode train --prompt "your training description" --image_dir "training_images"

# To generate images after training:
python script.py --mode generate --prompt "your generation prompt"

#------------------------------------- example ------------------------------------------
# Train the model
python script.py --mode train --prompt "portrait in anime style" --image_dir "anime_portraits"

# Generate new images
python script.py --mode generate --prompt "a happy anime girl with blue hair"

#---------------------------------------

pip install -r requirements.txt


python t.py
```