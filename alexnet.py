# Dependencies to silently install
import wget
import zipfile

# Data Files and Labels
wget.download('https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip')
wget.download('https://raw.githubusercontent.com/udacity/pytorch_challenge/master/cat_to_name.json')

with zipfile.ZipFile('/content/flower_data.zip', 'r') as zip_ref:
    zip_ref.extractall('/content')