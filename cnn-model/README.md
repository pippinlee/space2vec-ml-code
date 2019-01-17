## CNN Model

We have given 3 files:

1. The iPython/Jupyter notebook file (`Type Ia Supernova Classifier - Convolutional Neural Network.ipynb`)
2. The .py file outputted from iPython/Jupyter (`Type Ia Supernova Classifier - Convolutional Neural Network.py`)
3. Functions that are used in the other 2 files (`space_utils.py`)

For this specific model, we strongly recommend the iPython/Jupyter notebook file. The code
explanation is a lot nicer in the notebook interface, it will be easier to learn what is going on!

There are 2 main data files that are used in the code:

1. all_object_data_in_dictionary_format.pkl
2. normalized_image_object_data_in_numpy_format.pkl

The descriptions for what each one does it in the code.

However, there are 3 different sizes of each with the links below:

| Filename                                                     | S3 Link                                                                                                      | File Size |
|--------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|-----------|
| all_object_data_in_dictionary_format.pkl                     | https://s3.amazonaws.com/space2vec-public/post3/all_object_data_in_dictionary_format.pkl                     | 6.7GB     |
| normalized_image_object_data_in_numpy_format.pkl             | https://s3.amazonaws.com/space2vec-public/post3/normalized_image_object_data_in_numpy_format.pkl             | 13.0GB    |
| small_all_object_data_in_dictionary_format.pkl               | https://s3.amazonaws.com/space2vec-public/post3/small_all_object_data_in_dictionary_format.pkl               | 772.0MB   |
| small_normalized_image_object_data_in_numpy_format.pkl       | https://s3.amazonaws.com/space2vec-public/post3/small_normalized_image_object_data_in_numpy_format.pkl       | 1.5GB     |
| extra_small_all_object_data_in_dictionary_format.pkl         | https://s3.amazonaws.com/space2vec-public/post3/extra_small_all_object_data_in_dictionary_format.pkl         | 386.0MB   |
| extra_small_normalized_image_object_data_in_numpy_format.pkl | https://s3.amazonaws.com/space2vec-public/post3/extra_small_normalized_image_object_data_in_numpy_format.pkl | 744.2MB   |

You can pick any of the links from that table and use `wget <link>` to download the data.
