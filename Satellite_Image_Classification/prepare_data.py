import os
import time
import random
import shutil

try:
    shutil.rmtree('EuroSAT_RGB')
    shutil.rmtree('__MACOSX')
    shutil.rmtree('training_images')
    shutil.rmtree('testing_images')
except Exception as e:
    print(f"The folders do not exist or have already been deleted!")


images_source = 'EuroSAT_RGB'
training_destination = 'training_images'
testing_destination = 'testing_images'

image_class = 0
class_dict = {}

files = os.listdir(images_source)
files.sort()

for file_path in files:
  if file_path[0] != '.':
    images = os.listdir(os.path.join(images_source, file_path))
    sample_size = int(len(images) * 0.8)

    train = []
    final_dest = os.path.join(training_destination, str(image_class))
    os.mkdir(final_dest)

    for file_name in random.sample(images, sample_size):
      shutil.copy2(os.path.join(images_source, file_path, file_name), final_dest)
      train.append(file_name)

    test_images = list(set(images) - set(train))
    final_dest = os.path.join(testing_destination, str(image_class))
    os.mkdir(final_dest)

    for test_image in test_images:
      shutil.copy2(os.path.join(images_source, file_path, test_image), final_dest)

    class_dict[image_class] = file_path
    image_class += 1
