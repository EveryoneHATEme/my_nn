import os
import cv2
import numpy
import requests


random_image_url = 'https://picsum.photos/200'
cat_url = 'https://api.thecatapi.com/v1/images/search'


class ImageHandler:
    @staticmethod
    def prepare_images(path):
        labels_list, images_list = ImageHandler.open_images(path)

        images_array = numpy.array(images_list, dtype='float') / 255.0
        labels_array = numpy.array(labels_list)

        return labels_array, images_array

    @staticmethod
    def open_images(path):
        labels_list = []
        images_list = []

        for sub_path in os.listdir(path):
            full_sub_path = os.path.join(path, sub_path)
            if os.path.isdir(full_sub_path):
                labels_from_dir, images_from_dir = ImageHandler.open_images(full_sub_path)
                labels_list.extend(labels_from_dir)
                images_list.extend(images_from_dir)

            elif os.path.isfile(full_sub_path):
                labels_list.append(full_sub_path.split(os.path.sep)[-2])
                image = cv2.imread(full_sub_path)
                filtered_image = cv2.resize(image, (64, 64))
                images_list.append(filtered_image)

        return labels_list, images_list

    @staticmethod
    def prepare_image(image):
        result_image = cv2.resize(image, (64, 64)).astype('float') / 255.0
        return result_image.reshape((1, result_image.shape[0], result_image.shape[1], result_image.shape[2]))

    @staticmethod
    def open_image(path):
        return cv2.imread(path)

    @staticmethod
    def show_result(image, text):
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)
        cv2.imshow('Result', image)
        cv2.waitKey(0)

    @staticmethod
    def download_images():
        if not os.path.exists('images'):
            os.mkdir('images')

        if not os.path.exists('images/cats'):
            os.mkdir('images/cats')

        if not os.path.exists('images/others'):
            os.mkdir('images/others')

        for i in range(250):
            print(f'Iteration: {i} (cat image)')
            response = requests.get(cat_url)
            url = response.json()[0]['url']
            extension = url.split('.')[-1]
            response = requests.get(url)
            with open(f'images/cats/{i}.{extension}', 'wb') as img:
                img.write(response.content)

        for i in range(750):
            print(f'Iteration: {i + 250} (random picture)')
            response = requests.get(random_image_url)
            with open(f'images/others/{i}.jpg', 'wb') as img:
                img.write(response.content)

        print('Finished!')
