import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pixellib
from pixellib.instance import instance_segmentation


def object_detection_on_an_image():

    segment_image = instance_segmentation(infer_speed='low')
    segment_image.load_model() # Сюда вставляем путь до изображения


    target_class = segment_image.select_target_classes(person=True)
    result = segment_image.segmentImage(image_path='object.jpg',
                                        show_bboxes=True,
                                        segment_target_classes=target_class,
                                        output_image_name='segmentation.jpg')



    object_count = len(result[0]['scores'])
    print(f'Найдено объектов: {object_count}')


def main():
    object_detection_on_an_image()


if __name__ == '__main__':
    main()