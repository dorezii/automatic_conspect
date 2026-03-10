import os
from mpi4py import MPI
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\dondu\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

import cv2
import time
from concurrent.futures import ProcessPoolExecutor
dir = 'phil2'
# Путь к папке с изображениями

input_folder = f'C:\\Users\\dondu\\PycharmProjects\\KeyFrames\\{dir}'
output_folder = f'{dir}'
os.makedirs(output_folder, exist_ok=True)


# Функция для обработки одного изображения
def process_image(image_name):
    image_path = os.path.join(input_folder, image_name)
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img, lang='rus')
    cleaned = re.sub(r'[^а-яА-Яa-zA-Z0-9.,!?()\\s\\-]', '', text)
    # Удаляем лишние пробелы
    cleaned = re.sub(r'\\s+', ' ', cleaned).strip()
    output_path = os.path.join(output_folder, image_name + '.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return text  # Возвращаем распознанный текст для агрегации


if __name__ == '__main__':
    # image_files = [f for f in os.listdir(input_folder) if
    #                f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

    num_workers = os.cpu_count()

    num_workers = 12
    iter = 3

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        start = time.time()
        all_images = [f for f in os.listdir(input_folder) if
                      f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        chunks = [all_images[i::size] for i in range(size)]
    else:
        chunks = None

    my_images = comm.scatter(chunks, root=0)

    for image_name in my_images:
        image_path = os.path.join(input_folder, image_name)
        try:
            text = process_image(image_path)
        except Exception as e:
            text = f"Ошибка при распознавании {image_name}: {e}"


    # end = time.time()
    # elapsed = end - start
    # Синхронизация процессов
    comm.Barrier()

    if rank == 0:
        total_time = time.time() - start
        print(f"{size}: {total_time:.2f}")
