import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import requests
from io import BytesIO

def process_and_blur_faces(open_cv_image, model, conf_val, iou_val):
    """
    Вспомогательная функция для поиска и размытия лиц на OpenCV изображении.
    Передает параметры слайдеров прямо в модель YOLO.
    """
    # Инференс YOLO с динамическими параметрами уверенности и перекрытия
    results = model(open_cv_image, conf=conf_val, iou=iou_val, verbose=False)
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box[:4])
            face_region = open_cv_image[ymin:ymax, xmin:xmax]
            
            if face_region.size == 0:
                continue
            
            # Динамический размер ядра размытия в зависимости от размера лица
            ksize = int(max(face_region.shape[:2]) / 2) | 1
            if ksize < 3: ksize = 3
            
            # Размытие Гаусса
            blurred_face = cv2.GaussianBlur(face_region, (ksize, ksize), 0)
            open_cv_image[ymin:ymax, xmin:xmax] = blurred_face
            
    # Конвертируем обратно OpenCV (BGR) -> RGB для Streamlit
    return cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)


def render_face_detection_page():
    # === БОКОВАЯ ПАНЕЛЬ С НАСТРОЙКАМИ (ДОБАВЛЕНО) ===
    with st.sidebar:
        st.markdown("## ⚙️ Настройки детекции")
        st.write("Управляйте чувствительностью модели в реальном времени.")
        
        # Слайдер порога уверенности с предустановленной серединой
        conf_threshold = st.slider(
            "Порог уверенности (Conf)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.1,  # По умолчанию ползунок стоит ровно посередине шкалы
            step=0.05,
            help="Минимальная уверенность модели для фиксации лица. Чем выше, тем меньше ложных срабатываний."
        )
        
        
        # Слайдер порога перекрытия рамок
        iou_threshold = st.slider(
            "Порог перекрытия рамок (IoU)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.45, # Стандартное оптимальное значение для NMS алгоритма
            step=0.05,
            help="Порог для отсечения дублирующихся рамок вокруг одного лица."
        )
        
        st.markdown("---")

    # === ОСНОВНОЙ КОНТЕНТ СТРАНИЦЫ ===
    st.header("🎭 Автоматическое маскирование лиц")
    st.write("Вы можете загрузить фотографии с диска или указать прямую ссылку на изображение из интернета.")
    
    # Выбор способа загрузки через радио-кнопку
    input_method = st.radio("Выберите способ подачи изображения:", ("Загрузить файлы с ПК", "Указать прямую URL-ссылку"))
    
    model_path = "models/yolo_face.pt"
    pil_images_to_process = [] # Список кортежей (PIL_Image, имя_файла)

    # --- СБОР ИЗОБРАЖЕНИЙ ---
    if input_method == "Загрузить файлы с ПК":
        uploaded_files = st.file_uploader(
            "Выберите одно или несколько изображений", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True,
            key="face_uploader"
        )
        if uploaded_files:
            for file in uploaded_files:
                pil_images_to_process.append((Image.open(file), file.name))

    else:
        url_input = st.text_input("Вставьте прямую ссылку на изображение (JPG / PNG):", placeholder="https://example.com")
        if url_input:
            with st.spinner("Скачивание изображения по ссылке..."):
                try:
                    response = requests.get(url_input, timeout=5)
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        pil_images_to_process.append((img, "Изображение по ссылке"))
                    else:
                        st.error(f"Не удалось получить изображение. Ошибка сервера: {response.status_code}")
                except Exception as e:
                    st.error("Не удалось загрузить изображение. Проверьте правильность URL-ссылки или формат файла.")

        if pil_images_to_process:
        # Если файла нет на диске, скачиваем его из твоих GitHub Releases
            if not os.path.exists(model_path):
                with st.spinner("Загрузка обученных весов модели из GitHub Releases..."):
                    os.makedirs("models", exist_ok=True)
                
                # Твоя ссылка на релиз (замени на свою точную ссылку, когда опубликуешь)
                url = "https://github.com/mrkuzn/CV_Project_Ph2/releases/download/CV_project/yolo_face.pt"
                
                try:
                    import urllib.request
                    opener = urllib.request.build_opener()
                    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                    urllib.request.install_opener(opener)
                    
                    urllib.request.urlretrieve(url, model_path)
                    st.success("Веса модели успешно загружены!")
                except Exception as e:
                    st.error(f"Не удалось скачать веса автоматически: {e}. Положите файл yolo_face.pt в папку models/ вручную.")
                    return
            
        with st.spinner("Загрузка модели YOLO..."):
            model = YOLO(model_path)
        
        # Обрабатываем собранные картинки в цикле
        for pil_img, name in pil_images_to_process:
            st.markdown("---")
            st.subheader(f"Источник: {name}")
            
            # Конвертируем PIL -> OpenCV (BGR)
            open_cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # Запускаем функцию детекции и размытия с параметрами из слайдеров
            processed_image = process_and_blur_faces(open_cv_image, model, conf_threshold, iou_threshold)
            
            # Отображаем картинки «До» и «После» в две колонки
            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_img, caption="Оригинал", use_container_width=True)
            with col2:
                st.image(processed_image, caption=f"Результат (Conf: {conf_threshold}, IoU: {iou_threshold})", use_container_width=True)
                
    # --- БЛОК АНАЛИТИКИ И ТЗ ---
    with st.expander("📊 Информация о модели, качестве и процессе обучения (Лица)"):
        st.markdown("""
        ### Метрики обучения YOLOv11 Nano
        * **Число эпох обучения:** 30
        * **Объем выборки:** 3347 изображений в валидации (10299 объектов лица)
        """)
        



       
       
       
       
 # Создаем внутренние вкладки для графиков
        metric_tabs = st.tabs(["📈 Результаты обучения (Loss/mAP)", "🎯 PR Кривая", "🧩 Матрица ошибок"])
        
        # 1. Первая вкладка: Результаты и Лоссы
        with metric_tabs[0]:
            st.markdown("**Графики функций потерь (Loss) и метрик качества по эпохам**")
            if os.path.exists("images/face_results.png"):
                st.image("images/face_results.png", caption="YOLOv11 Training Results", use_container_width=True)
            else:
                st.info("Файл `images/face_results.png` еще не добавлен в проект.")
                
            # Помещаем текстовый анализ внутрь первой вкладки ПОД картинку
            st.markdown("""
            **Анализ графиков обучения:**
            * **Функции потерь (Loss):** Графики `box_loss`, `cls_loss` и `dfl_loss` стабильно снижаются как на обучающей, так и на валидационной выборках. Это свидетельствует об успешной сходимости модели и отсутствии признаков переобучения (overfitting).
            * **Метрики качества:** Метрики `mAP50` и `mAP50-95` плавно растут и выходят на плато к 30-й эпохе, подтверждая, что модель извлекла максимум полезных признаков из предоставленного датасета.
            """)
                
        # 2. Вторая вкладка: PR-кривая
        with metric_tabs[1]:
            st.markdown("**Кривая Точности-Полноты (Precision-Recall Curve)**")
            if os.path.exists("images/face_pr_curve.png"):
                st.image("images/face_pr_curve.png", caption="Precision-Recall Curve", use_container_width=True)
            else:
                st.info("Файл `images/face_pr_curve.png` еще не добавлен в проект.")
                
            st.markdown("*Кривая близка к правому верхнему углу, что говорит о высоком балансе между точностью детекции и полнотой охвата лиц.*")
                
        # 3. Третья вкладка: Матрица ошибок
        with metric_tabs[2]:
            st.markdown("**Матрица ошибок (Confusion Matrix) финальной эпохи**")
            if os.path.exists("images/face_confusion_matrix.png"):
                st.image("images/face_confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
            else:
                st.info("Файл `images/face_confusion_matrix.png` еще не добавлен в проект.")
                
            st.markdown("*Модель безошибочно находит 8547 лиц. Основная доля ошибок приходится на пропуск очень мелких или перекрытых лиц на заднем плане (1752 объекта).*")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render_face_detection_page()
