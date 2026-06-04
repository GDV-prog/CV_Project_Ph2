import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import requests
from io import BytesIO
from pathlib import Path

# ----------------------------------------------------------------------
# УНИВЕРСАЛЬНЫЙ ПОИСК ФАЙЛОВ (РАБОТАЕТ И В ЛОКАЛЕ, И НА GITHUB)
# ----------------------------------------------------------------------
def get_project_root():
    """Определяет корень проекта по маркерным файлам."""
    script_dir = Path(__file__).parent.resolve()
    for parent in [script_dir] + list(script_dir.parents):
        if (parent / "README.md").exists() and ((parent / "requirements.txt").exists() or (parent / "data.yaml").exists()):
            return parent
    return script_dir

PROJECT_ROOT = get_project_root()

def find_weights():
    """Ищет веса модели yolo_wind.pt."""
    candidates = [
        PROJECT_ROOT / "models" / "yolo_wind.pt",
        PROJECT_ROOT / "models" / "best.pt",
        PROJECT_ROOT / "wind_train" / "exp" / "weights" / "best.pt",
        PROJECT_ROOT / "notebooks" / "wind_train" / "exp" / "weights" / "best.pt",
        PROJECT_ROOT / "pages" / "wind_train" / "exp" / "weights" / "best.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

def find_image_path(filename):
    """Ищет график в папке images/ (сначала в корне, потом в pages/notebooks)."""
    candidates = [
        PROJECT_ROOT / "images" / filename,
        PROJECT_ROOT / "pages" / "images" / filename,
        PROJECT_ROOT / "notebooks" / "images" / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

# ----------------------------------------------------------------------
# ОСНОВНАЯ ФУНКЦИЯ СТРАНИЦЫ
# ----------------------------------------------------------------------
def render_wind_detection_page():
    st.set_page_config(page_title="Детекция ветрогенераторов", layout="wide")
    st.markdown("# 🌬️ Детекция ветрогенераторов (YOLOv11)")

    # === БОКОВАЯ ПАНЕЛЬ С НАСТРОЙКАМИ ===
    with st.sidebar:
        st.markdown("## ⚙️ Настройки детекции")
        conf_threshold = st.slider(
            "Порог уверенности (Conf)",
            min_value=0.0, max_value=1.0, value=0.25, step=0.05,
            help="Минимальная уверенность модели для фиксации объекта."
        )
        iou_threshold = st.slider(
            "Порог перекрытия рамок (IoU)",
            min_value=0.0, max_value=1.0, value=0.45, step=0.05,
            help="Порог для отсечения дублирующихся рамок."
        )
        st.markdown("---")
        st.info("Модель обучена на датасете ветрогенераторов")

    # === ЗАГРУЗКА МОДЕЛИ ===
    weights_path = find_weights()
    if weights_path is None:
        st.error("❌ Файл весов не найден! Убедитесь, что `models/yolo_wind.pt` существует.")
        return

    @st.cache_resource
    def load_model(path):
        return YOLO(path)

    with st.spinner("Загрузка модели YOLO..."):
        model = load_model(weights_path)
    st.sidebar.success("✅ Модель загружена")

    # === ВЫБОР ИСТОЧНИКА ИЗОБРАЖЕНИЙ ===
    input_method = st.radio("Выберите способ подачи изображения:", ("Загрузить файлы с ПК", "Указать прямую URL-ссылку"))
    images_to_process = []

    if input_method == "Загрузить файлы с ПК":
        uploaded_files = st.file_uploader(
            "Выберите одно или несколько изображений",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="wind_uploader"
        )
        if uploaded_files:
            for file in uploaded_files:
                images_to_process.append((Image.open(file), file.name))
    else:
        url_input = st.text_input("Вставьте прямую ссылку на изображение (JPG / PNG):", placeholder="https://example.com")
        if url_input:
            with st.spinner("Скачивание изображения по ссылке..."):
                try:
                    response = requests.get(url_input, timeout=5)
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        images_to_process.append((img, "Изображение по ссылке"))
                    else:
                        st.error(f"Не удалось получить изображение. Ошибка сервера: {response.status_code}")
                except Exception as e:
                    st.error("Не удалось загрузить изображение. Проверьте URL.")

    # === ОБРАБОТКА КАЖДОГО ИЗОБРАЖЕНИЯ ===
    if images_to_process:
        for pil_img, name in images_to_process:
            st.markdown("---")
            st.subheader(f"Источник: {name}")

            # Преобразуем PIL -> numpy (RGB) -> BGR для OpenCV
            img_np = np.array(pil_img)
            # YOLO принимает RGB, но .plot() возвращает BGR
            results = model(img_np, conf=conf_threshold, iou=iou_threshold, verbose=False)
            annotated = results[0].plot()  # BGR
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_img, caption="Оригинал", use_container_width=True)
            with col2:
                st.image(annotated_rgb, caption=f"Результат (Conf: {conf_threshold}, IoU: {iou_threshold})", use_container_width=True)

            # Вывод количества найденных объектов
            boxes = results[0].boxes
            if boxes is not None:
                st.write(f"Обнаружено объектов: {len(boxes)}")
            else:
                st.write("Объектов не обнаружено")

    # === БЛОК МЕТРИК И ГРАФИКОВ (аналогично face.py) ===
    with st.expander("📊 Информация о модели, качестве и процессе обучения (Ветрогенераторы)"):
        st.markdown("""
        **Метрики обучения YOLOv11 Nano**  
        * **Число эпох обучения:** 50  
        * **Объем выборки:** см. data.yaml (train/valid)  
        """)

        # Создаём внутренние вкладки для графиков
        metric_tabs = st.tabs(["📈 Результаты обучения (Loss/mAP)", "🎯 PR Кривая", "🧩 Матрица ошибок"])

        # 1. Результаты и Loss
        with metric_tabs[0]:
            st.markdown("**Графики функций потерь и метрик качества**")
            img_path = find_image_path("wind_results.png")
            if img_path:
                st.image(img_path, caption="YOLOv11 Training Results", use_container_width=True)
            else:
                st.info("Файл `images/wind_results.png` ещё не добавлен в проект.")

            st.markdown("""
            **Анализ графиков обучения:**  
            * **Функции потерь:** box_loss, cls_loss, dfl_loss стабильно снижаются на обеих выборках → нет переобучения.  
            * **Метрики:** mAP50 и mAP50-95 плавно растут и выходят на плато к концу обучения.
            """)

        # 2. PR-кривая
        with metric_tabs[1]:
            st.markdown("**Кривая Точности-Полноты (Precision-Recall Curve)**")
            img_path = find_image_path("wind_boxpr_curve.png")
            if img_path:
                st.image(img_path, caption="Precision-Recall Curve", use_container_width=True)
            else:
                st.info("Файл `images/wind_boxpr_curve.png` ещё не добавлен.")
            st.markdown("*Кривая близка к правому верхнему углу, что говорит о высоком балансе точности и полноты.*")

        # 3. Матрица ошибок
        with metric_tabs[2]:
            st.markdown("**Матрица ошибок (Confusion Matrix) финальной эпохи**")
            img_path = find_image_path("wind_confusion_matrix.png")
            if img_path:
                st.image(img_path, caption="Confusion Matrix", use_container_width=True)
            else:
                st.info("Файл `images/wind_confusion_matrix.png` ещё не добавлен.")
            st.markdown("*Основная доля ошибок — пропуск мелких или перекрытых объектов.*")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    render_wind_detection_page()