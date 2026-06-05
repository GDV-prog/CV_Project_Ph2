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
    """Ищет веса модели yolo11m_wind.pt."""
    candidates = [
        PROJECT_ROOT / "models" / "yolo11m_wind.pt",
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
    # Убрано: st.set_page_config(...) - конфликтует с app.py
    st.markdown("# 🌬️ Детекция ветрогенераторов (YOLOv11m)")

    # === БОКОВАЯ ПАНЕЛЬ С НАСТРОЙКАМИ ===
    with st.sidebar:
        st.markdown("## ⚙️ Настройки детекции")
        conf_threshold = st.slider(
            "Порог уверенности (Conf)",
            min_value=0.0, max_value=1.0, value=0.7, step=0.01,
            help="Минимальная уверенность модели для фиксации объекта."
        )
        iou_threshold = st.slider(
            "Порог перекрытия рамок (IoU)",
            min_value=0.0, max_value=1.0, value=0.7, step=0.01,
            help="Порог для отсечения дублирующихся рамок."
        )
        st.markdown("---")
        st.info("Модель обучена на датасете ветрогенераторов")

    # === ЗАГРУЗКА МОДЕЛИ ===
    weights_path = find_weights()
    if weights_path is None:
        st.error("❌ Файл весов не найден! Убедитесь, что `models/yolo11m_wind.pt` существует.")
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
                except Exception:
                    st.error("Не удалось загрузить изображение. Проверьте URL.")

    # === ОБРАБОТКА КАЖДОГО ИЗОБРАЖЕНИЯ ===
    if images_to_process:
        for pil_img, name in images_to_process:
            st.markdown("---")
            st.subheader(f"Источник: {name}")

            img_np = np.array(pil_img)
            results = model(img_np, conf=conf_threshold, iou=iou_threshold, verbose=False)
            annotated = results[0].plot()  # BGR
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_img, caption="Оригинал", use_container_width=True)
            with col2:
                st.image(annotated_rgb, caption=f"Результат (Conf: {conf_threshold}, IoU: {iou_threshold})", use_container_width=True)

            boxes = results[0].boxes
            if boxes is not None:
                st.write(f"Обнаружено объектов: {len(boxes)}")
            else:
                st.write("Объектов не обнаружено")

    # === БЛОК МЕТРИК И ГРАФИКОВ ===
    with st.expander("📊 Информация о модели, качестве и процессе обучения (Ветрогенераторы)"):
        st.markdown("""
        **Метрики обучения YOLOv11m**  
        * **Число эпох обучения:** 100  
        * **Объем выборки:** см. data.yaml (train/valid)  
        """)

        metric_tabs = st.tabs(["📈 Результаты обучения (Loss/mAP)", "🎯 PR Кривая", "🧩 Матрица ошибок"])

        with metric_tabs[0]:
            st.markdown("**Графики функций потерь и метрик качества**")
            img_path = find_image_path("wind_results.png")
            if img_path:
                st.image(img_path, caption="YOLOv11 Training Results", use_container_width=True)
            else:
                st.info("Файл `images/wind_results.png` ещё не добавлен в проект.")
            st.markdown("""
            **Анализ графиков обучения:**  
            - **Функции потерь:** box_loss, cls_loss, dfl_loss плавно снижаются с 0.25 до 0.01, демонстрируя устойчивую сходимость модели без явного переобучения.  
            - **Метрики:** По данным Precision-Recall кривой, итоговые значения:  
              - mAP@0.5: **0.826**  
              - AP для класса `cable tower`: **0.778**  
              - AP для класса `turbine`: **0.874**  
              - Precision и Recall на валидации стабилизировались на приемлемом уровне.
            """)

        with metric_tabs[1]:
            st.markdown("**Кривая Точности-Полноты (Precision-Recall Curve)**")
            img_path = find_image_path("wind_boxpr_curve.png")
            if img_path:
                st.image(img_path, caption="Precision-Recall Curve", use_container_width=True)
            else:
                st.info("Файл `images/wind_boxpr_curve.png` ещё не добавлен.")
            st.markdown("""
            **Анализ PR-кривой:**  
            - Площадь под кривой (AP) для класса `cable tower` = **0.778**, для `turbine` = **0.874**.  
            - Средний mAP@0.5 по всем классам = **0.826**, что является хорошим показателем для детекции ветрогенераторов.  
            - Кривая показывает высокую точность (precision) при recall до 0.8, затем плавное снижение – модель уверенно обнаруживает большинство объектов.
            """)

        with metric_tabs[2]:
            st.markdown("**Матрица ошибок (Confusion Matrix) финальной эпохи**")
            img_path = find_image_path("wind_confusion_matrix.png")
            if img_path:
                st.image(img_path, caption="Confusion Matrix", use_container_width=True)
            else:
                st.info("Файл `images/wind_confusion_matrix.png` ещё не добавлен.")
            st.markdown("""
            **Анализ матрицы ошибок (нормированной):**  
            - **cable tower:** правильно распознано **15** объектов, **341** ошибочно принято за `turbine`.  
            - **turbine:** правильно распознано **1197** объектов, **117** ошибочно принято за `cable tower`.  
            - **background:** 9 объектов фона ошибочно классифицированы как `cable tower`.  
            - Основная путаница – между классами «кабельная вышка» и «турбина» из-за визуальной схожести. Высокое количество ложных срабатываний для `cable tower` может быть снижено повышением порога уверенности.
            """)

# ----------------------------------------------------------------------
# Никаких вызовов функции на уровне модуля!
# ----------------------------------------------------------------------
