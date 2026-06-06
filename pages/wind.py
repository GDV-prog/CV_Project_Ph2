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
        PROJECT_ROOT / "models" / "yolo11m_best_wind.pt",
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
        st.error("❌ Файл весов не найден! Убедитесь, что `models/yolo11m_best_wind.pt` существует.")
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
            annotated_rgb = results[0].plot()  # уже RGB

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
        * **Число эпох обучения:** 50  
        * **Объем выборки:** см. data.yaml (train/valid)  
        """)

        metric_tabs = st.tabs(["📈 Результаты обучения (Loss/mAP)", "🎯 PR Кривая", "🧩 Матрица ошибок"])

        with metric_tabs[0]:
            st.markdown("**Графики функций потерь и метрик качества**")
            img_path = find_image_path("windF_results.png")
            if img_path:
                st.image(img_path, caption="YOLOv11 Training Results", use_container_width=True)
            else:
                st.info("Файл `images/windF_results.png` ещё не добавлен в проект.")
            st.markdown("""
            **Анализ графиков обучения (финальная модель YOLOv11m):**  
            - **Функции потерь:** `box_loss`, `cls_loss`, `dfl_loss` быстро снижаются с ~1.65 до ~0.50–0.52 и далее стабилизируются, демонстрируя отличную сходимость без переобучения.  
            - **Метрики (по итогам 80 эпох):**  
              - `Precision` = **0.96**  
              - `Recall` = **0.98**  
              - `mAP50` = **0.96**  
              - `mAP50-95` = **0.78**  
            - **Итоговый mAP@0.5 по Precision‑Recall кривой = 0.968**.  
            - Модель достигла очень высокой точности и полноты, пригодна для промышленного применения.
            """)

        with metric_tabs[1]:
            st.markdown("**Кривая Точности-Полноты (Precision-Recall Curve)**")
            img_path = find_image_path("windF_boxpr_curve.png")
            if img_path:
                st.image(img_path, caption="Precision-Recall Curve", use_container_width=True)
            else:
                st.info("Файл `images/windF_boxpr_curve.png` ещё не добавлен.")
            st.markdown("""
            **Анализ Precision‑Recall кривой (новые результаты):**  
            - **Средний mAP@0.5 = 0.968** – выдающийся результат для детекции ветрогенераторов.  
            - **AP для класса `cable tower` = 0.985** (почти идеально).  
            - **AP для класса `turbine` = 0.951** (также очень высоко).  
            - Кривая практически лежит в правом верхнем углу: при recall до 0.9 точность сохраняется около 0.95–1.0.  
            - Модель уверенно обнаруживает оба класса с минимальным количеством ложных срабатываний.
            """)

        with metric_tabs[2]:
            st.markdown("**Матрица ошибок (Confusion Matrix) финальной эпохи**")
            img_path = find_image_path("windF_confusion_matrix.png")
            if img_path:
                st.image(img_path, caption="Confusion Matrix", use_container_width=True)
            else:
                st.info("Файл `images/windF_confusion_matrix.png` ещё не добавлен.")
            st.markdown("""
            **Анализ матрицы ошибок (Confusion Matrix) финальной эпохи:**  
            - **Класс `cable tower`:**  
              - Верно распознано: **160** объектов.  
              - Ошибочно принято за `turbine`: **43** (ложные срабатывания).  
              - Пропущено (фон или другое): **6** (из строки background).  
            - **Класс `turbine`:**  
              - Верно распознано: **949** объектов.  
              - Ошибочно принято за `cable tower`: **6** (ложные срабатывания).  
              - Пропущено: **0**.  
            - **Фон:** 6 объектов фона ошибочно классифицированы как `cable tower`.  
            - **Вывод:** минимальная путаница между классами (43 и 6), высокое качество детекции. Повышение порога уверенности может убрать оставшиеся ложные срабатывания на фоне.
            """)

# ----------------------------------------------------------------------
# Никаких вызовов функции на уровне модуля!
# ----------------------------------------------------------------------
