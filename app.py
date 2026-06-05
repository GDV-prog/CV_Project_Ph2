import streamlit as st
import os
import sys
import traceback

# Добавляем пути для импорта
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "pages"))

st.set_page_config(
    page_title="CV Project • DetecronTeam",
    page_icon="🚀",
    layout="wide"
)

# Скрываем автоматические ссылки на страницы (app, face, wind)
st.markdown("""
    <style>
        section[data-testid="stSidebar"] ul {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# ФУНКЦИИ-ЗАГЛУШКИ НА СЛУЧАЙ ОШИБОК ИМПОРТА
# ------------------------------------------------------------------
def face_page_stub():
    st.error("❌ Страница 'Детекция лиц' недоступна.\n\n"
             "**Причина:** не удалось загрузить модуль `face.py` или в нём отсутствует функция `render_face_detection_page`.\n\n"
             "**Решение:** убедитесь, что файл `pages/face.py` существует и содержит определение:\n"
             "```python\n"
             "def render_face_detection_page():\n"
             "    # ваш код\n"
             "```")

def wind_page_stub():
    st.error("❌ Страница 'Детекция ветрогенераторов' недоступна.\n\n"
             "**Причина:** не удалось загрузить модуль `wind.py` или в нём отсутствует функция `render_wind_detection_page`.\n\n"
             "**Решение:** убедитесь, что файл `pages/wind.py` существует и содержит определение:\n"
             "```python\n"
             "def render_wind_detection_page():\n"
             "    # ваш код\n"
             "```")

# ------------------------------------------------------------------
# БЕЗОПАСНЫЙ ИМПОРТ МОДУЛЕЙ
# ------------------------------------------------------------------
render_face_detection_page = face_page_stub
render_wind_detection_page = wind_page_stub

# Импорт face.py
try:
    try:
        import face
    except ImportError:
        import pages.face as face
    if hasattr(face, "render_face_detection_page"):
        render_face_detection_page = face.render_face_detection_page
    else:
        st.warning("Модуль `face.py` загружен, но не содержит функцию `render_face_detection_page`.")
except Exception as e:
    st.warning(f"Не удалось загрузить `face.py`: {type(e).__name__}: {e}")

# Импорт wind.py
try:
    try:
        import wind
    except ImportError:
        import pages.wind as wind
    if hasattr(wind, "render_wind_detection_page"):
        render_wind_detection_page = wind.render_wind_detection_page
    else:
        st.warning("Модуль `wind.py` загружен, но не содержит функцию `render_wind_detection_page`.")
except Exception as e:
    st.warning(f"Не удалось загрузить `wind.py`: {type(e).__name__}: {e}")

# ------------------------------------------------------------------
# БОКОВАЯ ПАНЕЛЬ НАВИГАЦИИ
# ------------------------------------------------------------------
with st.sidebar:
    st.title("🧩 Навигация")
    page = st.radio(
        "Выберите интересующий модуль:",
        [
            "🏠 Главная страница",
            "🎭 Детекция и маскирование лиц",
            "💨 Детекция ветрогенераторов",
            "✳️ Семантическая сегментация аэрокосмических снимков"
        ]
    )
    st.markdown("---")
    st.markdown("### 👥 Наша команда:")
    st.caption("Разработчик Роман: Детекция и маскирование лиц")
    st.caption("Разработчик Дорджи: Детекция ветрогенераторов")
    st.caption("Совместно: Семантическая сегментация аэрокосмических снимков")

# ------------------------------------------------------------------
# РЕНДЕРИНГ ВЫБРАННОЙ СТРАНИЦЫ
# ------------------------------------------------------------------
if page == "🏠 Главная страница":
    st.title("🚀 Командный проект по Компьютерному Зрению (CV)")
    st.subheader("Веб-сервис искусственного интеллекта DetecronTeam")

    if os.path.exists("images/logo.jpg"):
        st.image("images/logo.jpg", width=300, caption="DetecronTeam Logo")

    st.markdown("""
    ### 👋 Добро пожаловать!
    Данное multipage-приложение представляет собой комплексный аналитический инструмент, объединяющий современные архитектуры глубокого обучения для решения различных прикладных задач Computer Vision.
    
    #### 🛠️ Доступные модули и технологии:
    1. **Детекция и маскирование лиц:** Автоматическое обнаружение лиц на базе обученной модели **YOLOv11n** с последующим конфиденциальным размытием Гаусса (OpenCV).
    2. **Детекция ветрогенераторов:** Поиск промышленных объектов на аэрофотоснимках с использованием архитектуры **YOLOv11n**.
    3. **Семантическая сегментация космоснимков:** (Бонусный модуль) Попиксельная классификация аэрокосмических снимков с использованием **U-Net**.
    
    *Используйте боковое меню слева, чтобы переключиться на нужный модуль приложения.*
    """)

elif page == "🎭 Детекция и маскирование лиц":
    render_face_detection_page()

elif page == "💨 Детекция ветрогенераторов":
    render_wind_detection_page()

 
elif page == "✳️ Сегментация космоснимков":
    # Вызываем готовую страницу сегментации лесов со всеми вкладками,
    # мультизагрузкой, URL-детекцией и графиками метрик обучения.
    run_forest_segmentation_page()
