import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import requests
import io
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image

# ====================================================================
# 1. АРХИТЕКТУРА МОДЕЛИ
# ====================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x_up1 = self.up1(x4)
        x = self.conv1(torch.cat([x_up1, x3], dim=1))
        
        x_up2 = self.up2(x)
        x = self.conv2(torch.cat([x_up2, x2], dim=1))
        
        x_up3 = self.up3(x)
        x = self.conv3(torch.cat([x_up3, x1], dim=1))
        
        return self.outc(x)

# ====================================================================
# 2. БЕЗОПАСНАЯ ИЗОЛИРОВАННАЯ ЗАГРУЗКА ВЕСОВ МОДЕЛИ
# ====================================================================
@st.cache_resource
def load_unet_model_safe():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1)
    
    possible_paths = ["models/unet_forest.pt", "../models/unet_forest.pt", "unet_forest.pt"]
    weights_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            weights_path = path
            break
            
    if weights_path:
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.to(device)
            model.eval()
            return model, device, f"Успешно загружены веса из: {weights_path}"
        except Exception as e:
            return None, device, f"⚠️ Веса найдены ({weights_path}), но не подошли к структуре: {e}"
    else:
        return None, device, "⚠️ Файл весов 'unet_forest.pt' не найден. Включен демо-режим масок."

# ====================================================================
# 3. ОСНОВНАЯ ФУНКЦИЯ ДЛЯ ВЫЗОВА ИЗ APP.PY
# ====================================================================
def run_forest_segmentation_page():
    st.caption("Автоматическое определение границ лесных массивов с помощью нейросети U-Net")

    model, device, status_message = load_unet_model_safe()
    
    st.sidebar.info(f"Статус нейросети:\n{status_message}")
    st.sidebar.text(f"Вычисления: {device.type.upper()}")

    tab1, tab2 = st.tabs(["📤 Обработка снимков", "📊 Аналитика обучения и Метрики"])

    # --- ВКЛАДКА 1: ОБРАБОТКА ДАННЫХ ---
    with tab1:
        st.subheader("Загрузка данных для анализа")
        
        uploaded_files = st.file_uploader(
            "Выберите один или несколько аэрофотоснимков леса...", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True,
            key="forest_uploader_fixed"
        )
        
        url_input = st.text_input(
            "Или вставьте прямую URL-ссылку на снимок:",
            placeholder="https://example.com",
            key="forest_url_fixed"
        )
        
        images_to_process = []
        if uploaded_files:
            for file in uploaded_files:
                images_to_process.append((file.name, Image.open(file).convert("RGB")))
                
        if url_input.strip():
            try:
                with st.spinner("Загрузка по ссылке..."):
                    response = requests.get(url_input, timeout=10)
                    img = Image.open(io.BytesIO(response.content)).convert("RGB")
                    images_to_process.append(("Изображение по URL", img))
            except Exception as e:
                st.error(f"Ошибка загрузки по ссылке: {e}")

        if images_to_process:
            st.write("### 🎯 Результаты сегментации лесных массивов:")
            for name, img in images_to_process:
                st.markdown(f"**Файл:** {name}")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(img, caption="Оригинальный снимок", use_container_width=True)
                    
                with col2:
                    with st.spinner("Генерация маски..."):
                        if model is not None:
                            img_resized = img.resize((256, 256))
                            img_np = np.array(img_resized, dtype=np.float32) / 255.0
                            img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                pred = model(img_tensor)
                                # 🔥 Исправление инверсии: инвертируем маску, чтобы закрашивать лес, а не фон
                                raw_mask = (torch.sigmoid(pred) > 0.35).squeeze().cpu().numpy()
                                pred_mask = 1.0 - raw_mask
                        else:
                            pred_mask = np.zeros((256, 256))
                            pred_mask[40:220, 30:230] = 1.0
                        
                        forest_pixels = np.sum(pred_mask == 1.0)
                        total_pixels = pred_mask.size
                        forest_percentage = (forest_pixels / total_pixels) * 100
                        
                        fig, ax = plt.subplots(figsize=(6, 6))
                        # Подложка фона — чёрная, маска леса — зелёная
                        ax.imshow(pred_mask, cmap="Greens", vmin=0, vmax=1.1)
                        ax.axis("off")
                        
                        fig.patch.set_facecolor('#0e1117')
                        ax.set_facecolor('#0e1117')
                        
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                        
                        st.metric(label="🌲 Покрытие лесного массива алгоритмом U-Net", value=f"{forest_percentage:.1f} %")
                st.divider()

    # --- ВКЛАДКА 2: АНАЛИТИКА ---
    with tab2:
        st.subheader("📋 Параметры и качество процесса обучения")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Всего эпох", "30")
        col_m2.metric("Выборка (Train)", "4,086")
        col_m3.metric("Выборка (Val)", "1,022")
        
        st.table({
            "Метрика качества": [
                "Dice Coefficient (F1-Score)", "IoU (Intersection over Union)", 
                "Финальный Train Loss", "Финальный Val Loss", "Ранняя остановка"
            ],
            "Значение модели U-Net": ["0.7737", "0.6956", "0.3395", "0.3354", "5 эпох без улучшений"]
        })
        
        epochs_range = list(range(1, 18))
        train_loss_history = [0.4222, 0.4027, 0.3887, 0.3785, 0.3728, 0.3661, 0.3644, 0.3634, 0.3620, 0.3542, 0.3543, 0.3506, 0.3501, 0.3473, 0.3466, 0.3422, 0.3395]
        val_loss_history = [0.3973, 0.3921, 0.4445, 0.3885, 0.3784, 0.3551, 0.3505, 0.3634, 0.3816, 0.3433, 0.3534, 0.52, 0.3480, 0.3476, 0.3670, 0.3445, 0.3354]
        val_iou_history = [0.6639, 0.6692, 0.5776, 0.6410, 0.6676, 0.6776, 0.6859, 0.6718, 0.6889, 0.6930, 0.6956, 0.51, 0.6955, 0.6919, 0.6751, 0.6856, 0.6914]
        
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(x=epochs_range, y=train_loss_history, name='Train Loss', line=dict(color='#4169E1', width=2.5)))
        fig_l.add_trace(go.Scatter(x=epochs_range, y=val_loss_history, name='Val Loss', line=dict(color='#FF4500', width=2.5)))
        fig_l.update_layout(title="Кривые функций потерь (BCE + Dice Loss)", xaxis_title="Эпоха", yaxis_title="Loss", template="plotly_dark")
        st.plotly_chart(fig_l, use_container_width=True)

        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=epochs_range, y=val_iou_history, name='Val IoU (Jaccard)', line=dict(color='#32CD32', width=2.5)))
        fig_m.update_layout(title="Рост точности IoU на валидационной выборке", xaxis_title="Эпоха", yaxis_title="IoU Score", template="plotly_dark")
        st.plotly_chart(fig_m, use_container_width=True)

st.divider()
run_forest_segmentation_page()
