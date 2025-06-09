# RiskRadarSystem_Threaded.py

import cv2
import torch
import numpy as np
import time
import json
import os
from datetime import datetime
from ultralytics import YOLO
from collections import Counter
import matplotlib.pyplot as plt
from threading import Thread
from queue import Queue


class VideoStream:
    """
    Clase para leer frames de un video en un hilo dedicado para evitar
    bloqueos de I/O y asegurar que siempre se procese el frame más reciente.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de video: {src}")

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

        # Obtener propiedades del video
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                self.stop()
                return

    def read(self):
        return self.frame

    def more(self):
        return not self.stopped

    def stop(self):
        self.stopped = True
        self.stream.release()


class InferenceEngine:
    """
    Clase para ejecutar los modelos de inferencia (YOLO, MiDaS) en un
    hilo dedicado para no bloquear el hilo principal.
    """

    def __init__(self, config):
        self.config = config
        self.stopped = False
        self.CONFIDENCE_THRESHOLD = self.config['YOLO_CONFIDENCE_THRESHOLD']

        # Resultados que el hilo principal leerá
        self.annotated_frame = None
        self.risk_level = "Bajo"
        self.risk_color = (0, 255, 0)
        self.total_heat = 0.0
        self.detections = []
        self.processing_fps = 0

        # Componentes del motor
        self._load_models()
        self._setup_risk_components(config['INFERENCE_RESOLUTION'][1], config['INFERENCE_RESOLUTION'][0])

        # Atributos para optimización
        self.last_depth_map = None
        self.frame_count = 0

    def _load_models(self):
        print("Cargando modelos optimizados...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
        self.model_vehicles = YOLO(config['MODEL_PATH_VEHICLES'])
        self.model_coco = YOLO('yolov8n.pt')
        model_type = "MiDaS_small"
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.midas.to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = midas_transforms.small_transform
        print("Modelos cargados exitosamente.")

    def _setup_risk_components(self, frame_height, frame_width):
        self.frame_height, self.frame_width = frame_height, frame_width
        heatmap_h = int(self.frame_height * self.config['HEATMAP_RESOLUTION_FACTOR'])
        heatmap_w = int(self.frame_width * self.config['HEATMAP_RESOLUTION_FACTOR'])
        self.risk_heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)

        self.cone_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        bottom_y = int(self.frame_height * self.config['CONE_BOTTOM_Y_FACTOR'])
        top_y = 0
        top_width = self.frame_width * self.config['CONE_TOP_WIDTH_FACTOR']
        p1 = (int(self.frame_width / 2 - top_width / 2), top_y)
        p2 = (int(self.frame_width / 2 + top_width / 2), top_y)
        p3_cone = (int(self.frame_width / 2 + 20), bottom_y)
        p4_cone = (int(self.frame_width / 2 - 20), bottom_y)
        cone_points = np.array([p1, p2, p3_cone, p4_cone], np.int32)
        cv2.fillPoly(self.cone_mask, [cone_points], 255)
        self.cone_mask_low_res = cv2.resize(self.cone_mask, (heatmap_w, heatmap_h), interpolation=cv2.INTER_NEAREST) > 0

    def start(self, video_stream):
        self.video_stream = video_stream
        Thread(target=self.run, args=(), daemon=True).start()
        return self

    def run(self):
        start_time = time.time()
        num_frames_processed = 0

        while not self.stopped:
            if not self.video_stream.more():
                self.stop()
                break

            original_frame = self.video_stream.read()
            if original_frame is None:
                continue

            frame = cv2.resize(original_frame, (self.frame_width, self.frame_height))

            if self.frame_count % self.config['MIDAS_FRAME_SKIP'] == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    input_tensor = self.transform(img_rgb).to(self.device)
                    prediction = self.midas(input_tensor)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=frame.shape[:2],
                        mode="bicubic", align_corners=False
                    ).squeeze()
                self.last_depth_map = prediction.cpu().numpy()

            if self.last_depth_map is None:
                self.frame_count += 1
                continue

            self._process_detections_and_update_heatmap(frame, self.last_depth_map)

            self.total_heat = np.sum(self.risk_heatmap[self.cone_mask_low_res])
            if self.total_heat > self.config['HEAT_THRESHOLD_HIGH']:
                self.risk_level, self.risk_color = "Alto", (0, 0, 255)
            elif self.total_heat > self.config['HEAT_THRESHOLD_MEDIUM']:
                self.risk_level, self.risk_color = "Medio", (0, 165, 255)
            else:
                self.risk_level, self.risk_color = "Bajo", (0, 255, 0)

            self.annotated_frame = self._visualize_frame(original_frame, self.risk_level, self.risk_color,
                                                         self.total_heat)

            self.frame_count += 1
            num_frames_processed += 1
            elapsed_time = time.time() - start_time
            self.processing_fps = num_frames_processed / elapsed_time if elapsed_time > 0 else 0

    def _process_detections_and_update_heatmap(self, frame, depth_map):
        self.risk_heatmap *= self.config['HEATMAP_DECAY_RATE']

        # --- CORREGIDO: Estructura de datos para detecciones ---
        # En lugar de mezclar las cajas, guardamos la caja y su mapa de nombres juntos.
        detections_with_models = []

        results_v = self.model_vehicles.predict(source=frame, conf=self.CONFIDENCE_THRESHOLD, verbose=False)
        if results_v[0].boxes:
            for box in results_v[0].boxes:
                detections_with_models.append({'box': box, 'names': self.model_vehicles.names})

        results_c = self.model_coco.predict(source=frame, conf=self.CONFIDENCE_THRESHOLD,
                                            classes=self.config['COCO_CLASSES_TO_SEEK_IDS'], verbose=False)
        if results_c[0].boxes:
            for box in results_c[0].boxes:
                detections_with_models.append({'box': box, 'names': self.model_coco.names})

        # self.detections ahora contendrá diccionarios para la visualización
        self.detections = detections_with_models

        # --- CORREGIDO: Bucle de procesamiento ---
        for det_info in self.detections:
            box_data = det_info['box']
            names_map = det_info['names']

            x1, y1, x2, y2 = map(int, box_data.xyxy[0])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            if 0 <= cy < self.frame_height and 0 <= cx < self.frame_width and self.cone_mask[cy, cx] == 255:
                class_id = int(box_data.cls[0])
                class_name = names_map[class_id]

                depth_roi = depth_map[y1:y2, x1:x2]
                if depth_roi.size > 0:
                    median_depth_value = np.median(depth_roi)
                    base_heat = self.config['HEAT_INTENSITY_FACTORS'].get(class_name, 0.3)
                    depth_factor = np.clip(median_depth_value / 50.0, 0.1, 2.0)
                    heat_to_add = base_heat * depth_factor
                    hm_cx = int(cx * self.config['HEATMAP_RESOLUTION_FACTOR'])
                    hm_cy = int(cy * self.config['HEATMAP_RESOLUTION_FACTOR'])
                    self._add_heat(hm_cx, hm_cy, 5, heat_to_add)

    def _add_heat(self, center_x, center_y, radius, value):
        h, w = self.risk_heatmap.shape
        y, x = np.ogrid[:h, :w]
        dist_sq_from_center = (x - center_x) ** 2 + (y - center_y) ** 2
        mask = dist_sq_from_center <= radius ** 2
        self.risk_heatmap[mask] += value

    def _visualize_frame(self, original_frame, risk_level, risk_color, total_heat):
        vis_frame = original_frame.copy()
        orig_h, orig_w, _ = original_frame.shape

        heatmap_upscaled = cv2.resize(self.risk_heatmap, (self.frame_width, self.frame_height))
        heatmap_full_res = cv2.resize(heatmap_upscaled, (orig_w, orig_h))

        heatmap_normalized = cv2.normalize(heatmap_full_res, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

        cone_mask_full_res = cv2.resize(self.cone_mask, (orig_w, orig_h)) > 0
        masked_heatmap = cv2.bitwise_and(heatmap_colored, heatmap_colored, mask=cone_mask_full_res.astype(np.uint8))
        vis_frame = cv2.addWeighted(vis_frame, 0.7, masked_heatmap, 0.5, 0)

        scale_x = orig_w / self.frame_width
        scale_y = orig_h / self.frame_height

        # --- CORREGIDO: Bucle de visualización ---
        for det_info in self.detections:
            box_data = det_info['box']
            x1, y1, x2, y2 = map(int, box_data.xyxy[0])
            x1_o, y1_o = int(x1 * scale_x), int(y1 * scale_y)
            x2_o, y2_o = int(x2 * scale_x), int(y2 * scale_y)

            cv2.rectangle(vis_frame, (x1_o, y1_o), (x2_o, y2_o), (200, 200, 0), 1)

            class_id = int(box_data.cls[0])
            class_name = det_info['names'][class_id]
            conf = float(box_data.conf[0])
            label = f"{class_name} {conf:.2f}"
            cv2.putText(vis_frame, label, (x1_o, y1_o - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.rectangle(vis_frame, (0, 0), (orig_w, 40), (0, 0, 0), -1)
        cv2.putText(vis_frame, f"NIVEL DE RIESGO: {risk_level.upper()}", (10, 28), cv2.FONT_HERSHEY_DUPLEX, 1,
                    risk_color, 2)
        cv2.putText(vis_frame, f"Motor FPS: {self.processing_fps:.2f}", (orig_w - 250, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1)
        return vis_frame

    def stop(self):
        self.stopped = True


def main():
    global config
    config = {
        'MODEL_PATH_VEHICLES': 'F:/Documents/PycharmProjects/RiskRadarThreading/best.pt',
        'VIDEO_INPUT_PATH': 'F:/Documents/PycharmProjects/RiskRadarThreading/GH012372_no_audio.mp4',
        'OUTPUT_DIR': 'results_risk_radar_balanced', # Nuevo directorio de salida para esta prueba
        'YOLO_CONFIDENCE_THRESHOLD': 0.40,
        'COCO_CLASSES_TO_SEEK_IDS': [0, 1, 16],  # 0: person, 1: bicycle, 16: dog

        # --- AJUSTES DE BALANCE: ¡PRUEBA CON ESTOS VALORES! ---

        # Perilla 1: Aumentamos un poco la resolución para detectar más objetos
        'INFERENCE_RESOLUTION': [854, 480], # Calidad 480p, un buen compromiso.

        # Perilla 2: Actualizamos la profundidad más a menudo para alertas más estables
        'MIDAS_FRAME_SKIP': 3,  # Ejecutar MiDaS 1 de cada 3 frames.

        # Perilla 3: Reducimos los umbrales para compensar la menor precisión de MiDaS_small
        'HEAT_THRESHOLD_MEDIUM': 8.0,   # ANTES: 15.0
        'HEAT_THRESHOLD_HIGH': 15.0,  # ANTES: 30.0

        # -----------------------------------------------------------------

        'CONE_BOTTOM_Y_FACTOR': 0.95,
        'CONE_TOP_WIDTH_FACTOR': 0.8,
        'HEATMAP_RESOLUTION_FACTOR': 0.25,
        'HEATMAP_DECAY_RATE': 0.92,
        'HEAT_INTENSITY_FACTORS': {
            'car': 0.8, 'threewheel': 0.7, 'bus': 1.0, 'truck': 1.0,
            'motorbike': 0.6, 'van': 0.9, 'person': 0.4, 'bicycle': 0.3, 'dog': 0.5
        },
    }

    # El resto de la función main es idéntica
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)

    print("Iniciando hilos con configuración BALANCEADA...")
    vs = VideoStream(config['VIDEO_INPUT_PATH']).start()
    engine = InferenceEngine(config).start(vs)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(
        os.path.join(config['OUTPUT_DIR'], 'output_balanced.mp4'),
        fourcc, vs.fps, (vs.width, vs.height)
    )

    print("Iniciando bucle principal (visualización)...")
    time.sleep(2.0)

    while vs.more() and not engine.stopped:
        annotated_frame = engine.annotated_frame
        if annotated_frame is not None:
            cv2.imshow("Risk Radar", annotated_frame)
            out_video.write(annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    print("Limpiando...")
    vs.stop()
    engine.stop()
    out_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
