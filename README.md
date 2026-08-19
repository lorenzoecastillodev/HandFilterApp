# 🖐️ HandFilterApp

Aplicación de escritorio en **C# (WPF)** que detecta las manos en tiempo real a través de la webcam y aplica filtros visuales dentro del área que formas con los dedos — sin depender de Python ni librerías externas de visión por computadora en tiempo de ejecución.

## 🎥 Demo

https://github.com/user-attachments/assets/900f8c9f-f123-43a9-9e3c-b2b1b8928a35

## Características

- Detección de hasta **2 manos simultáneas** en tiempo real, con 21 landmarks por mano.
- Gesto de **pulgar tocando el meñique** para cambiar entre 7 filtros distintos.
- El filtro se aplica solo dentro del **"marco de foto"** que formas con ambas manos (pulgar con pulgar, índice con índice).
- 7 filtros visuales: Mapa de Calor, Boceto, Glitch, Bordes, Pop Art, Escala de Grises, Invertido.
- Arquitectura multi-hilo: la cámara, la detección de IA, y la interfaz corren en paralelo sin bloquearse entre sí.
- Inferencia de modelos de IA (ONNX) corriendo 100% nativo en C#, sin depender de Python.

## Tecnologías

- **C# / WPF** — interfaz de escritorio
- **OpenCvSharp4** — procesamiento de imágenes y video
- **ONNX Runtime** — inferencia de los modelos de detección de manos
- **Modelos:** [MediaPipe Hands](https://github.com/opencv/opencv_zoo) (palm detection + hand landmarks), convertidos a ONNX por el equipo de OpenCV Zoo

## Cómo funciona

1. **Captura de cámara** — un hilo dedicado lee frames de la webcam en tiempo real (1280x720).
2. **Detección de palma** — un modelo ONNX localiza hasta 2 manos por frame, generando un rectángulo por cada una.
3. **Landmarks** — por cada mano detectada, un segundo modelo calcula los 21 puntos clave (muñeca, nudillos, puntas de dedos).
4. **Gestos** — se mide la distancia entre pulgar y meñique (normalizada al tamaño de la mano) para detectar el gesto de cambio de filtro.
5. **Área de filtro** — se calcula un cuadrilátero entre los índices y pulgares de ambas manos, y el filtro activo se aplica solo dentro de esa máscara.

## Cómo correrlo

1. Clona el repositorio:
```bash
   git clone https://github.com/lorenzoecastillodev/HandFilterApp.git
```
2. Abre `HandFilterApp.sln` en **Visual Studio**.
3. Restaura los paquetes NuGet (se hace automático al compilar, o clic derecho en la solución → *Restaurar paquetes NuGet*).
4. Verifica que los modelos `.onnx` estén en la carpeta `Models/` del proyecto (se incluyen en el repo).
5. Compila y corre con **F5**.

## Gestos

| Gesto | Acción |
|---|---|
| Formar un rectángulo con ambas manos (pulgar-pulgar, índice-índice) | Aplica el filtro activo dentro del área |
| Tocar la punta del pulgar con la punta del meñique | Cambia al siguiente filtro |
