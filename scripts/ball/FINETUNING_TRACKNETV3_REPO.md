# Instructivo — Fine-tuning de TrackNetV3 para detectar una pelota de básquet

## Objetivo

Adaptar el checkpoint preentrenado de TrackNetV3 para que aprenda a localizar una pelota de básquet en tus propios videos.

El flujo será:

```text
TrackNet_best.pt
        ↓
pesos preentrenados
        ↓
videos de básquet etiquetados
        ↓
fine-tuning
        ↓
TrackNet_basketball_best.pt
        ↓
inferencia sobre nuevos videos
```

No vamos a entrenar TrackNet desde cero.

La idea es aprovechar lo que el modelo ya aprendió sobre:

* objetos deportivos pequeños;
* movimiento rápido;
* información temporal;
* generación de heatmaps;

y adaptarlo visualmente a una pelota de básquet.

---

# 1. Estructura del proyecto

Partimos del repositorio oficial de TrackNetV3.

La estructura debería quedar aproximadamente así:

```text
TrackNetV3/
│
├── ckpts/
│   └── TrackNet_best.pt
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── model.py
├── dataset.py
├── preprocess.py
├── train.py
├── predict.py
│
└── ...
```

El archivo importante es:

```text
ckpts/TrackNet_best.pt
```

Este será nuestro punto de partida.

---

# 2. Crear el dataset de básquet

No necesitás bounding boxes para entrenar TrackNet.

Para cada frame necesitamos saber:

```text
¿La pelota es visible?

Si es visible:
    ¿dónde está su centro?
```

Por lo tanto, cada etiqueta contiene:

```text
Frame
Visibility
X
Y
```

---

# 3. Formato de las etiquetas

Crear un CSV por video.

Ejemplo:

```csv
Frame,Visibility,X,Y
0,1,853,421
1,1,861,415
2,1,869,408
3,0,0,0
4,0,0,0
5,1,901,384
6,1,914,375
```

Interpretación:

```text
Frame 0
pelota visible
posición = (853, 421)

Frame 1
pelota visible
posición = (861, 415)

Frame 3
pelota no visible

Frame 5
pelota vuelve a aparecer
posición = (901, 384)
```

Cuando la pelota no sea visible:

```text
Visibility = 0
X = 0
Y = 0
```

---

# 4. Qué punto etiquetar

Etiquetar aproximadamente el **centro de la pelota**.

No hace falta precisión de bounding box.

Ejemplo:

```text
       pelota

       ┌─────┐
       │  ●  │
       └─────┘
          ↑
        (X,Y)
```

TrackNet transformará posteriormente ese punto en un heatmap.

---

# 5. Qué frames incluir

No armes un dataset compuesto solamente por situaciones fáciles.

Necesitamos especialmente ejemplos difíciles:

```text
pelota grande
pelota pequeña
pelota muy pequeña
motion blur
pases rápidos
rebotes
pelota cerca del aro
pelota cerca de jugadores
pelota parcialmente tapada
pelota contra el parquet
pelota contra fondos claros
pelota contra fondos oscuros
pelota fuera de cámara
```

Esto es particularmente importante porque queremos solucionar los casos en los que el modelo actual falla.

---

# 6. Cantidad inicial de datos

No hace falta comenzar con un dataset enorme.

Para el primer experimento apuntaría aproximadamente a:

```text
10-20 clips
```

de:

```text
20-60 segundos
```

intentando obtener aproximadamente:

```text
3.000-8.000 frames útiles
```

No es necesario llegar a ese número exacto.

Es preferible tener:

```text
3.000 frames diversos
```

que:

```text
20.000 frames prácticamente iguales
```

---

# 7. No extraer únicamente frames independientes

TrackNet utiliza información temporal.

Por lo tanto, necesitamos conservar secuencias:

```text
frame 100
frame 101
frame 102
frame 103
frame 104
...
```

No seleccionar únicamente:

```text
frame 100
frame 500
frame 900
```

Queremos que el modelo pueda aprender cómo se mueve la pelota entre frames consecutivos.

---

# 8. Separar Train / Validation / Test

Separar los datos **por video**, no por frame.

Ejemplo:

```text
15 videos

10 → train
2  → validation
3  → test
```

Nunca hacer:

```text
video_01 frames 0-500 → train

video_01 frames 501-700 → validation
```

porque los frames son demasiado parecidos y generaríamos leakage.

La estructura conceptual será:

```text
data/
├── train/
│   ├── game_01/
│   ├── game_02/
│   ├── game_03/
│   └── ...
│
├── val/
│   ├── game_11/
│   └── game_12/
│
└── test/
    ├── game_13/
    ├── game_14/
    └── game_15/
```

Idealmente, el test debería contener partidos o condiciones que el modelo nunca vio durante entrenamiento.

---

# 9. Estructura de cada partido

TrackNetV3 espera una estructura similar a:

```text
game_01/
├── video/
│   └── clip_01.mp4
│
└── csv/
    └── clip_01_ball.csv
```

Ejemplo completo:

```text
data/
└── train/
    ├── game_01/
    │   ├── video/
    │   │   └── clip_01.mp4
    │   └── csv/
    │       └── clip_01_ball.csv
    │
    └── game_02/
        ├── video/
        │   └── clip_02.mp4
        └── csv/
            └── clip_02_ball.csv
```

---

# 10. Preprocesar los videos

Una vez preparados los videos y CSV, ejecutar:

```bash
python preprocess.py
```

El script extraerá los frames que necesita TrackNet.

Después aparecerán estructuras similares a:

```text
game_01/
├── video/
├── csv/
└── frame/
    └── clip_01/
        ├── 0.png
        ├── 1.png
        ├── 2.png
        ├── 3.png
        └── ...
```

---

# 11. Verificar el dataset antes de entrenar

Este paso es obligatorio.

Antes del entrenamiento, tomar varios ejemplos aleatorios y dibujar:

```text
(X,Y)
```

sobre el frame.

Queremos verificar visualmente:

```text
            ● ← etiqueta
          (ball)
```

y no:

```text
● etiqueta


                     (ball)
```

Un dataset con coordenadas incorrectas puede hacer que el entrenamiento parezca funcionar mientras el modelo aprende objetivos equivocados.

---

# 12. Verificar resolución y coordenadas

Comprobar que:

```text
0 <= X < width
0 <= Y < height
```

para todas las observaciones visibles.

También verificar que el CSV utiliza las coordenadas correspondientes a la resolución original del video.

---

# 13. Inspeccionar el checkpoint

Antes de modificar `train.py`, verificar cómo está guardado:

```python
import torch

checkpoint = torch.load(
    "ckpts/TrackNet_best.pt",
    map_location="cpu"
)

print(type(checkpoint))

if isinstance(checkpoint, dict):
    print(checkpoint.keys())
```

Podríamos obtener:

```text
dict_keys([
    'epoch',
    'model',
    'optimizer',
    'param_dict'
])
```

En ese caso los pesos están probablemente en:

```python
checkpoint["model"]
```

Si el archivo directamente contiene un state dict:

```python
checkpoint
```

serán los pesos.

No asumir la estructura sin verificarla.

---

# 14. Modificar `train.py`

Queremos agregar soporte explícito para fine-tuning.

Agregar al parser:

```python
parser.add_argument(
    "--pretrained_tracknet",
    type=str,
    default=None,
    help="Checkpoint preentrenado para fine-tuning",
)
```

---

# 15. Cargar los pesos

Después de crear el modelo TrackNet:

```python
model = TrackNet(...)
```

agregar:

```python
if args.pretrained_tracknet is not None:

    checkpoint = torch.load(
        args.pretrained_tracknet,
        map_location="cpu"
    )

    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print(
        "Pesos preentrenados cargados desde:",
        args.pretrained_tracknet
    )
```

---

# 16. No reutilizar el optimizer anterior

Queremos conservar:

```text
pesos aprendidos
```

pero no:

```text
estado anterior del optimizer
```

Por lo tanto, crear un optimizer nuevo.

Ejemplo:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
)
```

No cargar:

```python
checkpoint["optimizer"]
```

del entrenamiento original.

---

# 17. Agregar learning rate configurable

Si todavía no existe:

```python
parser.add_argument(
    "--lr",
    type=float,
    default=1e-5,
)
```

Para fine-tuning comenzaremos con:

```text
1e-5
```

es decir:

```text
0.00001
```

---

# 18. Por qué utilizar learning rate bajo

El modelo ya contiene conocimiento útil.

No queremos:

```text
TrackNet pretrained
       ↓
learning rate enorme
       ↓
destruir representaciones aprendidas
```

Queremos:

```text
TrackNet pretrained
       ↓
pequeños ajustes
       ↓
adaptación a basketball
```

Por eso comenzaremos aproximadamente con:

```text
1e-5
```

y posteriormente:

```text
5e-6
```

---

# 19. Primera etapa de entrenamiento

Idealmente congelar las primeras capas durante unas pocas epochs.

Conceptualmente:

```text
TrackNet

early layers       → congeladas
middle layers      → congeladas
final layers       → entrenables
```

Durante aproximadamente:

```text
3-5 epochs
```

Esto permite adaptar primero las capas más específicas.

---

# 20. Congelar capas

La implementación exacta depende de los nombres definidos en `model.py`.

Conceptualmente:

```python
for param in model.parameters():
    param.requires_grad = True
```

y después congelar el bloque correspondiente:

```python
for param in model.encoder.parameters():
    param.requires_grad = False
```

`model.encoder` es solamente un ejemplo.

Antes de hacerlo inspeccionar:

```python
print(model)
```

o:

```python
for name, _ in model.named_parameters():
    print(name)
```

para identificar qué bloques existen realmente.

---

# 21. Entrenar las capas finales

Ejecutar aproximadamente:

```bash
python train.py \
    --model_name TrackNet \
    --seq_len 8 \
    --epochs 5 \
    --batch_size 4 \
    --lr 1e-5 \
    --bg_mode concat \
    --alpha 0.5 \
    --pretrained_tracknet ckpts/TrackNet_best.pt \
    --save_dir exp_basketball_stage1 \
    --verbose
```

Si la GPU no soporta:

```text
batch_size = 4
```

usar:

```text
batch_size = 2
```

o:

```text
batch_size = 1
```

---

# 22. Segunda etapa

Después de la primera etapa:

```text
descongelar todo el modelo
```

Conceptualmente:

```python
for param in model.parameters():
    param.requires_grad = True
```

Y continuar con:

```text
lr = 5e-6
```

durante aproximadamente:

```text
10-15 epochs
```

---

# 23. Segunda ejecución

Conceptualmente:

```bash
python train.py \
    --model_name TrackNet \
    --seq_len 8 \
    --epochs 15 \
    --batch_size 4 \
    --lr 5e-6 \
    --bg_mode concat \
    --alpha 0.5 \
    --pretrained_tracknet exp_basketball_stage1/TrackNet_best.pt \
    --save_dir exp_basketball_stage2 \
    --verbose
```

La ruta exacta del checkpoint dependerá de cómo `train.py` guarde los experimentos.

---

# 24. Resultado

Al finalizar deberíamos obtener algo como:

```text
exp_basketball_stage2/
├── TrackNet_best.pt
├── TrackNet_cur.pt
└── ...
```

Renombrar o copiar el mejor checkpoint como:

```text
TrackNet_basketball_best.pt
```

---

# 25. Probar el modelo

Ejecutar sobre un video que **no haya sido utilizado para entrenar**.

Por ejemplo:

```bash
python predict.py \
    --video_file basketball_test.mp4 \
    --tracknet_file TrackNet_basketball_best.pt \
    --save_dir prediction_basketball \
    --output_video
```

Si `predict.py` exige InpaintNet, mantener el checkpoint original únicamente para esa etapa o adaptar el script para evaluar TrackNet solo.

---

# 26. Qué comparar

No mirar únicamente si el video "se ve mejor".

Comparar:

```text
TrackNet original
vs
TrackNet basketball fine-tuned
```

sobre exactamente los mismos videos.

Medir al menos:

```text
Detection Rate
Error espacial
Frames perdidos
Falsos positivos
```

---

# 27. Detection Rate

Calcular:

```text
frames donde encontró pelota
────────────────────────────
frames donde pelota era visible
```

Por ejemplo:

```text
Original TrackNet:

312 / 1000 = 31.2%

Fine-tuned:

861 / 1000 = 86.1%
```

---

# 28. Error espacial

Para cada predicción:

```python
error = sqrt(
    (pred_x - gt_x)**2
    +
    (pred_y - gt_y)**2
)
```

Medir:

```text
P@10px
P@20px
P@40px
```

Ejemplo:

```text
P@20px = 89%
```

significa que:

```text
89% de las predicciones
estuvieron a menos de 20 px
del centro real de la pelota.
```

---

# 29. No entrenar InpaintNet todavía

En esta etapa queremos solucionar:

```text
¿TrackNet ve la pelota?
```

No:

```text
¿Podemos reconstruir perfectamente
una trayectoria perdida?
```

Por eso primero:

```text
TrackNet fine-tuning
```

Después:

```text
Kalman
interpolation
smoothing
```

Y recién después evaluar si InpaintNet aporta algo.

---

# 30. Qué hacer si sigue fallando

Si después del fine-tuning TrackNet sigue sin detectar la pelota, no asumir inmediatamente que faltan datos.

Revisar primero:

```text
1. tamaño de pelota en píxeles
2. resolución de entrada de TrackNet
3. resizing
4. etiquetas
5. heatmaps objetivo
6. preprocessing
7. motion blur
```

Especialmente:

```text
resolución original
        ↓
resize TrackNet
        ↓
¿cuántos píxeles ocupa la pelota?
```

Si una pelota de:

```text
10 × 10 px
```

termina siendo:

```text
2 × 2 px
```

después del resize, el problema puede ser la resolución de entrada y no el entrenamiento.

---

# 31. No cambiar la resolución sin cuidado

Si se decide aumentar la resolución de TrackNet:

```text
512 × 288
       ↓
1024 × 576
```

por ejemplo, hay que verificar que:

* la arquitectura soporte las dimensiones;
* los heatmaps tengan dimensiones correctas;
* las coordenadas se escalen correctamente;
* la memoria GPU alcance.

Este experimento deberá hacerse separado del fine-tuning inicial.

---

# 32. Data augmentation recomendada

Agregar augmentations que reproduzcan los problemas reales.

### Brillo

```text
más claro
más oscuro
```

### Contraste

Variaciones moderadas.

### Motion blur

Muy importante para pases y tiros.

### Gaussian blur

Moderado.

### Compresión

Simular artefactos de video.

### Escala

Simular pelota más grande/pequeña.

---

# 33. Cuidado con augmentations espaciales

Si se modifica:

```text
posición
escala
crop
flip
```

también deben modificarse correctamente:

```text
X
Y
```

de la etiqueta.

Nunca aplicar una transformación espacial a la imagen dejando las coordenadas originales.

---

# 34. Casos negativos

El dataset debe contener secuencias donde:

```text
Visibility = 0
```

Por ejemplo:

* pelota fuera de cámara;
* pelota completamente ocluida;
* pelota imposible de identificar.

Esto enseña al modelo que no siempre debe producir una pelota.

---

# 35. No abusar de frames negativos

Mantener un balance razonable.

No queremos que:

```text
80% dataset = pelota invisible
```

porque el modelo podría aprender que predecir ausencia es demasiado conveniente.

---

# 36. Validación correcta

Durante entrenamiento observar:

```text
training loss
validation loss
```

Si ocurre:

```text
train loss ↓↓↓

val loss ↓
val loss ↑↑
```

probablemente estamos entrando en overfitting.

En ese caso:

```text
detener entrenamiento
```

y conservar el checkpoint anterior.

---

# 37. Early stopping

Implementar idealmente:

```text
patience = 3-5 epochs
```

Si la validación no mejora durante ese período:

```text
stop
```

Esto es especialmente útil con datasets relativamente pequeños.

---

# 38. Flujo completo recomendado

```text
VIDEOS DE BÁSQUET
       ↓
seleccionar clips diversos
       ↓
etiquetar (Visibility, X, Y)
       ↓
split por video
       ↓
train / val / test
       ↓
preprocess.py
       ↓
verificar etiquetas visualmente
       ↓
TrackNet_best.pt
       ↓
cargar pretrained weights
       ↓
STAGE 1
freeze parcial
LR = 1e-5
3-5 epochs
       ↓
STAGE 2
unfreeze
LR = 5e-6
10-15 epochs
       ↓
early stopping
       ↓
TrackNet_basketball_best.pt
       ↓
TEST SET
       ↓
comparar contra modelo original
```

---

# 39. Criterios para considerar exitoso el fine-tuning

El modelo nuevo deberá:

* detectar significativamente más pelotas que el checkpoint original;
* reducir frames perdidos;
* mantener buena precisión espacial;
* funcionar en videos no vistos;
* detectar pelotas pequeñas;
* soportar motion blur razonablemente;
* no generar grandes cantidades de falsos positivos.

No evaluar el éxito mirando únicamente los videos utilizados para entrenar.

---

# 40. Qué NO implementar todavía

Durante esta tarea no agregar:

```text
Kalman
trajectory gating
RTS smoother
InpaintNet training
detección de posesión
detección de tiros
event detection
```

Primero resolver exclusivamente:

```text
frames consecutivos
       ↓
TrackNet fine-tuned
       ↓
heatmap
       ↓
posición correcta de pelota
```

Una vez que esto tenga buen recall, recién entonces agregar el tracker temporal posterior.

---

# 41. Resultado esperado

Antes:

```text
Frame 100 → pelota
Frame 101 → ?
Frame 102 → ?
Frame 103 → pelota
Frame 104 → ?
Frame 105 → ?
```

Después del fine-tuning:

```text
Frame 100 → ●
Frame 101 → ●
Frame 102 → ●
Frame 103 → ●
Frame 104 → ●
Frame 105 → ●
```

El objetivo de esta tarea es obtener un:

```text
TrackNet_basketball_best.pt
```

capaz de generar heatmaps confiables para pelotas de básquet en videos que no fueron utilizados durante el entrenamiento.

Ese checkpoint será posteriormente la entrada del pipeline de tracking temporal con Kalman, gating e interpolación.
