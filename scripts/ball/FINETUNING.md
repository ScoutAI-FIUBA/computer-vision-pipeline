# Fine-tuning de TrackNet para tracking de pelota de básquet

Esta guía explica el camino más corto y útil para hacer fine-tuning de `TrackNet_best.pt` sobre videos de básquet.

El objetivo es simple: lograr que TrackNet produzca un heatmap útil sobre la pelota **antes** de agregar Kalman, gating o interpolación.

## Camino rápido

1. Preparar entre 3 y 5 clips cortos de básquet con situaciones variadas.
2. Etiquetar cada frame con `Frame,Visibility,X,Y`.
3. Convertir esos clips al formato de dataset esperado por TrackNetV3.
4. Hacer fine-tuning a partir de `TrackNet_best.pt`, no desde cero.
5. Evaluar heatmaps y error espacial sobre videos reservados para validación.

## Qué entrenar

Entrená solo el detector visual temporal.

No se entrena:

- Kalman filter
- temporal gating
- interpolación
- smoothing
- state machine

La parte aprendida es solamente:

```text
frames consecutivos -> heatmap de ubicación probable de la pelota
```

## Qué etiquetar

Para cada frame:

- `Frame`: índice del frame
- `Visibility`: `1` cuando la pelota es visible, `0` cuando no lo es
- `X`: coordenada X del centro de la pelota en la imagen
- `Y`: coordenada Y del centro de la pelota en la imagen

Ejemplo:

```csv
Frame,Visibility,X,Y
0,1,850,410
1,1,857,404
2,0,0,0
3,1,881,391
```

Usá `0,0` cuando `Visibility=0` para mantener compatibilidad con el estilo de CSV de TrackNet.

## Reglas del dataset

Arrancá con un dataset chico pero variado.

Incluí clips con:

- pelota pequeña
- motion blur
- pases
- tiros
- rebotes
- oclusión parcial
- pelota saliendo de cuadro

El split debe hacerse por **video**, no mezclando frames al azar.

Split inicial recomendado:

- train: 3 clips
- val: 1 clip
- test: 1 clip

## Estructura de carpetas

Intentá respetar lo más posible la estructura de TrackNetV3.

```text
tracknet_basketball_data/
├── train/
│   └── match1/
│       ├── csv/
│       │   └── clip_01_ball.csv
│       ├── frame/
│       │   └── clip_01/
│       │       ├── 0.png
│       │       ├── 1.png
│       │       └── ...
│       └── video/
│           └── clip_01.mp4
├── val/
│   └── match1/
└── test/
    └── match1/
```

No necesitás muchas matches para el primer intento. Necesitás una estructura limpia.

## Flujo de etiquetado

### Opción A: etiquetado manual

1. Extraer frames de cada clip.
2. Abrir los frames en una herramienta de etiquetado o script simple.
3. Marcar con click el centro de la pelota cuando sea visible.
4. Marcar como no visible cuando la pelota esté completamente perdida.
5. Exportar un CSV por clip.

### Opción B: bootstrap con tus experimentos actuales

Si tu detector experimental encuentra la pelota algunas veces:

1. exportar puntos tentativos
2. revisarlos manualmente
3. corregir los frames incorrectos
4. guardar el CSV corregido

Esto suele ser más rápido que etiquetar todo desde cero.

## Notas de preprocessing

No inventes preprocessing nuevo al principio.

Arrancá respetando las suposiciones originales de TrackNetV3:

- input secuencial
- ventana temporal esperada por el checkpoint
- resize original
- normalización original
- generación original del heatmap target

Si cambiás preprocessing demasiado temprano, después no vas a saber si falla el dataset o el pipeline.

## Estrategia de fine-tuning

Arrancá desde el checkpoint pretrained:

```text
TrackNet_best.pt
```

No entrenes desde cero en el primer intento.

Estrategia inicial recomendada:

- inicializar desde `TrackNet_best.pt`
- usar learning rate bajo
- entrenar pocas epochs primero
- inspeccionar temprano los ejemplos de validación

El objetivo del primer entrenamiento no es lograr máxima accuracy.
El objetivo es confirmar si el modelo puede adaptarse al básquet.

## Checklist de entrenamiento

- [ ] clips seleccionados
- [ ] labels exportados como CSV
- [ ] split train/val/test hecho por video
- [ ] extracción de frames completada
- [ ] estructura compatible con TrackNetV3 armada
- [ ] checkpoint pretrained disponible
- [ ] primer fine-tune lanzado
- [ ] heatmaps de validación revisados visualmente

## Evaluación

Medí como mínimo esto:

### 1. Detection rate

```text
frames con localización útil de pelota / frames donde la pelota es visible
```

### 2. Error espacial

```text
sqrt((pred_x - gt_x)^2 + (pred_y - gt_y)^2)
```

### 3. Precision por distancia

- P@10px
- P@20px
- P@40px

### 4. Lost rate

```text
frames sin señal útil / frames donde la pelota es visible
```

## Regla GO / NO-GO

Usá esta regla después del primer fine-tune:

- **GO**: los heatmaps activan de forma consistente cerca de la pelota en los clips de validación
- **NO-GO**: los heatmaps siguen fallando mucho o se pegan a objetos incorrectos

Si da GO:

- seguir mejorando el detector
- después agregar Kalman y lógica temporal

Si da NO-GO:

- revisar labels
- revisar preprocessing
- aumentar diversidad del dataset
- recién después considerar cambios de arquitectura

## Qué no hacer todavía

No metas esto antes de que el detector funcione:

- tuning de Kalman filter
- smoothing de trayectoria
- interpolación
- heurísticas de reacquisition
- benchmark grande

Todo eso es downstream. No arregla un detector ciego.

## Próximo paso práctico en este repo

1. Crear un dataset chico etiquetado de básquet.
2. Agregar un script de conversión en `scripts/ball/` si hace falta.
3. Hacer fine-tuning de `TrackNet_best.pt` en un entorno de entrenamiento aislado.
4. Traer el checkpoint ajustado de vuelta a `scripts/ball/` para probar inferencia.

## Template mínimo de anotación

Usá esto por clip:

```csv
Frame,Visibility,X,Y
0,1,0,0
1,1,0,0
2,0,0,0
```

Reemplazá `X,Y` por el centro real de la pelota cuando sea visible.

## Notas para este proyecto

- Mantené este trabajo como experimental dentro de `scripts/ball/` por ahora.
- Validá señal del detector antes de integrar nada al pipeline principal.
- Conviene más un dataset chico y correcto que uno grande y ruidoso.
