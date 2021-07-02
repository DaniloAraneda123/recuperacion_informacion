# Proyecto de Recuperacion de la informacion

## Pre-requisitos
* Tener corriendo elasticsearch
* tensorflow 2.5.0
* elasticsearch 7.12.1
* tensorflow-hub 0.12.0

## Crear indices y poblar base de datos
Ejecute el main.py, el cual creara un indice llamado posts e indexara 20000 documentos de stackoverflow con su representacion vectorial
```
python main.py
```

## Correr la API y ejecutar HTML 
```
python app.py
```