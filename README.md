# Mi librería

Inserta aquí descripción corta del proyecto. Una o dos oraciones.


## Características

- **Velocidad:** Alta velocidad y rendimiento.
- **Velocidad:** Alta velocidad y rendimiento.
- **Velocidad:** Alta velocidad y rendimiento.

## Índice de documentación

- Índice de paginas relevantes en otro lugar

<!-- GETTING STARTED -->
# Instalación

Este paquete usa PyTorch, que necesita una build distinta según tu hardware (CPU o GPU NVIDIA). El extra que elijas (`cpu`, `cuda121` o `cuda118`) ya trae configurado automáticamente el índice correcto — no necesitas pasar `--index` a mano ni tocar ningún archivo.

## 1. ¿Qué CUDA tengo?

Corre en tu terminal:

```bash
nvidia-smi
```

Busca la línea `CUDA Version: X.X` en la esquina superior derecha. Esa es la versión **máxima** que tu driver soporta — puedes usar esa versión o cualquiera menor.

Si el comando no existe o da error, no tienes GPU NVIDIA (o el driver no está instalado): usa `cpu`.

| `nvidia-smi` dice | Usa el extra |
|---|---|
| CUDA Version: 12.x | `cuda126` o `cuda130` |
| No hay `nvidia-smi` / no hay GPU NVIDIA | `cpu` |

## 2. Instalación

### Si vas a usar la librería como dependencia en tu propio proyecto

```bash
# CPU
uv add depth-estimation-models --extra cpu

# CUDA 12.6
uv add depth-estimation-models --extra cuda126

# CUDA 13.0
uv add depth-estimation-models --extra cuda130
```

### Si vas a desarrollar dentro de este repo (lo clonaste)

```bash
git clone https://github.com/MichaelSuarez0/depth-estimation-models
cd depth-estimation-models

# CPU
uv sync --extra cpu

# CUDA 12.1
uv sync --extra cuda121

# CUDA 11.8
uv sync --extra cuda118
```

## 3. Verificar que quedó bien instalado

```python
import torch
print(torch.cuda.is_available())  # True = CUDA activo, False = CPU
```
## Uso
Para que los usuarios utilicen la librería, debes mostrar ejemplos de uso útiles en la práctica.
Puedes dividirlo en subsecciones para diferentes usos.
Esta es probablemente la sección más importante para un usuario nuevo, y la sección de mayor extensión.

### Importar

```python
import mi_libreria
from mi_libreria import funcion
```

### Lectura

```python
import mi_libreria

x = a + b
```

### Transformación

```python
import mi_libreria

x = a + b
```



