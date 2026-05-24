# nops — Issue Tracker

> Generated from full codebase review. All 192 tests currently pass.

---

## 🔴 P0 — Críticos (bloquean funcionalidad o performance)

### P0-1: `radius_graph` Python loops — no funciona en GPU
- **File:** `nops/gno/layers/neighbor_search.py:81-96`
- **Problema:** Triple loop nested en Python puro con `.item()` hace CPU↔GPU sync por cada punto. O(N²) Python iterations — impracticable para meshes >100 nodos.
- **Fix:** Reemplazar con `torch.cdist(pos_flat, pos_flat)` → tensor de distancias → boolean mask `dist <= r` → `torch.nonzero` o `torch.where`. Eliminar todos los `.item()` call.
- **Extra:** Mismo patrón en `knn_graph` (líneas 196-226) pero ya usa `torch.topk`, así que está mejor. Revisar si se puede optimizar igual con `cdist`.

### P0-2: `example_gno.py` usa `max_num_neighbors` en GNO.__init__ — bug runtime
- **File:** `example_gno.py:92`
- **Problema:** `GNO.__init__` no acepta `max_num_neighbors`. El ejemplo va a fallar con `TypeError`.
- **Fix:** Agregar `max_num_neighbors: Optional[int] = None` a `GNO.__init__` y pasarlo a `radius_graph()` en `_build_graph()`.

---

## 🟡 P1 — Significativos (usabilidad, features rotas, bloat)

### P1-1: `nops/__init__.py` vacío — nada importable desde top-level
- **File:** `nops/__init__.py`
- **Problema:** `from nops import FNO` o `from nops import DataLoss` fallan. El usuario no tiene API entry point.
- **Fix:** Exporrer los públicos principales:
  ```python
  # nops/__init__.py
  from nops.fno.models.original import FNO
  from nops.gno.models.original import GNO, GNO2D, GNO3D
  from nops.gno.models.mGNO import mGNO
  from nops.deeponet.models.deeponet import DeepONet, DeepONetCartesianProd

  __all__ = ["FNO", "GNO", "GNO2D", "GNO3D", "mGNO", "DeepONet", "DeepONetCartesianProd"]
  ```

### P1-2: `FourierBlock` hardcodea `factorization="dense"` — Tucker/CP/TT no funcionan
- **File:** `nops/fno/layers/fno_block.py:57`
- **Problema:** `SpectralConvolution(in_channels, out_channels, modes, factorization='dense')` ignora cualquier parámetro externo. Si el usuario configura `SpectralConvolution` con Tucker, no tiene efecto porque `FourierBlock` siempre overridea a `'dense'`.
- **Fix:** Agregar `factorization` y `rank` como parámetros de `__init__` de `FourierBlock` y pasarlos a `SpectralConvolution`.

### P1-3: `lightning` en dependencies sin usarse en el código
- **File:** `pyproject.toml:41`
- **Problema:** `lightning>=2.6.0` no se importa en ninguna parte del código. Es una dependency pesada (~200MB+ transitive).
- **Fix:** Quitar de `dependencies` o, si está en el roadmap, mover a un optional extra `[project.optional-dependencies]` y documentar.

### P1-4: `nops/deeponet/__init__.py` y `nops/gno/__init__.py` sin exports
- **Files:** `nops/deeponet/__init__.py`, `nops/gno/__init__.py`
- **Problema:** Sub-módulos sin exports. `from nops.deeponet.models import DeepONet` funciona, pero no `from nops.deeponet import DeepONet`.
- **Fix:** Agregar `__init__.py` con exports consistentes al estilo de `nops/data/__init__.py` y `nops/losses/__init__.py`.

---

## 🟢 P2 — Menores (robustez, limpieza, mejores prácticas)

### P2-1: Código muerto en `radius_graph` (post-return)
- **File:** `nops/gno/layers/neighbor_search.py:136-144`
- **Problema:** `return edge_index, edge_weights` en línea 136. Las líneas 138-144 son inaccesibles. Código duplicado de return.
- **Fix:** Borrar líneas 138-144.

### P2-2: `SpectralConvolution` reconstruye tensor factorizado en cada forward
- **File:** `nops/fno/layers/spectral_convolution.py:232-252`
- **Problema:** `tl.tucker_to_tensor(...)`, `tl.cp_to_tensor(...)`, `tl.tt_to_tensor(...)` se ejecutan en cada forward pass. Costoso y evitable.
- **Fix:** Reconstruir en `__init__` como `nn.Buffer` y actualizar con `register_buffer`, o usar `torch.func`/checkpoint. Mínimo: cachear en `_last_weights` con hash de parámetros.

### P2-3: `IntegralTransform` lazy-init `first_layer` en forward
- **File:** `nops/gno/layers/integral_transform.py:99-105`
- **Problema:** `self.first_layer` se crea en el primer forward. Esto rompe `state_dict`, tracing (torch.jit, ONNX), y puede causar device mismatch si el modelo se mueve a otro device después del primer forward.
- **Fix:** Requerir `input_dim` en `__init__` (agregar parámetro `pos_dim`) o crear una capa placeholder de dim 1 y reasignar con un método `.setup(pos_dim)`.

### P2-4: `.gitignore` demasiado mínimo
- **File:** `.gitignore`
- **Problema:** Falta `.venv/`, `.ruff_cache/`, `*.egg-info/`, `dist/`, build artefacts.
- **Evidencia:** `notebooks/navier_stokes_v1e-3_N1200_T20.pt` existe — archivos `.pt` de datos pueden quedar en el repo.
- **Fix:**
  ```
  __pycache__/
  *.py[cod]
  *.so
  .venv/
  .ruff_cache/
  *.egg-info/
  dist/
  build/
  *.pt
  .pytest_cache/
  coverage.xml
  .coverage
  ```

### P2-5: Tres MLPs duplicados con APIs diferentes
- **Files:** `nops/fno/layers/mlp.py`, `nops/deeponet/layers/mlp.py`, `nops/gno/layers/mlp.py`
- **Problema:** Cada módulo tiene su propia `MLP` con firma de `__init__` diferente. Code duplication, harder to maintain.
- **Fix:** Unificar en `nops/layers/mlp.py` con una API configurable, o documentar que son diferentes por intención. Low-priority si no molesta.

### P2-6: `pyproject.toml` sin tool config (ruff, pytest, mypy)
- **File:** `pyproject.toml`
- **Problema:** README claims "Code Style: Black" pero no hay dev dependency ni `[tool.ruff]` section. No hay `[tool.pytest.ini_options]`.
- **Fix:** Agregar:
  ```toml
  [dependency-groups]
  dev = [
      "pytest>=8.0",
      "ruff>=0.9",
  ]

  [tool.ruff]
  line-length = 100

  [tool.pytest.ini_options]
  testpaths = ["tests"]
  ```

### P2-7: `test_fno.py` backward sin `model.train()`
- **File:** `tests/test_fno.py:31-34`
- **Problema:** `out.mean().backward()` se llama sin asegurar `model.train()`. Funciona porque `dropout=0.0` en el test, pero si se cambia la config a `dropout>0`, el dropout en eval mode se comporta diferente.
- **Fix:** Agregar `model.train()` antes del backward. O documentar la intencionalidad.

### P2-8: `mGNO` no exportado desde ningún `__init__.py`
- **File:** `nops/gno/models/mGNO.py`
- **Problema:** `mGNO` y `mGNOBlock` existen pero no están en `__init__.py` de ninguno. Solo accesible por import directo de módulo.
- **Fix:** Agregar a `nops/gno/__init__.py` y `nops/gno/models/__init__.py`.

---

## 📋 Checklist de verificación post-fix

- [ ] `uv run pytest` → 192+ tests passing
- [ ] `uv run ruff check nops/ tests/` → limpio (una vez configurado)
- [ ] `python -c "from nops import FNO, GNO, DeepONet"` → funciona
- [ ] `python example_gno.py` → funciona sin error
- [ ] `python example_deeponet.py` → funciona sin error
- [ ] `python -m nops.fno.models.original` → self-test pasa
- [ ] `python -m nops.gno.models.original` → self-test pasa
- [ ] `python -m nops.gno.models.mGNO` → self-test pasa
