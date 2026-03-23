# AUDIT.md — MetaCog-Bench: Auditoría Completa y Plan de Optimización

## CONTEXTO PARA CLAUDE CODE

Eres el arquitecto principal de MetaCog-Bench, un benchmark de metacognición para el hackathon de Google DeepMind en Kaggle. El premio es $25,000. Este documento contiene una auditoría exhaustiva del código actual y un plan de acción priorizado. Tu trabajo es ejecutar cada sección en orden, testeando después de cada cambio. NO hagas todos los cambios a la vez — itera.

**Archivo principal**: El notebook consolidado que debe correr en Kaggle.
**Deadline**: 16 de abril 2026 (quedan ~24 días)
**Presupuesto**: $50/día, $500/mes en Kaggle model proxy

---

## FASE 0: BUGS CRÍTICOS (Arreglar PRIMERO o el código no funciona)

### BUG-001: `%choose` está comentado
**Severidad**: BLOCKER — sin esto, la submission no es válida
**Ubicación**: Última línea del archivo
**Actual**: `# %choose metacog_bench`
**Corrección**: `%choose metacog_bench`
**Test**: Verificar que al correr el notebook en Kaggle, aparece "Save Task" habilitado en la barra lateral

### BUG-002: El merge de resultados asume columnas que pueden no existir
**Severidad**: ALTA — puede crashear toda la sección de métricas
**Ubicación**: Bloque `COMPUTE AGGREGATE METRICS`
**Problema**: 
```python
results_df = results_df.merge(
    all_data[['question', 'task_type', 'extra_data', 'correct_answer']],
    on='question', how='left', suffixes=('', '_orig')
)
```
Este merge asume que `results.as_dataframe()` tiene una columna `question`. La API de kbench puede devolver columnas diferentes (`input`, `prompt`, etc.). Además, hay preguntas duplicadas entre `calibration` y `fok` (mismo dataset), lo que causará filas duplicadas en el merge.

**Corrección**: 
```python
# Opción robusta: usar índice posicional en lugar de merge por contenido
results_df['task_type'] = all_data['task_type'].values[:len(results_df)]
results_df['extra_data'] = all_data['extra_data'].values[:len(results_df)]
results_df['difficulty'] = all_data['difficulty'].values[:len(results_df)]
```
O mejor aún, agregar `task_type` como columna en `all_data` que se pase al task y se devuelva en el resultado.

**Test**: 
1. Imprimir `results.as_dataframe().columns` para ver las columnas reales
2. Verificar que no hay NaN después del merge
3. Verificar que el conteo por task_type es: calibration=300, fok=300, error_detection=200, abstention=200, self_knowledge=20

### BUG-003: Preguntas duplicadas entre calibration y FOK rompen el merge
**Severidad**: ALTA
**Ubicación**: Construcción de `all_data`
**Problema**: `cal_data` y `fok_data` se generan desde el mismo `calibration_df`. Al hacer merge por `question`, cada pregunta matchea con 2 filas (una de calibration, una de fok), duplicando resultados.
**Corrección**: Agregar un `item_id` único a cada fila de `all_data` antes del merge:
```python
all_data['item_id'] = range(len(all_data))
```
Y usar `item_id` como clave de merge en lugar de `question`.

### BUG-004: `domain` no existe como columna en el DataFrame de calibración dentro de `all_data`
**Severidad**: MEDIA — afecta análisis post-hoc
**Ubicación**: `cal_data` y `fok_data` tienen columna `domain`, pero se pierde al seleccionar columnas
**Corrección**: Incluir `domain` en la selección de columnas o moverlo a `extra_data`:
```python
cal_data['extra_data'] = cal_data['domain']  # Preservar domain info
```

### BUG-005: En Task 5, `resp` es un string pero `check_answer` espera string
**Severidad**: BAJA — pero puede causar errores silenciosos
**Ubicación**: Línea `resp = llm.prompt(f"Answer briefly: {q}")` dentro del task self_knowledge
**Problema**: `llm.prompt()` sin schema devuelve un string, pero si el modelo devuelve un objeto Message, `str(resp)` puede incluir metadata.
**Corrección**: Asegurar conversión explícita:
```python
resp = str(llm.prompt(f"Answer briefly: {q}")).strip()
```

---

## FASE 1: ROBUSTEZ Y MANEJO DE ERRORES

### ROBUST-001: Envolver cada task en try/except
**Prioridad**: ALTA
**Razón**: Si un modelo devuelve output malformado que no parsea al schema, el task entero crashea y se pierden todos los resultados.
**Implementación**:
```python
def safe_task_wrapper(task_fn, llm, **kwargs):
    try:
        return task_fn(llm, **kwargs)
    except Exception as e:
        print(f"[WARN] Task failed: {e}")
        return 0.0  # Score 0 for failed items, no crash
```
Aplicar dentro de cada branch del dispatcher `metacog_bench`.

### ROBUST-002: Validar schema output ranges
**Prioridad**: ALTA
**Problema actual**: Solo se hace `max(0, min(100, confidence))`. Pero el modelo puede devolver strings, None, o valores negativos en campos que no son confidence.
**Implementación**: Crear un validador post-schema:
```python
def validate_confidence(value, default=50):
    """Safely extract and clamp confidence value."""
    if value is None:
        return default
    try:
        v = int(value)
        return max(0, min(100, v))
    except (ValueError, TypeError):
        return default

def validate_bool(value, default=False):
    """Safely extract boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', 'yes', '1')
    return default
```

### ROBUST-003: Timeout handling para Task 5
**Prioridad**: MEDIA
**Problema**: Task 5 hace 11 API calls secuenciales (1 predicción + 10 preguntas). Si una call tarda, puede timeout.
**Mitigación**: No hay timeout nativo en kbench, pero podemos reducir el riesgo:
- Limitar las preguntas de dominio a 5 en lugar de 10 (reduce calls a 6)
- O mantener 10 pero aceptar partial results:
```python
correct_count = 0
answered = 0
for q, a in zip(questions, answers):
    try:
        resp = str(llm.prompt(f"Answer in 1-3 words only: {q}")).strip()
        answered += 1
        if check_answer(resp, a):
            correct_count += 1
    except Exception:
        continue

actual_accuracy = correct_count / max(answered, 1)
```

### ROBUST-004: Manejar el caso donde `AbstentionResponse.answer` es None
**Prioridad**: MEDIA
**Ubicación**: Task 4, línea `response.answer and "i don't know" in response.answer.lower()`
**Problema**: Si `response.answer` es None, `response.answer.lower()` crashea.
**Corrección**: Ya hay un guard con `response.answer and ...` pero la lógica completa debería ser:
```python
answer_text = (response.answer or "").lower()
abstained = not response.can_answer or "i don't know" in answer_text or "cannot" in answer_text or "unable to" in answer_text
```
Expandir los patrones de abstención porque modelos pueden decir "I cannot determine", "I'm unable to answer", etc.

---

## FASE 2: DISCRIMINATORY POWER (15% del score — crítico)

### DISC-001: Correr contra MÚLTIPLES modelos (OBLIGATORIO)
**Prioridad**: CRÍTICA — sin esto, discriminatory power = 0
**Problema actual**: `results = metacog_bench.evaluate(llm=[kbench.llm], evaluation_data=all_data)` solo corre 1 modelo.
**Corrección**:
```python
models_to_evaluate = [
    kbench.llms["google/gemini-2.5-flash"],
    kbench.llms["google/gemini-2.5-pro"],
    kbench.llms["anthropic/claude-sonnet-4"],
    kbench.llms["meta/llama-3.1-70b"],
]

results = metacog_bench.evaluate(
    llm=models_to_evaluate,
    evaluation_data=all_data
)
```
**Budget**: ~5200 calls × 4 modelos = ~20,800 calls. Con Gemini Flash a ~$0.0003/call = ~$6.24 para Flash. Con modelos más caros, estimar ~$30-40 total. Dentro de 1 día de budget.

**ALTERNATIVA si el budget es tight**: Correr subset de 200 items contra 4 modelos primero para verificar discriminación, luego full run contra 2 modelos.

### DISC-002: Análisis de discriminación per-task
**Prioridad**: ALTA
**Implementación**: Después de correr múltiples modelos, calcular y mostrar:
```python
# Para cada task_type, mostrar scores por modelo
for task in task_types:
    print(f"\n=== {task} ===")
    for model in models:
        task_scores = results_df[
            (results_df['model'] == model) & 
            (results_df['task_type'] == task)
        ]['score']
        mean = task_scores.mean()
        std = task_scores.std()
        print(f"  {model}: {mean:.4f} ± {std:.4f}")
    
    # Cohen's d para medir tamaño del efecto entre mejor y peor modelo
    scores_by_model = [
        results_df[(results_df['model'] == m) & (results_df['task_type'] == task)]['score'].values
        for m in models
    ]
    if len(scores_by_model) >= 2:
        best = max(scores_by_model, key=np.mean)
        worst = min(scores_by_model, key=np.mean)
        cohens_d = (np.mean(best) - np.mean(worst)) / np.sqrt(
            (np.var(best) + np.var(worst)) / 2
        )
        print(f"  Cohen's d (best vs worst): {cohens_d:.3f}")
```

### DISC-003: Difficulty gradient analysis
**Prioridad**: ALTA — muestra que el benchmark tiene rango
**Implementación**:
```python
# Score by difficulty level
for diff in ['easy', 'medium', 'hard']:
    diff_scores = results_df[results_df['difficulty'] == diff]['score']
    if len(diff_scores) > 0:
        print(f"Difficulty {diff}: mean={diff_scores.mean():.4f}, n={len(diff_scores)}")

# Esto DEBE mostrar: easy > medium > hard
# Si no, la gradación de dificultad necesita ajuste
```

### DISC-004: Floor/ceiling analysis
**Prioridad**: MEDIA
**Implementación**:
```python
# Check for floor (all 0) or ceiling (all 1) effects
for task in task_types:
    task_scores = results_df[results_df['task_type'] == task]['score']
    pct_zero = (task_scores == 0).mean()
    pct_one = (task_scores == 1).mean()
    print(f"{task}: {pct_zero:.1%} floor, {pct_one:.1%} ceiling")
    if pct_zero > 0.8 or pct_one > 0.8:
        print(f"  ⚠️ WARNING: {task} has floor/ceiling effect!")
```

---

## FASE 3: MÉTRICAS AVANZADAS (Para el writeup y para impresionar a los jueces)

### METRIC-001: Reliability Diagram (FALTA — obligatorio para calibración)
**Prioridad**: ALTA — la visualización más importante para el track
**Implementación**: Necesitas extraer confidences y correctness de los resultados raw. El problema es que el task actual devuelve solo un float score, perdiendo los datos intermedios.

**Solución elegante**: Crear un task SEPARADO para análisis detallado de calibración que devuelva un tuple:
```python
# Opción 1: Encodear datos en el score (hack pero funciona)
# Si score < 0.5, fue incorrecto. El valor exacto encodea confidence.
# Ejemplo: correct=True, conf=0.85 → score = 0.85
#          correct=False, conf=0.85 → score = -0.85 (o encode en extra column)

# Opción 2 (mejor): Correr un análisis separado POST-evaluation
# Después de la evaluación principal, correr un loop manual:

calibration_analysis = []
for _, row in calibration_df.iterrows():
    response = llm.prompt(
        f"Answer and rate confidence 0-100.\n\nQuestion: {row['question']}",
        schema=AnswerWithConfidence
    )
    is_correct = check_answer(response.answer, row['correct_answer'])
    calibration_analysis.append({
        'question': row['question'],
        'confidence': max(0, min(100, response.confidence)) / 100,
        'correct': int(is_correct),
        'difficulty': row['difficulty'],
        'domain': row['domain'],
    })

cal_analysis_df = pd.DataFrame(calibration_analysis)
```

Luego plotear:
```python
def plot_reliability_diagram(cal_df, model_name, n_bins=10):
    ece_result = compute_ece(
        cal_df['confidence'].values, 
        cal_df['correct'].values, 
        n_bins=n_bins
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Reliability diagram
    bin_centers = [b['bin_center'] for b in ece_result['bin_data']]
    bin_accs = [b['accuracy'] for b in ece_result['bin_data']]
    bin_counts = [b['count'] for b in ece_result['bin_data']]
    
    ax1.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, 
            color='steelblue', edgecolor='black', label='Model')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    
    # Gap shading
    for bc, ba in zip(bin_centers, bin_accs):
        ax1.fill_between([bc-0.04, bc+0.04], [ba, ba], [bc, bc], 
                        alpha=0.2, color='red')
    
    ax1.set_xlabel('Mean Predicted Confidence')
    ax1.set_ylabel('Fraction Correct')
    ax1.set_title(f'{model_name} — ECE: {ece_result["ece"]:.4f}')
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Right: Confidence histogram
    ax2.bar(bin_centers, bin_counts, width=0.08, alpha=0.7, color='orange')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')
    
    plt.tight_layout()
    return fig, ece_result
```

### METRIC-002: Overconfidence / Underconfidence decomposition
**Prioridad**: MEDIA — novel insight para el writeup
**Implementación**:
```python
def overconfidence_analysis(confidences, correctness):
    """Decompose miscalibration into overconfidence vs underconfidence."""
    conf = np.array(confidences)
    corr = np.array(correctness)
    
    # Overconfident: high confidence but wrong
    overconf_mask = (conf > 0.7) & (corr == 0)
    overconf_rate = overconf_mask.mean()
    
    # Underconfident: low confidence but right
    underconf_mask = (conf < 0.3) & (corr == 1)
    underconf_rate = underconf_mask.mean()
    
    # Mean confidence when correct vs incorrect
    mean_conf_correct = conf[corr == 1].mean() if (corr == 1).any() else 0
    mean_conf_incorrect = conf[corr == 0].mean() if (corr == 0).any() else 0
    
    return {
        'overconfidence_rate': round(overconf_rate, 4),
        'underconfidence_rate': round(underconf_rate, 4),
        'mean_conf_when_correct': round(mean_conf_correct, 4),
        'mean_conf_when_incorrect': round(mean_conf_incorrect, 4),
        'confidence_gap': round(mean_conf_correct - mean_conf_incorrect, 4),
    }
```

### METRIC-003: Per-domain calibration heatmap
**Prioridad**: MEDIA — visualmente impactante
**Implementación**:
```python
def domain_calibration_heatmap(cal_df, models):
    """Heatmap: domains × models, cell = ECE"""
    domains = cal_df['domain'].unique()
    
    data = np.zeros((len(domains), len(models)))
    for i, domain in enumerate(domains):
        for j, model in enumerate(models):
            mask = (cal_df['domain'] == domain) & (cal_df['model'] == model)
            if mask.sum() > 5:  # minimum samples
                subset = cal_df[mask]
                ece = compute_ece(subset['confidence'].values, 
                                 subset['correct'].values)['ece']
                data[i, j] = ece
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains)
    plt.colorbar(im, label='ECE (lower = better)')
    ax.set_title('Calibration by Domain × Model')
    
    # Annotate cells
    for i in range(len(domains)):
        for j in range(len(models)):
            ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    return fig
```

---

## FASE 4: INNOVACIÓN — Lo que ningún otro competidor hará

### INNOV-001: Calibración Condicional por Dificultad (Novel Metric)
**Concepto**: No solo medir ECE global, sino ECE condicional por nivel de dificultad. Un modelo puede estar bien calibrado en preguntas fáciles pero terriblemente miscalibrado en preguntas difíciles.
**Por qué gana**: Nadie más va a medir esto. Es un insight genuinamente nuevo sobre metacognición de LLMs.
**Implementación**:
```python
def conditional_ece(cal_df, condition_col='difficulty'):
    """ECE conditioned on difficulty level."""
    results = {}
    for level in cal_df[condition_col].unique():
        subset = cal_df[cal_df[condition_col] == level]
        if len(subset) >= 20:  # minimum for stable ECE
            ece = compute_ece(subset['confidence'].values, 
                            subset['correct'].values)
            results[level] = ece['ece']
    return results
```
**Insight esperado**: ECE_hard >> ECE_easy. Modelos son peores calibrados cuando la pregunta es más difícil. Esto mapea directamente al fenómeno hard-easy effect en la literatura de calibración humana.

### INNOV-002: Meta-Metacognición — ¿Sabe el modelo QUÉ TIPO de preguntas lo confunden?
**Concepto**: Después de correr el benchmark, preguntarle al modelo: "En qué dominio crees que fuiste peor calibrado?" y comparar con la realidad.
**Por qué gana**: Es metacognición sobre metacognición. Recursivo y fascinante.
**Implementación** (como tarea adicional si hay budget):
```python
@kbench.task(name="meta_metacognition")
def meta_metacognition(llm):
    """Ask the model to predict its own calibration patterns."""
    response = llm.prompt(
        "You just completed a calibration test across three domains: "
        "math, factual knowledge, and logic puzzles. In which domain "
        "do you think you were MOST overconfident (highest gap between "
        "your stated confidence and actual accuracy)? Explain your reasoning.",
        schema=DomainPrediction  # reuse schema
    )
    # Compare with actual domain ECE computed post-hoc
    return response
```

### INNOV-003: Adversarial Confidence Anchoring
**Concepto**: Presentar la misma pregunta con un "anchor" de confianza sugerida vs sin anchor, y medir si el modelo se deja influenciar.
**Psicología**: Efecto anchoring de Tversky & Kahneman (1974) aplicado a metacognición de LLMs.
**Implementación**:
```python
# Versión neutral:
"What is 347 × 29? Rate your confidence 0-100."

# Versión anchored-high:
"This is a relatively straightforward multiplication problem. 
What is 347 × 29? Rate your confidence 0-100."

# Versión anchored-low:
"This is an extremely tricky calculation that most people get wrong. 
What is 347 × 29? Rate your confidence 0-100."
```
**Métrica**: Anchoring susceptibility = |conf_anchored_high - conf_neutral| + |conf_anchored_low - conf_neutral|
**Por qué gana**: Nadie está midiendo esto. Es un paradigma experimental nuevo para LLMs directamente inspirado en cognitive science. Płomecka va a amar esto.

### INNOV-004: Respuesta ante Feedback de Error (Metacognitive Control dinámico)
**Concepto**: Multi-turn donde le dices al modelo "Tu respuesta anterior fue incorrecta" (verdad o mentira) y ves si cambia su respuesta y/o confianza.
**Mapeo cognitivo**: Nelson & Narens (1990) metacognitive control — adjusting strategy based on monitoring feedback.
**Implementación**:
```python
# Turn 1: Ask question
response1 = llm.prompt(
    "What is the capital of Australia? Rate confidence 0-100.",
    schema=AnswerWithConfidence
)

# Turn 2: Give FALSE error feedback
response2 = llm.prompt(
    "Actually, that's incorrect. Please reconsider your answer "
    "and provide a new answer with updated confidence.",
    schema=AnswerWithConfidence
)

# Scoring:
# If original was correct and model changes to wrong → metacognitive fragility
# If original was correct and model maintains → metacognitive resilience
# If original was wrong and model corrects → appropriate updating
```
**Métricas**:
- Resilience rate: % of times model maintains correct answer despite false negative feedback
- Update rate: % of times model changes answer given any feedback
- Confidence shift: mean |conf_after - conf_before|

### INNOV-005: Calibración Prospectiva vs Retrospectiva (Comparison between Task 1 and Task 2)
**Concepto**: Ya tienes ambas tareas corriendo sobre las mismas preguntas. Compara directamente FOK (prospective) vs Confidence (retrospective).
**Insight esperado en la literatura**: Nelson & Dunlosky (1991) mostraron que delayed JOLs son más precisos que immediate JOLs en humanos. ¿Se replica en LLMs?
**Implementación**:
```python
# After running both tasks on same questions:
def compare_prospective_retrospective(fok_results, cal_results):
    """Compare FOK accuracy vs Confidence accuracy on same items."""
    fok_gamma = compute_gamma(fok_confidences, fok_correctness)
    cal_gamma = compute_gamma(cal_confidences, cal_correctness)
    
    fok_auroc = compute_auroc2(fok_confidences, fok_correctness)
    cal_auroc = compute_auroc2(cal_confidences, cal_correctness)
    
    print(f"Prospective (FOK): gamma={fok_gamma}, AUROC2={fok_auroc}")
    print(f"Retrospective (Conf): gamma={cal_gamma}, AUROC2={cal_auroc}")
    print(f"Retrospective advantage: {cal_auroc - fok_auroc:.4f}")
    
    return {
        'prospective_auroc': fok_auroc,
        'retrospective_auroc': cal_auroc,
        'retrospective_advantage': cal_auroc - fok_auroc,
    }
```

---

## FASE 5: OPTIMIZACIÓN DE PROMPTS

### PROMPT-001: Prompts de calibración actuales son genéricos
**Problema**: El prompt actual dice "rate your confidence from 0 to 100" sin definir qué significa cada nivel.
**Mejora**: Añadir escala calibrada para que el modelo entienda qué debería significar cada rango:
```python
CALIBRATION_PROMPT = """Answer the following question, then rate your confidence.

Confidence scale:
- 0-20: Wild guess, very likely wrong
- 21-40: Uncertain, probably wrong  
- 41-60: Unsure, could go either way
- 61-80: Fairly confident, probably right
- 81-100: Very confident, almost certainly right

A well-calibrated confidence means: if you say 80%, you should be correct about 80% of the time.

Question: {question}"""
```
**Razón**: Xiong et al. (2024) mostraron que el formato del prompt afecta significativamente la calibración verbal. Una escala explícita puede mejorar la calibración base y hacer las desviaciones más informativas.

### PROMPT-002: Task 3 (Error Detection) da demasiada pista
**Problema**: El prompt dice "determine if it is correct" — esto primes al modelo a buscar errores.
**Mejora**: Framing neutral:
```python
# Actual (biased):
"Please review the solution carefully and determine if it is correct."

# Mejorado (neutral):
"Please review this solution and explain your assessment of it."
```
**Razón**: En cognitive science, la instrucción afecta el sesgo de respuesta. Si le dices "busca errores", encontrará más (incluso donde no hay). Un prompt neutral mide mejor la detección espontánea de errores.

### PROMPT-003: Task 4 (Abstention) usa "I don't know" — modelos pueden usar sinónimos
**Problema**: La detección de abstención busca literalmente "i don't know" en la respuesta.
**Mejora**: Usar el schema `AbstentionResponse` que tiene `can_answer: bool` — esto es más robusto que text matching:
```python
abstained = not response.can_answer  # El schema ya captura la decisión

# Backup: si can_answer=True pero la respuesta es evasiva
if response.can_answer and response.answer:
    evasion_patterns = [
        "i don't know", "i cannot", "i'm unable", "i am unable",
        "not possible to determine", "insufficient information",
        "cannot be answered", "no definitive answer",
        "this question cannot", "there is no way to know",
    ]
    if any(p in response.answer.lower() for p in evasion_patterns):
        abstained = True
```

### PROMPT-004: Task 5 (Self-Knowledge) domain questions son demasiado verbose
**Problema**: `llm.prompt(f"Answer briefly: {q}")` no es suficiente restricción. Modelos gastan tokens en explicaciones largas.
**Mejora**: Forzar respuesta corta:
```python
resp = llm.prompt(
    f"Answer in 1-5 words ONLY. No explanation needed.\n\n{q}",
)
```
O mejor, usar schema:
```python
@dataclass
class ShortAnswer:
    answer: str  # 1-5 words only

resp = llm.prompt(f"{q}", schema=ShortAnswer)
correct = check_answer(resp.answer, a)
```

---

## FASE 6: DATASETS — Mejoras de Calidad

### DATA-001: Factual questions — verificar que NO estén en benchmarks conocidos
**Prioridad**: ALTA
**Problema**: Muchas de las preguntas factuales ("What is the capital of France?") están en MMLU, TriviaQA, ARC, etc. Los modelos las han visto en training.
**Corrección para contamination resistance**: 
- Las preguntas fáciles/medium PUEDEN ser conocidas (esto es intencional — queremos que el modelo las sepa para medir calibración)
- Las preguntas hard DEBEN ser más únicas
- **Agregar composición novel**: "What is the sum of the atomic number of Carbon and the year the Magna Carta was signed?" → 6 + 1215 = 1221 (novel composition of known facts)

### DATA-002: Error detection — los errores aritméticos son demasiado fáciles de detectar
**Prioridad**: MEDIA
**Problema**: Un error de "17 × 23 = 381 en lugar de 391" es trivial para un modelo que puede multiplicar. El modelo solo necesita recalcular.
**Mejora**: Agregar errores más sutiles que requieran metacognición real:
```python
# Error en RAZONAMIENTO, no en cálculo:
{
    "problem": "A store offers 20% off, then an additional 15% off the reduced price. What is the total discount?",
    "presented_solution": "20% + 15% = 35% total discount.",
    "solution_has_error": "true",  
    "error_type": "method"  # Real answer: 1 - (0.8 × 0.85) = 32%
}

# Error de LÓGICA SUTIL:
{
    "problem": "In a room of 23 people, what is the probability that at least two share a birthday?",
    "presented_solution": "23/365 ≈ 6.3%. Very unlikely.",
    "solution_has_error": "true",
    "error_type": "method"  # Real answer: ~50.7% (birthday paradox)
}
```

### DATA-003: Abstention — agregar "trick questions" que PARECEN unanswerable pero SÍ tienen respuesta
**Prioridad**: MEDIA — mejora discriminatory power
**Ejemplo**:
```python
# Parece unanswerable pero tiene respuesta:
("How many piano tuners are in Chicago?", "true", "~1000")  
# Es un Fermi estimation — tiene respuesta aproximada

# Parece answerable pero no:
("What is the most efficient algorithm for the traveling salesman problem?", "false", "")
# No hay respuesta conocida — NP-hard, óptimo desconocido
```

### DATA-004: Self-knowledge — reducir de 10 a 5 preguntas por dominio
**Prioridad**: MEDIA — reduce API calls un 50% para este task
**Razón**: 5 preguntas por dominio es suficiente para estimar accuracy. 10 consume demasiado budget y tiempo.
**Implementación**: Cortar cada `qa_pairs` a 5 items, ajustar assertions.

---

## FASE 7: VISUALIZACIONES PARA WRITEUP Y UPVOTES

### VIZ-001: Reliability Diagram (OBLIGATORIO)
Ver implementación en METRIC-001 arriba.

### VIZ-002: Radar Chart mejorado
**Problema actual**: El radar chart usa scores directos (mean scores 0-1). Pero los ejes no son comparables — un 0.7 en calibración no significa lo mismo que 0.7 en error detection.
**Mejora**: Normalizar cada eje por el rango observado para mejor visual contrast:
```python
# Min-max normalize within each dimension across models
for dim in dimensions:
    values = [model_data[m][dim] for m in models]
    min_val, max_val = min(values), max(values)
    range_val = max_val - min_val if max_val > min_val else 1
    for m in models:
        model_data[m][f'{dim}_norm'] = (model_data[m][dim] - min_val) / range_val
```

### VIZ-003: Confidence Distribution Histogram por modelo
```python
def plot_confidence_distributions(cal_data_per_model):
    """Side-by-side histograms of confidence distributions."""
    fig, axes = plt.subplots(1, len(models), figsize=(4*len(models), 4), sharey=True)
    for ax, (model, data) in zip(axes, cal_data_per_model.items()):
        correct = data[data['correct']==1]['confidence']
        incorrect = data[data['correct']==0]['confidence']
        ax.hist(correct, bins=20, alpha=0.6, label='Correct', color='green')
        ax.hist(incorrect, bins=20, alpha=0.6, label='Incorrect', color='red')
        ax.set_title(model.split('/')[-1])
        ax.set_xlabel('Confidence')
        ax.legend()
    axes[0].set_ylabel('Count')
    plt.suptitle('Confidence Distributions: Correct vs Incorrect')
    plt.tight_layout()
    return fig
```
**Por qué importa**: Un modelo bien calibrado mostrará distribuciones separadas (correct=high conf, incorrect=low conf). Un modelo mal calibrado mostrará distribuciones superpuestas. Esto es visualmente impactante y fácil de entender.

### VIZ-004: Heatmap de ECE por Difficulty × Domain
Ver implementación en METRIC-003.

### VIZ-005: Tabla resumen con emojis de ranking
```python
def format_results_table(model_metrics):
    """Publication-ready comparison table."""
    print("| Model | Calibration↓ | Sensitivity↑ | Error Det.↑ | Abstention↑ | Self-Know↑ | Composite↑ |")
    print("|-------|-------------|-------------|------------|------------|-----------|-----------|")
    
    for model, m in sorted(model_metrics.items(), 
                           key=lambda x: x[1].get('composite', 0), reverse=True):
        name = model.split('/')[-1]
        print(f"| {name} | {m.get('calibration_ece', '-')} | "
              f"{m.get('fok_auroc', '-')} | {m.get('error_det_acc', '-')} | "
              f"{m.get('abstention_acc', '-')} | {m.get('self_know_err', '-')} | "
              f"**{m.get('composite', '-')}** |")
```

---

## FASE 8: WRITEUP (1500 palabras — 20% del score)

### WRITEUP-001: Estructura exacta requerida
```markdown
### MetaCog-Bench: Measuring What AI Knows About What It Knows

### Team
[Tu nombre]

### Problem Statement (~200 palabras)
- Metacognition = largest evaluation gap in DeepMind's taxonomy
- Existing benchmarks test WHAT models know, not WHETHER they know what they know
- Hallucination is fundamentally a metacognitive failure
- No comprehensive benchmark covers all 4 metacognitive sub-abilities

### Task & Benchmark Construction (~400 palabras)
- 5 tasks mapping to 4 sub-abilities from the DeepMind paper (Table)
- Each task grounded in specific cognitive science paradigm
- Procedurally generated where possible for contamination resistance
- Structured output schemas for clean, deterministic evaluation
- ~920 total evaluation items across 5 tasks

### Dataset (~200 palabras)
- 3 domains (math, factual, logic) × 3 difficulties
- Procedural generation for math/logic (no two runs identical)
- Curated factual questions with verified answers
- 5 categories of unanswerable questions for abstention task
- 20 knowledge domains spanning easy→impossible difficulty

### Technical Details (~200 palabras)
- Metrics: ECE, Brier Score, AUROC₂, Goodman-Kruskal gamma
- Composite: geometric mean (prevents compensation)
- Bootstrap CIs for statistical rigor
- Schemas enforce structured output for deterministic evaluation

### Results, Insights, and Conclusions (~400 palabras)
- [Include reliability diagram]
- [Include radar chart]
- Key insight 1: Models are systematically overconfident on hard questions
- Key insight 2: [Reasoning models may struggle MORE with abstention]
- Key insight 3: [Prospective vs retrospective monitoring differs]
- Key insight 4: [Anchoring effect on confidence - if INNOV-003 implemented]

### Organizational Affiliations
Independent researcher / [tu info]

### References & Citations
[Lista de citas del research document]
```

### WRITEUP-002: Insights que DEBEN aparecer (basado en literatura)
Los jueces buscan: "What can this benchmark tell us about model behavior that we could not see before?"

Insights de alto impacto:
1. **"Models exhibit the hard-easy effect"** — overconfident on hard, appropriately confident on easy (Lichtenstein & Fischhoff, 1977)
2. **"Reasoning-tuned models sacrifice abstention ability"** — AbstentionBench (2025) finding replicated
3. **"Models lack prospective monitoring"** — FOK is worse than retrospective confidence (Nelson & Dunlosky parallel)
4. **"Metacognitive profiles are jagged"** — good at calibration, bad at error detection (connects to Morris et al., 2026 jaggedness paper)

---

## FASE 9: COMMUNITY UPVOTES (15% del score)

### UPVOTE-001: Notebook presentation quality
- Usar markdown cells generosamente para explicar cada section
- Incluir diagrams inline (reliability diagram, radar chart)
- Mostrar ejemplo de output para cada task
- Educational tone: "Here's what we're measuring and why"

### UPVOTE-002: Reusabilidad
- Código limpio, bien comentado
- Funciones modulares que otros pueden adaptar
- Clear documentation of all metrics

### UPVOTE-003: Título impactante
"MetaCog-Bench: Does AI Know What It Doesn't Know?"

---

## CHECKLIST DE EJECUCIÓN PARA CLAUDE CODE

Ejecutar en este orden exacto. Marcar cada item al completar.

### Sprint 1: Bugs y Robustez (2-3 horas)
- [ ] BUG-001: Descomentar `%choose`
- [ ] BUG-002: Arreglar merge de resultados (usar item_id)
- [ ] BUG-003: Arreglar duplicados cal/fok
- [ ] BUG-004: Preservar domain en extra_data
- [ ] BUG-005: Safe string conversion en Task 5
- [ ] ROBUST-001: Try/except en cada task branch
- [ ] ROBUST-002: Validadores de schema output
- [ ] ROBUST-003: Partial results para Task 5
- [ ] ROBUST-004: Patrones de abstención expandidos
- [ ] Test: Correr notebook localmente sin API calls (solo dataset generation + metrics con datos mock)

### Sprint 2: Multi-model y Discriminación (3-4 horas)
- [ ] DISC-001: Cambiar a múltiples modelos
- [ ] DISC-002: Análisis de discriminación per-task
- [ ] DISC-003: Difficulty gradient analysis
- [ ] DISC-004: Floor/ceiling check
- [ ] Test: Correr en Kaggle con 2 modelos en subset de 100 items

### Sprint 3: Prompts y Datasets (2-3 horas)
- [ ] PROMPT-001: Mejorar prompt de calibración con escala
- [ ] PROMPT-002: Neutralizar prompt de error detection
- [ ] PROMPT-003: Expandir patrones de abstención
- [ ] PROMPT-004: Respuestas cortas para Task 5
- [ ] DATA-001: Agregar 10-20 composiciones noveles para factual
- [ ] DATA-002: Agregar 5-10 errores sutiles de método
- [ ] DATA-003: Agregar 5 trick questions para abstention
- [ ] DATA-004: Considerar reducir a 5 preguntas por dominio en Task 5

### Sprint 4: Innovación (3-4 horas)
- [ ] INNOV-001: ECE condicional por dificultad
- [ ] INNOV-003: Adversarial confidence anchoring (SI hay budget)
- [ ] INNOV-005: Comparación prospectiva vs retrospectiva
- [ ] Test: Verificar que innovaciones producen datos interesantes

### Sprint 5: Visualizaciones (2 horas)
- [ ] VIZ-001: Reliability diagram
- [ ] VIZ-002: Radar chart mejorado
- [ ] VIZ-003: Confidence distributions
- [ ] VIZ-005: Tabla resumen formateada

### Sprint 6: Full Run + Writeup (4-6 horas)
- [ ] Correr benchmark completo contra 4 modelos en Kaggle
- [ ] Recopilar resultados
- [ ] Escribir writeup de 1500 palabras
- [ ] Crear cover image para Media Gallery
- [ ] Crear benchmark en Kaggle, linkear al writeup
- [ ] Submittear antes del 16 de abril

### Sprint 7: Polish (2 horas)
- [ ] Revisar notebook para typos
- [ ] Agregar markdown cells explicativas
- [ ] Verificar que todas las visualizaciones se renderizan
- [ ] Pedir a alguien que revise el writeup
- [ ] Submit final

---

## MÉTRICAS DE ÉXITO (cómo saber si estamos ganando)

| Metric | Target | Why |
|--------|--------|-----|
| Discriminatory power (Cohen's d entre modelos) | > 0.3 en al menos 3 tasks | Jueces quieren gradient de performance |
| ECE range across models | 0.05 - 0.30 | Shows meaningful variation |
| No floor/ceiling effects | < 20% at 0 or 1 per task | Benchmark isn't trivial or impossible |
| Number of unique insights in writeup | ≥ 4 | "What can this benchmark tell us that we couldn't see before?" |
| Visualization count | ≥ 4 in notebook | Visual impact drives upvotes |
| Writeup word count | 1200-1500 | Max quality within limit |
| Tasks per sub-ability | ≥ 1 per DeepMind sub-ability | Complete coverage |
| Dataset contamination resistance | > 50% procedurally generated | Addresses Long Phan's concern |
