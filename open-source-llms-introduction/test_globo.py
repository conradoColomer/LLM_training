from mlx_lm import load, generate

# 1. Ruta al modelo base y a tus nuevos "conocimientos" (adaptadores)
model_path = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
adapter_path = "adapters" # Por defecto mlx_lm guarda aquí sus resultados

print("Cargando modelo con esteroides de Globomantics... 🧠")
model, tokenizer = load(model_path, adapter_path=adapter_path)

def preguntar_a_globo(pregunta):
    # Formateamos la pregunta para que la IA entienda el rol
    messages = [{"role": "user", "content": pregunta}]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Generamos la respuesta
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=200, # Límite de palabras de la respuesta
        verbose=False
    )
    return response

# --- PRUEBA DE FUEGO ---
pregunta_test = "¿Qué es Globomantics y quién es su CEO?"
print(f"\nUsuario: {pregunta_test}")

respuesta = preguntar_a_globo(pregunta_test)
print(f"Asistente: {respuesta}")
