import gradio as gr
from config import (
    DATA_TYPES,
    PARAMETERS,
    OPTIMIZERS,
    MEMORY_UNITS,
    load_predefined_models,
)
from utils import (
    calculate_inference_memory,
    calculate_training_memory,
)

def update_values(model_name):
    """모델 선택에 따라 파라미터 값 업데이트"""
    MODELS = load_predefined_models()
    if model_name in MODELS:
        model_info = MODELS[model_name]
        return [
            model_info.get(PARAMETERS["model_size"], None),
            model_info.get(PARAMETERS["precision"], DATA_TYPES[0]),
            model_info.get(PARAMETERS["hidden_size"], None),
            model_info.get(PARAMETERS["num_hidden_layers"], None),
            model_info.get(PARAMETERS["num_attention_heads"], None),
        ]
    return [None, DATA_TYPES[0], None, None, None]

def calculate_memory(
    model_name, model_size, precision, batch_size, sequence_length,
    hidden_size, num_hidden_layers, num_attention_heads,
    optimizer, trainable_parameters, memory_unit
):
    """메모리 계산 및 결과 반환"""
    inference_memory = calculate_inference_memory(
        model_size, precision, batch_size, sequence_length,
        hidden_size, num_hidden_layers, num_attention_heads,
        memory_unit
    )

    training_memory = calculate_training_memory(
        model_size, precision, batch_size, sequence_length,
        hidden_size, num_hidden_layers, num_attention_heads,
        optimizer, trainable_parameters, memory_unit
    )

    return (
        f"Inference Memory:\n"
        f"Total: {inference_memory['inference_memory']}\n"
        f"Model Weights: {inference_memory['model_weights']}\n"
        f"KV Cache: {inference_memory['kv_cache']}\n"
        f"Activation Memory: {inference_memory['activation_memory']}\n\n"
        f"Training Memory:\n"
        f"Total: {training_memory['training_memory']}\n"
        f"Model Weights: {training_memory['model_weights']}\n"
        f"KV Cache: {training_memory['kv_cache']}\n"
        f"Activation Memory: {training_memory['activation_memory']}\n"
        f"Optimizer Memory: {training_memory['optimizer_memory']}\n"
        f"Gradients Memory: {training_memory['gradients_memory']}"
    )

def create_interface():
    MODELS = load_predefined_models()

    with gr.Blocks(title="LLM Memory Requirements") as demo:
        gr.Markdown("# LLM Memory Requirements")

        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    label="Select Pre-defined Model"
                )
                model_size = gr.Number(
                    label="Number of parameters (in billions)",
                    minimum=0,
                    step=1
                )
                precision = gr.Dropdown(
                    choices=DATA_TYPES,
                    label="Precision",
                    value=DATA_TYPES[0]
                )
                batch_size = gr.Number(
                    label="Batch Size",
                    minimum=0,
                    step=1,
                    value=1
                )
                sequence_length = gr.Number(
                    label="Sequence Length",
                    minimum=0,
                    step=1,
                    value=2048
                )

            with gr.Column():
                hidden_size = gr.Number(
                    label="Hidden Size",
                    minimum=0,
                    step=1
                )
                num_hidden_layers = gr.Number(
                    label="Number of Layers",
                    minimum=0,
                    step=1
                )
                num_attention_heads = gr.Number(
                    label="Number of Attention Heads",
                    minimum=0,
                    step=1
                )
                optimizer = gr.Dropdown(
                    choices=list(OPTIMIZERS.keys()),
                    label="Optimizer",
                    value=list(OPTIMIZERS.keys())[0]
                )
                trainable_parameters = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=100,
                    step=1,
                    label="Percentage of trainable parameters"
                )
                memory_unit = gr.Radio(
                    choices=MEMORY_UNITS,
                    label="Memory Unit",
                    value=MEMORY_UNITS[0]
                )

        output = gr.Textbox(label="Memory Requirements", lines=10)

        # 이벤트 핸들러 연결
        model.change(
            fn=update_values,
            inputs=[model],
            outputs=[model_size, precision, hidden_size, num_hidden_layers, num_attention_heads]
        )

        calculate_btn = gr.Button("Calculate Memory")
        calculate_btn.click(
            fn=calculate_memory,
            inputs=[
                model, model_size, precision, batch_size, sequence_length,
                hidden_size, num_hidden_layers, num_attention_heads,
                optimizer, trainable_parameters, memory_unit
            ],
            outputs=[output]
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()