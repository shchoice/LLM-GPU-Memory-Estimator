from config import DATA_TYPE_SIZES, OPTIMIZERS


# ----------------- Memory Functions ----------------- #
def get_memory(*args, unit="GB"):
    """Convert total memory from bytes to human-readable format."""
    total = 0
    warning = False
    for arg in args:
        if arg > 0:
            total += arg
        else:
            warning = True

    # Define conversion factors
    if unit == "GB":
        KILO = 1000
    else:  # GiB
        KILO = 1024

    # Convert bytes to human-readable format
    if total == 0:
        result = ""
    elif total < KILO:
        result = f"{total} Bytes"
    elif total < KILO ** 2:
        result = f"{total / KILO:.2f} K{unit[1:]}"
    elif total < KILO ** 3:
        result = f"{total / (KILO ** 2):.2f} M{unit[1:]}"
    elif total < KILO ** 4:
        result = f"{total / (KILO ** 3):.2f} {unit}"
    else:
        result = f"{total / (KILO ** 4):.2f} T{unit[1:]}"

    result += " * " if warning else ""
    return result

def get_model_weights(model_size, precision):
    """Calculate the memory required for model weights."""
    try:
        return model_size * DATA_TYPE_SIZES[precision] * (10**9)
    except:
        return 0


def get_kv_cache(
    precision, batch_size, sequence_length, hidden_size, num_hidden_layers
):
    """Calculate the memory required for key-value cache."""
    try:
        return (
            2
            * batch_size
            * sequence_length
            * num_hidden_layers
            * hidden_size
            * DATA_TYPE_SIZES[precision]
        )
    except:
        return 0


def get_activation_memory(
    batch_size, sequence_length, hidden_size, num_attention_heads
):
    """Calculate the memory required for activations."""
    precision = "float32"
    try:
        return (
            batch_size
            * sequence_length
            * hidden_size
            * (34 + (5 * sequence_length * num_attention_heads) / hidden_size)
            * DATA_TYPE_SIZES[precision]
        )
    except:
        return 0


def get_optimizer_memory(model_size, optimizer):
    """Calculate the memory required for optimizer."""
    try:
        return OPTIMIZERS[optimizer] * model_size * (10**9)
    except:
        return 0


def get_gradient_memory(model_size, precision):
    """Calculate the memory required for gradients."""
    precision = "float32"
    try:
        return DATA_TYPE_SIZES[precision] * model_size * (10**9)
    except:
        return 0


def calculate_inference_memory(
    model_size,
    precision,
    batch_size,
    sequence_length,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
    memory_unit="GB"
):
    """Calculate the total memory required for inference."""
    model_weights = get_model_weights(model_size, precision)
    kv_cache = get_kv_cache(
        precision, batch_size, sequence_length, hidden_size, num_hidden_layers
    )
    activation_memory = get_activation_memory(
        batch_size, sequence_length, hidden_size, num_attention_heads
    )
    return {
        "model_weights": get_memory(model_weights, unit=memory_unit),
        "kv_cache": get_memory(kv_cache, unit=memory_unit),
        "activation_memory": get_memory(activation_memory, unit=memory_unit),
        "inference_memory": get_memory(model_weights, kv_cache, activation_memory, unit=memory_unit),
    }

def calculate_training_memory(
    model_size,
    precision,
    batch_size,
    sequence_length,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
    optimizer,
    trainable_parameters,
    memory_unit="GB"
):
    """Calculate the total memory required for training."""
    model_weights = get_model_weights(model_size, precision)
    kv_cache = get_kv_cache(
        precision, batch_size, sequence_length, hidden_size, num_hidden_layers
    )
    activation_memory = get_activation_memory(
        batch_size, sequence_length, hidden_size, num_attention_heads
    )
    optimizer_memory = (
        get_optimizer_memory(model_size, optimizer) * trainable_parameters / 100
    )
    gradients_memory = (
        get_gradient_memory(model_size, precision) * trainable_parameters / 100
    )

    return {
        "model_weights": get_memory(model_weights, unit=memory_unit),
        "kv_cache": get_memory(kv_cache, unit=memory_unit),
        "activation_memory": get_memory(activation_memory, unit=memory_unit),
        "optimizer_memory": get_memory(optimizer_memory, unit=memory_unit),
        "gradients_memory": get_memory(gradients_memory, unit=memory_unit),
        "training_memory": get_memory(
            model_weights,
            kv_cache,
            activation_memory,
            optimizer_memory,
            gradients_memory,
            unit=memory_unit
        ),
    }
