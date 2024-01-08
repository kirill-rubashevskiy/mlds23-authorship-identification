from llama_cpp import Llama

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
    model_path="/Users/dariamishina/Downloads/openchat_3.5.Q4_K_M.gguf",  # Download the model file first
    n_ctx=8192,  # The max sequence length to use - note that longer sequence lengths require much more resources
    n_threads=8,  # The number of CPU threads to use, tailor to your system and the resulting performance
    n_gpu_layers=0,  # The number of layers to offload to GPU, if you have GPU acceleration available
)


output = llm(
    #   "GPT4 Correct User: какая ты модель<|end_of_turn|>GPT4 Correct Assistant:", # Prompt для теста
    "GPT4 Correct User: кто написал эти строки: Мороз и солнце, день чудесный! <|end_of_turn|>GPT4 Correct Assistant:",
    max_tokens=512,  # Generate up to 512 tokens
    stop=[
        "</s>"
    ],  # Example stop token - not necessarily correct for this specific model! Please check before using.
    echo=True,  # Whether to echo the prompt
)
print(output["choices"][0]["text"].split("GPT4 Correct Assistant: ")[1])
