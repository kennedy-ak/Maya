"""
Ghana Sexual Health Chatbot - Local Inference Script
====================================================
This script loads your finetuned model and allows you to interact with it locally.

Requirements:
- Python 3.8+
- GPU recommended (but can run on CPU)
- Downloaded model files in 'ghana_contraception_lora' folder
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

# Available models
AVAILABLE_MODELS = {
    "new": "ghana_contraception_lora",          # New model trained with full dataset
    "old": "ghana_contraception_lora_old",      # Previous model
}

DEFAULT_MODEL = "new"  # Default model to use
LORA_ADAPTER_PATH = AVAILABLE_MODELS[DEFAULT_MODEL]  # Will be overridden by command-line arg
BASE_MODEL = "unsloth/Llama-3.2-3B"  # Base model (will be downloaded if not cached)
MAX_SEQ_LENGTH = 2048
USE_CPU = True  # Running on CPU
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Hugging Face token for downloading models

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model(model_path=None):
    """Load the base model and apply LoRA adapter"""
    if model_path is None:
        model_path = LORA_ADAPTER_PATH

    print("🔄 Loading model...")
    print(f"   Base model: {BASE_MODEL}")
    print(f"   LoRA adapter: {model_path}")
    print(f"   Device: CPU (this may be slower than GPU)")
    print(f"   ⏳ First run may take a while to download the base model...")

    try:
        # Load tokenizer from the LoRA adapter folder (it has the tokenizer files)
        print("\n📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)

        # Load base model on CPU
        print("📦 Loading base model (this may take a few minutes)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            token=HF_TOKEN,
        )

        # Apply LoRA adapter
        print("🔧 Applying LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, model_path)

        # Merge LoRA weights into base model for faster inference
        print("⚡ Merging LoRA weights (this improves inference speed)...")
        model = model.merge_and_unload()

        # Set model to evaluation mode
        model.eval()

        print("✅ Model loaded successfully!")
        return model, tokenizer

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure 'ghana_contraception_lora' folder is in the same directory")
        print("2. Check that you've unzipped the model files")
        print("3. Install required packages: uv sync")
        print("4. Make sure you have internet connection (to download base model)")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def ask_question(model, tokenizer, question, temperature=0.7, max_tokens=256, stream=False):
    """
    Ask the model a question and get a response

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        question: User's question
        temperature: Controls randomness (0.0-1.0, higher = more creative)
        max_tokens: Maximum length of response
        stream: Whether to stream the response token by token

    Returns:
        str: Model's response (only if stream=False)
    """
    # Format the prompt
    prompt = f"""Below is a question about contraception in Ghana. Write a helpful, accurate response.

### Question:
{question}

### Response:
"""

    # Tokenize input
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    if stream:
        # Create a custom streamer that skips the prompt
        class CustomStreamer(TextStreamer):
            def __init__(self, tokenizer, skip_prompt=True):
                super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=True)
                self.text_queue = []
                self.prompt_tokens = 0

            def put(self, value):
                # Store tokens for later extraction
                if len(value.shape) > 1:
                    value = value[0]
                self.text_queue.append(value)
                super().put(value)

        streamer = CustomStreamer(tokenizer, skip_prompt=True)

        # Generate response with streaming
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            use_cache=True,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            streamer=streamer,
        )

        # Return the full generated text
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Response:" in full_response:
            return full_response.split("### Response:")[-1].strip()
        else:
            return full_response.split(question)[-1].strip()
    else:
        # Generate response without streaming
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            use_cache=True,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
        )

        # Decode and extract response
        full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract just the answer part
        if "### Response:" in full_response:
            answer = full_response.split("### Response:")[-1].strip()
        else:
            answer = full_response.split(question)[-1].strip()

        return answer

# ============================================================================
# INTERACTIVE CHAT MODE
# ============================================================================

def interactive_chat(model, tokenizer):
    """
    Interactive chat mode - ask multiple questions in a conversation
    """
    print("\n" + "="*70)
    print("🤖 GHANA SEXUAL HEALTH CHATBOT")
    print("="*70)
    print("\nI can help answer questions about:")
    print("  • Contraception methods")
    print("  • STIs and treatment")
    print("  • Reproductive health")
    print("  • Family planning services in Ghana")
    print("\nType 'quit', 'exit', or 'bye' to end the conversation")
    print("="*70 + "\n")
    
    while True:
        # Get user input
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break
        
        # Check for exit commands
        if question.lower() in ['quit', 'exit', 'bye', 'q']:
            print("\n👋 Goodbye! Stay safe and healthy!")
            break
        
        # Skip empty input
        if not question:
            continue
        
        # Get response from model
        print("\n🤖 Bot: ", end="", flush=True)
        print("💭 Generating response...\n", end="", flush=True)
        try:
            ask_question(model, tokenizer, question, stream=True)
            print()  # New line after streaming completes
        except Exception as e:
            print(f"\n❌ Error generating response: {e}")

        print()  # Empty line for readability

# ============================================================================
# BATCH PROCESSING MODE
# ============================================================================

def batch_process(model, tokenizer, questions_file, output_file):
    """
    Process multiple questions from a file and save responses
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        questions_file: Path to text file with one question per line
        output_file: Path to save responses
    """
    print(f"\n📂 Processing questions from: {questions_file}")
    
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        print(f"   Found {len(questions)} questions")
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"   Processing {i}/{len(questions)}...", end="\r")
            response = ask_question(model, tokenizer, question)
            results.append({
                'question': question,
                'response': response
            })
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"Q: {result['question']}\n")
                f.write(f"A: {result['response']}\n")
                f.write("-" * 70 + "\n\n")
        
        print(f"\n✅ Results saved to: {output_file}")
    
    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{questions_file}'")
    except Exception as e:
        print(f"❌ Error: {e}")

# ============================================================================
# SINGLE QUESTION MODE
# ============================================================================

def single_question(model, tokenizer, question):
    """Ask a single question and print the response"""
    print(f"\n❓ Question: {question}")
    print(f"{'='*70}")
    print(f"🤖 Response:")
    print(f"💭 Generating response...\n")
    ask_question(model, tokenizer, question, stream=True)
    print(f"\n{'='*70}\n")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ghana Sexual Health Chatbot - Ask questions about contraception, STIs, and reproductive health"
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=list(AVAILABLE_MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f'Model to use: {", ".join(AVAILABLE_MODELS.keys())} (default: {DEFAULT_MODEL})'
    )

    parser.add_argument(
        '--question', '-q',
        type=str,
        help='Ask a single question'
    )

    parser.add_argument(
        '--batch', '-b',
        type=str,
        help='Process questions from a file (one question per line)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='responses.txt',
        help='Output file for batch processing (default: responses.txt)'
    )

    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=0.7,
        help='Temperature for generation (0.0-1.0, default: 0.7)'
    )

    args = parser.parse_args()

    # Load model
    model_path = AVAILABLE_MODELS[args.model]
    model, tokenizer = load_model(model_path)
    
    # Choose mode based on arguments
    if args.question:
        # Single question mode
        single_question(model, tokenizer, args.question)
    
    elif args.batch:
        # Batch processing mode
        batch_process(model, tokenizer, args.batch, args.output)
    
    else:
        # Interactive chat mode (default)
        interactive_chat(model, tokenizer)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()