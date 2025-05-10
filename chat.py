import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from dotenv import load_dotenv
import sys
import time

def load_model_and_tokenizer():
    """Load the base model, tokenizer, and our fine-tuned adapter."""
    print("🧘 Loading the base model and tokenizer...")
    
    # Load environment variables
    load_dotenv()
    hf_username = os.getenv("HF_USERNAME")
    if not hf_username:
        raise ValueError("HF_USERNAME not found in environment variables!")
    
    # Model paths
    base_model_path = "microsoft/Phi-4-mini-instruct"
    adapter_path = f"{hf_username}/Phi-4-mini-instruct-persona-qlora"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load adapter
    print(f"✨ Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, messages, max_new_tokens=128):
    """Generate a streaming response from the model."""
    # Format the messages using the chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Generate response with streaming
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Start generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream the output and clean it up
    print("\n🤖 Assistant: ", end="", flush=True)
    full_response = ""
    for text in streamer:
        # Clean up special tokens and chat template artifacts
        if "<|end|>" in text:
            text = text.split("<|end|>")[0]
            print(text, end="", flush=True)
            full_response += text
            break
        text = text.replace("<|user|>", "").replace("<|assistant|>", "")
        if text.strip():  # Only print non-empty text
            print(text, end="", flush=True)
            full_response += text
    
    print("\n")
    return full_response.strip()

def main():
    """Main chat loop."""
    try:
        model, tokenizer = load_model_and_tokenizer()
        print("\n✨ Model loaded successfully! Let's begin our zen chat session.")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'clear' to start a new conversation.")
        print("\n" + "="*50 + "\n")
        
        messages = []
        
        while True:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("\n🙏 Thank you for this enlightening conversation!")
                break
            elif user_input.lower() == 'clear':
                messages = []
                print("\n🧹 Conversation cleared!")
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Generate response and capture it
            response = generate_response(model, tokenizer, messages)
            
            # Add assistant's response to the conversation
            messages.append({"role": "assistant", "content": response})
            
    except Exception as e:
        print(f"\n😭 An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    from threading import Thread
    from transformers import TextIteratorStreamer
    main() 