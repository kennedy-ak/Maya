# Ghana Sexual Health Chatbot

An AI-powered chatbot designed to provide accurate, helpful information about contraception, STIs, and reproductive health in Ghana. Built using a fine-tuned Llama-3.2-3B model with LoRA adapters.

## 🌟 Features

- **Interactive Chat Interface**: Web-based chat interface for user-friendly interactions
- **REST API**: FastAPI-powered API endpoints for programmatic access
- **Multiple Interaction Modes**: 
  - Interactive chat mode
  - Single question answering
  - Batch processing
- **Streaming Responses**: Real-time token-by-token response generation
- **Context-Aware**: Specialized training data for Ghana-specific sexual health topics

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- 8GB+ RAM (16GB recommended for optimal performance)
- Internet connection for model downloading

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd maya
   ```

2. **Install dependencies**
   ```bash
   # Using pip
   pip install -r requirements.txt
   
   # Or using uv (recommended)
   uv sync
   ```

3. **Download the model**
   The model will be automatically downloaded on first run. Make sure you have a stable internet connection.

## 📖 Usage

### Web Interface

1. **Start the server**
   ```bash
   python server.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:8000` to access the chat interface

### Command Line Interface

#### Interactive Chat Mode
```bash
python maya.py
```

#### Single Question Mode
```bash
python maya.py --question "What are the different types of contraception available in Ghana?"
```

#### Batch Processing Mode
```bash
python maya.py --batch questions.txt --output responses.txt
```

### API Endpoints

The FastAPI server provides the following endpoints:

- `GET /` - Web chat interface
- `GET /health` - Health check
- `POST /ask` - Ask a single question
- `POST /ask/stream` - Stream response token by token
- `GET /docs` - API documentation

#### Example API Usage

**Ask a question:**
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What are the side effects of birth control pills?",
       "temperature": 0.7,
       "max_tokens": 256
     }'
```

**Stream response:**
```bash
curl -X POST "http://localhost:8000/ask/stream" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "Where can I get STI testing in Accra?",
       "temperature": 0.7,
       "max_tokens": 256
     }' \
     --no-buffer
```

## 🏗️ Project Structure

```
maya/
├── README.md                    # This file
├── main.py                      # Training data merging script
├── maya.py                      # Main CLI interface for the chatbot
├── server.py                    # FastAPI server for web interface
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
├── Dockerfile                   # Docker containerization
├── docker-compose.yml          # Docker Compose configuration
├── static/
│   └── index.html              # Web chat interface
├── ghana_contraception_lora/   # Fine-tuned model directory (auto-downloaded)
├── training_data/
│   ├── batch1.json             # Training data batches
│   ├── batch2.json
│   └── ...
└── scripts/
    ├── combine_training_data.py # Data combination utilities
    ├── add_new_data.py         # Data addition utilities
    └── add_last_data.py        # Final data processing
```

## 🤖 Model Information

- **Base Model**: Llama-3.2-3B
- **Fine-tuning**: LoRA adapters for Ghana sexual health domain
- **Training Data**: Curated Q&A pairs covering contraception, STIs, and reproductive health
- **Context**: Specialized for Ghana healthcare system and cultural context

## ⚙️ Configuration

### Environment Variables

- `MODEL_PATH`: Path to the LoRA adapter model (default: `ghana_contraception_lora`)
- `HF_TOKEN`: Hugging Face token for model downloads

### Model Options

Available models can be configured in `maya.py`:
- `new`: Latest model trained with full dataset
- `old`: Previous model version

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
docker-compose up -d
```

### Using Docker

```bash
# Build the image
docker build -t ghana-chatbot .

# Run the container
docker run -p 8000:8000 ghana-chatbot
```

## 🔧 Development

### Training Data Processing

The project includes several scripts for processing training data:

```bash
# Combine multiple training data batches
python main.py

# Add new training data
python add_new_data.py

# Process final dataset
python add_last_data.py
```

### Data Format

Training data follows this JSON structure:
```json
{
  "metadata": {
    "total_in_batch": 100,
    "batch": "batch1",
    "language": "en",
    "domain": "ghana_sexual_health"
  },
  "qa_pairs": [
    {
      "question": "What are the different types of contraception?",
      "answer": "There are several types of contraception available..."
    }
  ]
}
```

## 📊 Performance

- **Response Time**: Typically 2-5 seconds per question
- **Memory Usage**: ~4-6GB RAM for model inference
- **Supported Languages**: English (primary), with potential for local languages

## 🛠️ Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Check internet connection
   - Verify Hugging Face token
   - Ensure sufficient disk space (5GB+)

2. **Out of Memory Errors**
   - Use CPU-only mode: `USE_CPU = True` in configuration
   - Reduce batch size or max tokens
   - Close other applications to free memory

3. **Slow Performance**
   - Use GPU if available
   - Increase system RAM
   - Optimize model settings

### Logs

Check console output for detailed error messages and progress indicators.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Submit a pull request

### Areas for Contribution

- Training data expansion
- Additional language support
- Performance optimizations
- Web interface improvements
- Testing and validation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This chatbot provides general information for educational purposes only and should not replace professional medical advice. Always consult healthcare professionals for medical concerns.

## 🔗 Links

- [Project Documentation](./docs/)
- [API Reference](./docs/api.md)
- [Training Guide](./docs/training.md)

## 📞 Support

For support, please open an issue in the repository or contact the development team.

---

*Built with ❤️ for improving sexual health education in Ghana*