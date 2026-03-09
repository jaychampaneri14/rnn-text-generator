# RNN Character-Level Text Generator

Character-level LSTM language model trained on classic literature for text generation.

## Features
- Embedding layer + 2-layer LSTM + dropout
- Temperature sampling with top-k filtering
- Gradient clipping to prevent exploding gradients
- Perplexity tracking during training
- Multiple temperature demos (0.3 = conservative, 1.5 = creative)

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Output
- `training.png` — loss and perplexity curves
- `char_rnn.pth` — saved model + vocabulary

## Text Generation
```python
# Load and generate
checkpoint = torch.load('char_rnn.pth')
model = CharRNN(checkpoint['vocab_size'])
model.load_state_dict(checkpoint['model'])
text = generate_text(model, checkpoint['ch2i'], checkpoint['i2ch'],
                     seed_text="Once upon a", temperature=0.7)
print(text)
```

## Temperature Effect
- Low (0.3): Conservative, repetitive but coherent
- Medium (0.7): Balanced creativity and coherence
- High (1.5): Creative but less coherent
