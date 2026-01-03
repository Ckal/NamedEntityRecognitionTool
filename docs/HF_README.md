---
title: Named Entity Recognition Tool
emoji: 🌍
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 5.27.0
app_file: app.py
pinned: false
tags:
- tool
---

# Advanced Named Entity Recognition (NER) Tool for smolagents

This repository contains an enhanced Named Entity Recognition tool built for the `smolagents` library from Hugging Face. This tool allows you to:

- Identify named entities (people, organizations, locations, dates, etc.) in text
- Choose from multiple NER models for different languages and use cases
- Configure different output formats and confidence thresholds
- Use with smolagents for AI agents that can understand entities in text

## Installation

```bash
pip install smolagents transformers torch gradio
```

For faster inference on GPU:
```bash
pip install smolagents transformers torch gradio accelerate
```

## Basic Usage

```python
from ner_tool import NamedEntityRecognitionTool

# Initialize the NER tool
ner_tool = NamedEntityRecognitionTool()

# Analyze text with default settings
result = ner_tool("Apple Inc. is planning to open a new store in Paris, France next year.")
print(result)

# Analyze with custom settings
detailed_result = ner_tool(
    text="Apple Inc. is planning to open a new store in Paris, France next year.",
    model="Babelscape/wikineural-multilingual-ner",  # Different model
    aggregation="detailed",  # More detailed output format
    min_score=0.7  # Lower confidence threshold
)
print(detailed_result)
```

## Available Models

The tool includes several pre-configured models:

| Model ID | Description |
|----------|-------------|
| dslim/bert-base-NER | Standard NER (English) - Default |
| jean-baptiste/camembert-ner | French NER |
| Davlan/bert-base-multilingual-cased-ner-hrl | Multilingual NER |
| Babelscape/wikineural-multilingual-ner | WikiNeural Multilingual NER |
| flair/ner-english-ontonotes-large | OntoNotes English (fine-grained) |
| elastic/distilbert-base-cased-finetuned-conll03-english | CoNLL (fast) |

## Output Formats

The tool supports three output formats:

1. **Simple** - A simple list of entities found with their types and confidence scores
2. **Grouped** - Entities grouped by their category (default)
3. **Detailed** - A detailed analysis including the original text with entity markers

## Using with an Agent

```python
from smolagents import CodeAgent, InferenceClientModel
from ner_tool import NamedEntityRecognitionTool

# Initialize the NER tool
ner_tool = NamedEntityRecognitionTool()

# Create an agent model
model = InferenceClientModel(
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    token="your_huggingface_token"
)

# Create the agent with our NER tool
agent = CodeAgent(tools=[ner_tool], model=model)

# Run the agent
result = agent.run(
    "Analyze this text and identify all entities: 'The European Union and United Kingdom finalized a trade deal on Tuesday.'"
)
print(result)
```

## Interactive Gradio Interface

For an interactive experience, run the Gradio app:

```bash
python gradio_app.py
```

This provides a web interface where you can:
- Enter custom text or select from samples
- Choose different NER models
- Configure display formats and confidence thresholds
- See immediate results

## Customization Options

### Entity Confidence Score

- Use `min_score` parameter to filter entities by confidence
- Range: 0.0 (include all) to 1.0 (only highest confidence)
- Default: 0.8

### Entity Types

The tool can identify various entity types including:
- People (PER, PERSON)
- Organizations (ORG, ORGANIZATION)
- Locations (LOC, LOCATION, GPE)
- Dates and Times (DATE, TIME)
- Money and Percentages (MONEY, PERCENT)
- Products (PRODUCT)
- Events (EVENT)
- Works of Art (WORK_OF_ART)
- Laws (LAW)
- Languages (LANGUAGE)
- Facilities (FAC)
- Miscellaneous (MISC)

The exact entity types available depend on the chosen model.

## Sharing Your Tool

You can share your tool on the Hugging Face Hub:

```python
ner_tool.push_to_hub("your-username/advanced-ner-tool", token="your_huggingface_token")
```

## Limitations

- First-time model loading may take some time
- Some models may require significant memory (especially larger ones)
- Entity recognition accuracy varies by model and language

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

MIT