import os
from typing import Dict, List, Any, Optional, Union
from smolagents import Tool

class NamedEntityRecognitionTool(Tool):
    name = "ner_tool"
    description = """
    Identifies and labels named entities in text using customizable NER models.
    Can recognize entities such as persons, organizations, locations, dates, etc.
    Returns a structured analysis of all entities found in the input text.
    """
    inputs = {
        "text": {
            "type": "string",
            "description": "The text to analyze for named entities",
        },
        "model": {
            "type": "string",
            "description": "The NER model to use (default: 'dslim/bert-base-NER')",
            "nullable": True
        },
        "aggregation": {
            "type": "string",
            "description": "How to aggregate entities: 'simple' (just list), 'grouped' (by label), or 'detailed' (with confidence scores)",
            "nullable": True
        },
        "min_score": {
            "type": "number",
            "description": "Minimum confidence score threshold (0.0-1.0) for including entities",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self):
        """Initialize the NER Tool with default settings."""
        super().__init__()
        self.default_model = "dslim/bert-base-NER"
        self.available_models = {
            "dslim/bert-base-NER": "Standard NER (English)",
            "jean-baptiste/camembert-ner": "French NER",
            "Davlan/bert-base-multilingual-cased-ner-hrl": "Multilingual NER",
            "Babelscape/wikineural-multilingual-ner": "WikiNeural Multilingual NER",
            "flair/ner-english-ontonotes-large": "OntoNotes English (fine-grained)",
            "elastic/distilbert-base-cased-finetuned-conll03-english": "CoNLL (fast)"
        }
        self.entity_colors = {
            "PER": "🟥 Person",
            "PERSON": "🟥 Person",
            "LOC": "🟨 Location",
            "LOCATION": "🟨 Location",
            "GPE": "🟨 Location",
            "ORG": "🟦 Organization",
            "ORGANIZATION": "🟦 Organization",
            "MISC": "🟩 Miscellaneous",
            "DATE": "🟪 Date",
            "TIME": "🟪 Time",
            "MONEY": "💰 Money",
            "PERCENT": "📊 Percentage",
            "PRODUCT": "🛒 Product",
            "EVENT": "🎫 Event",
            "WORK_OF_ART": "🎨 Work of Art",
            "LAW": "⚖️ Law",
            "LANGUAGE": "🗣️ Language",
            "FAC": "🏢 Facility",
            # Fix for models that don't properly tag entities
            "O": "Not an entity",
            "UNKNOWN": "🔷 Entity"
        }
        # Pipeline will be lazily loaded
        self._pipeline = None

    def _load_pipeline(self, model_name: str):
        """Load the NER pipeline with the specified model."""
        try:
            from transformers import pipeline
            import torch
            
            # Try to detect if GPU is available
            device = 0 if torch.cuda.is_available() else -1
            
            # For some models, we need special handling
            if "dslim/bert-base-NER" in model_name:
                # This model works better with a specific aggregation strategy
                self._pipeline = pipeline(
                    "ner", 
                    model=model_name, 
                    aggregation_strategy="first",
                    device=device
                )
            else:
                self._pipeline = pipeline(
                    "ner", 
                    model=model_name, 
                    aggregation_strategy="simple",
                    device=device
                )
            return True
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            try:
                # Fall back to default model
                from transformers import pipeline
                import torch
                device = 0 if torch.cuda.is_available() else -1
                self._pipeline = pipeline(
                    "ner", 
                    model=self.default_model, 
                    aggregation_strategy="first",
                    device=device
                )
                return True
            except Exception as fallback_error:
                print(f"Error loading fallback model: {str(fallback_error)}")
                return False

    def _get_friendly_label(self, label: str) -> str:
        """Convert technical entity labels to friendly descriptions with color indicators."""
        # Strip B- or I- prefixes that indicate beginning or inside of entity
        clean_label = label.replace("B-", "").replace("I-", "")
        
        # Handle common name and location patterns with heuristics
        if clean_label == "UNKNOWN" or clean_label == "O":
            # Apply some basic heuristics to detect entity types
            # This is a fallback when the model fails to properly tag
            text = self._current_entity_text.lower() if hasattr(self, '_current_entity_text') else ""
            
            # Check for capitalized words which might be names or places
            if text and text[0].isupper():
                # Countries and major cities
                countries_and_cities = ["germany", "france", "spain", "italy", "london", 
                                        "paris", "berlin", "rome", "new york", "tokyo", 
                                        "beijing", "moscow", "canada", "australia", "india",
                                        "china", "japan", "russia", "brazil", "mexico"]
                
                if text.lower() in countries_and_cities:
                    return self.entity_colors.get("LOC", "🟨 Location")
                
                # Common first names (add more as needed)
                common_names = ["john", "mike", "sarah", "david", "michael", "james",
                               "robert", "mary", "jennifer", "linda", "michael", "william",
                               "kristof", "chris", "thomas", "daniel", "matthew", "joseph",
                               "donald", "richard", "charles", "paul", "mark", "kevin"]
                
                name_parts = text.lower().split()
                if name_parts and name_parts[0] in common_names:
                    return self.entity_colors.get("PER", "🟥 Person")
            
        return self.entity_colors.get(clean_label, f"🔷 {clean_label}")

    def forward(self, text: str, model: str = None, aggregation: str = None, min_score: float = None) -> str:
        """
        Perform Named Entity Recognition on the input text.
        
        Args:
            text: The text to analyze
            model: NER model to use (default: dslim/bert-base-NER)
            aggregation: How to aggregate results (simple, grouped, detailed)
            min_score: Minimum confidence threshold (0.0-1.0)
            
        Returns:
            Formatted string with NER analysis results
        """
        # Set default values if parameters are None
        if model is None:
            model = self.default_model
        if aggregation is None:
            aggregation = "grouped"
        if min_score is None:
            min_score = 0.8
            
        # Validate model choice
        if model not in self.available_models and not model.startswith("dslim/"):
            return f"Model '{model}' not recognized. Available models: {', '.join(self.available_models.keys())}"
            
        # Load the model if not already loaded or if different from current
        if self._pipeline is None or self._pipeline.model.name_or_path != model:
            if not self._load_pipeline(model):
                return "Failed to load NER model. Please try a different model."
                
        # Perform NER analysis
        try:
            entities = self._pipeline(text)
            
            # Filter by confidence score
            entities = [e for e in entities if e.get('score', 0) >= min_score]
            
            # Store the text for better heuristics
            for entity in entities:
                word = entity.get("word", "")
                start = entity.get("start", 0)
                end = entity.get("end", 0)
                # Store the actual text from the input for better entity type detection
                entity['actual_text'] = text[start:end]
                # Set this for _get_friendly_label to use
                self._current_entity_text = text[start:end]
            
            if not entities:
                return "No entities were detected in the text with the current settings."
                
            # Process results based on aggregation method
            if aggregation == "simple":
                return self._format_simple(text, entities)
            elif aggregation == "detailed":
                return self._format_detailed(text, entities)
            else:  # default to grouped
                return self._format_grouped(text, entities)
                
        except Exception as e:
            return f"Error analyzing text: {str(e)}"
            
    def _format_simple(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Format entities as a simple list."""
        # Process word pieces and handle subtoken merging
        merged_entities = []
        current_entity = None
        
        for entity in sorted(entities, key=lambda e: e.get("start", 0)):
            word = entity.get("word", "")
            start = entity.get("start", 0)
            end = entity.get("end", 0)
            label = entity.get("entity", "UNKNOWN")
            score = entity.get("score", 0)
            
            # Check if this is a continuation (subtoken)
            if word.startswith("##"):
                if current_entity:
                    # Extend the current entity
                    current_entity["word"] += word.replace("##", "")
                    current_entity["end"] = end
                    # Keep the average score
                    current_entity["score"] = (current_entity["score"] + score) / 2
                continue
            
            # Start a new entity
            current_entity = {
                "word": word,
                "start": start,
                "end": end,
                "entity": label,
                "score": score
            }
            merged_entities.append(current_entity)
        
        result = "Named Entities Found:\n\n"
        
        for entity in merged_entities:
            word = entity.get("word", "")
            label = entity.get("entity", "UNKNOWN")
            score = entity.get("score", 0)
            friendly_label = self._get_friendly_label(label)
            
            result += f"• {word} - {friendly_label} (confidence: {score:.2f})\n"
            
        return result
            
    def _format_grouped(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Format entities grouped by their category."""
        # Process word pieces and handle subtoken merging
        merged_entities = []
        current_entity = None
        
        for entity in sorted(entities, key=lambda e: e.get("start", 0)):
            word = entity.get("word", "")
            start = entity.get("start", 0)
            end = entity.get("end", 0)
            label = entity.get("entity", "UNKNOWN")
            score = entity.get("score", 0)
            
            # Check if this is a continuation (subtoken)
            if word.startswith("##"):
                if current_entity:
                    # Extend the current entity
                    current_entity["word"] += word.replace("##", "")
                    current_entity["end"] = end
                    # Keep the average score
                    current_entity["score"] = (current_entity["score"] + score) / 2
                continue
            
            # Start a new entity
            current_entity = {
                "word": word,
                "start": start,
                "end": end,
                "entity": label,
                "score": score
            }
            merged_entities.append(current_entity)
        
        # Group entities by their label
        grouped = {}
        
        for entity in merged_entities:
            word = entity.get("word", "")
            label = entity.get("entity", "UNKNOWN").replace("B-", "").replace("I-", "")
            
            if label not in grouped:
                grouped[label] = []
                
            grouped[label].append(word)
            
        # Build the result string
        result = "Named Entities by Category:\n\n"
        
        for label, words in grouped.items():
            friendly_label = self._get_friendly_label(label)
            unique_words = list(set(words))
            result += f"{friendly_label}: {', '.join(unique_words)}\n"
            
        return result
            
    def _format_detailed(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Format entities with detailed information including position in text."""
        # Process word pieces and handle subtoken merging
        merged_entities = []
        current_entity = None
        
        for entity in sorted(entities, key=lambda e: e.get("start", 0)):
            word = entity.get("word", "")
            start = entity.get("start", 0)
            end = entity.get("end", 0)
            label = entity.get("entity", "UNKNOWN")
            score = entity.get("score", 0)
            
            # Check if this is a continuation (subtoken)
            if word.startswith("##"):
                if current_entity:
                    # Extend the current entity
                    current_entity["word"] += word.replace("##", "")
                    current_entity["end"] = end
                    # Keep the average score
                    current_entity["score"] = (current_entity["score"] + score) / 2
                continue
            
            # Start a new entity
            current_entity = {
                "word": word,
                "start": start,
                "end": end,
                "entity": label,
                "score": score
            }
            merged_entities.append(current_entity)
            
        # First, build an entity map to highlight the entire text
        character_labels = [None] * len(text)
        
        # Mark each character with its entity
        for entity in merged_entities:
            start = entity.get("start", 0)
            end = entity.get("end", 0)
            label = entity.get("entity", "UNKNOWN")
            
            for i in range(start, min(end, len(text))):
                character_labels[i] = label
                
        # Build highlighted text sections
        highlighted_text = ""
        current_label = None
        current_segment = ""
        
        for i, char in enumerate(text):
            label = character_labels[i]
            
            if label != current_label:
                # End the previous segment if any
                if current_segment:
                    if current_label:
                        clean_label = current_label.replace("B-", "").replace("I-", "")
                        highlighted_text += f"[{current_segment}]({clean_label}) "
                    else:
                        highlighted_text += current_segment + " "
                        
                # Start a new segment
                current_label = label
                current_segment = char
            else:
                current_segment += char
                
        # Add the final segment
        if current_segment:
            if current_label:
                clean_label = current_label.replace("B-", "").replace("I-", "")
                highlighted_text += f"[{current_segment}]({clean_label})"
            else:
                highlighted_text += current_segment
                
        # Get entity details
        entity_details = []
        for entity in merged_entities:
            word = entity.get("word", "")
            label = entity.get("entity", "UNKNOWN")
            score = entity.get("score", 0)
            friendly_label = self._get_friendly_label(label)
            
            entity_details.append(f"• {word} - {friendly_label} (confidence: {score:.2f})")
            
        # Combine into final result
        result = "Entity Analysis:\n\n"
        result += "Text with Entities Marked:\n"
        result += highlighted_text + "\n\n"
        result += "Entity Details:\n"
        result += "\n".join(entity_details)
        
        return result
        
    def get_available_models(self) -> Dict[str, str]:
        """Return the dictionary of available models with descriptions."""
        return self.available_models

# Example usage:
# ner_tool = NamedEntityRecognitionTool()
# result = ner_tool("Apple Inc. is planning to open a new store in Paris, France next year.", model="dslim/bert-base-NER")
# print(result)