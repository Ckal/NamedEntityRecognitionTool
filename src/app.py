import gradio as gr
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import our NER Tool
from ner_tool import NamedEntityRecognitionTool

# Initialize the NER Tool
ner_tool = NamedEntityRecognitionTool()

# Function to analyze text using our tool
def analyze_text(text, model, aggregation, min_score):
    try:
        result = ner_tool(
            text=text,
            model=model,
            aggregation=aggregation,
            min_score=float(min_score)
        )
        return result
    except Exception as e:
        return f"Error analyzing text: {str(e)}"

# Sample texts for quick testing
sample_texts = {
    "Business News": """Apple Inc. CEO Tim Cook announced yesterday that the company will invest $5 billion
in new AI research centers across the United States and Europe. The first center will 
open in San Francisco by December 2025, followed by additional facilities in New York, 
London, and Berlin. This initiative, called 'Project Horizon', aims to compete with 
Microsoft and Google in the rapidly growing artificial intelligence market.""",
    
    "Political News": """The United Nations Security Council met in New York on Monday to discuss 
the ongoing conflict in Eastern Europe. Representatives from the United States, 
Russia, China, and the European Union presented their positions. Secretary-General 
António Guterres urged all parties to return to diplomatic negotiations by July 15th.""",
    
    "Sports News": """Manchester United defeated Liverpool 3-2 in yesterday's Premier League match 
at Old Trafford. Marcus Rashford scored two goals, while Mohamed Salah managed to score 
for Liverpool. The victory puts Manchester United in second place in the league standings, 
just behind Manchester City.""",
    
    "Academic Text": """According to researchers at Stanford University and MIT, the latest advancements 
in quantum computing could revolutionize cryptography within the next decade. The paper, 
published in the Journal of Computational Physics, suggests that Shor's algorithm implemented 
on quantum systems with just 100 qubits could potentially break RSA encryption."""
}

# Create Gradio interface
with gr.Blocks(title="Named Entity Recognition Tool") as demo:
    gr.Markdown("# 🔍 Named Entity Recognition Tool")
    gr.Markdown("Identify and analyze named entities in text using different models and display formats.")
    
    with gr.Row():
        with gr.Column():
            # Input section
            text_input = gr.Textbox(
                label="Text to Analyze", 
                placeholder="Enter text to analyze for named entities...",
                lines=10
            )
            
            # Sample texts dropdown
            sample_dropdown = gr.Dropdown(
                choices=list(sample_texts.keys()),
                label="Or Select a Sample Text"
            )
            
            # Configuration options
            with gr.Row():
                with gr.Column():
                    model_dropdown = gr.Dropdown(
                        choices=list(ner_tool.available_models.keys()),
                        value="dslim/bert-base-NER",
                        label="NER Model"
                    )
                    
                    aggregation_dropdown = gr.Dropdown(
                        choices=["simple", "grouped", "detailed"],
                        value="grouped",
                        label="Display Format"
                    )
                    
                with gr.Column():
                    min_score_slider = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.8, 
                        step=0.05,
                        label="Minimum Confidence Score"
                    )
                    
            analyze_button = gr.Button("Analyze Text")
        
        with gr.Column():
            # Output section
            result_output = gr.Textbox(label="Analysis Results", lines=20)
            
            # Model info
            gr.Markdown("### Available Models:")
            model_info = gr.HTML(
                "".join([f"<p><strong>{k}</strong>: {v}</p>" for k, v in ner_tool.available_models.items()])
            )
    
    # Set up event handlers
    def load_sample(sample_name):
        return sample_texts.get(sample_name, "")
    
    sample_dropdown.change(
        load_sample,
        inputs=sample_dropdown,
        outputs=text_input
    )
    
    analyze_button.click(
        analyze_text,
        inputs=[text_input, model_dropdown, aggregation_dropdown, min_score_slider],
        outputs=result_output
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()