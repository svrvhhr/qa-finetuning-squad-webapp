"""
Multi-Model Question Answering Comparison
Compare DistilBERT, RoBERTa, and DeBERTa on the same question
"""

import gradio as gr
from transformers import pipeline
import torch

# Model configurations
MODELS = {
    "DistilBERT": {
        "name": "khaledbouabdallah/distilbert-squad-finetuned",
        "params": "66M",
        "f1": "84.41%",
        "em": "75.81%",
        "color": "#FF6B6B",
    },
    "RoBERTa": {
        "name": "khaledbouabdallah/roberta-squad-finetuned",
        "params": "125M",
        "f1": "91.96%",
        "em": "85.65%",
        "color": "#4ECDC4",
    },
    "DeBERTa": {
        "name": "khaledbouabdallah/deberta-squad-finetuned",
        "params": "184M",
        "f1": "93.01%",
        "em": "86.58%",
        "color": "#95E1D3",
    },
}

# Initialize pipelines
print("🔄 Loading models...")
device = 0 if torch.cuda.is_available() else -1
pipelines = {}

for name, config in MODELS.items():
    try:
        print(f"   Loading {name}...")
        pipelines[name] = pipeline(
            "question-answering", model=config["name"], device=device
        )
        print(f"   ✓ {name} loaded")
    except Exception as e:
        print(f"   ⚠️  Failed to load {name}: {e}")

print("✅ All models loaded!")


def highlight_answer(context, answer, start, end, confidence, color):
    """Highlight answer in context with color intensity based on confidence"""
    # Calculate opacity based on confidence
    opacity = max(0.3, min(0.9, confidence))

    # Create highlighted version
    before = context[:start]
    highlighted = f'<span style="background-color: {color}; opacity: {opacity}; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{answer}</span>'
    after = context[end:]

    return f'<div style="line-height: 1.8; font-family: system-ui;">{before}{highlighted}{after}</div>'


def answer_question(context, question):
    """Get answers from all three models"""

    if not context or not question:
        empty_msg = "Please provide both context and question."
        return [empty_msg, "0%", ""] * 3

    results = {}

    for model_name, pipeline_obj in pipelines.items():
        try:
            result = pipeline_obj(question=question, context=context)

            answer = result["answer"]
            score = result["score"]
            start = result["start"]
            end = result["end"]

            # Create highlighted context
            color = MODELS[model_name]["color"]
            highlighted_html = highlight_answer(
                context, answer, start, end, score, color
            )

            results[model_name] = {
                "answer": answer,
                "confidence": f"{score:.2%}",
                "highlighted": highlighted_html,
            }

        except Exception as e:
            results[model_name] = {
                "answer": f"Error: {str(e)}",
                "confidence": "0%",
                "highlighted": context,
            }

    # Return outputs for all three models
    return (
        # DistilBERT
        results["DistilBERT"]["answer"],
        results["DistilBERT"]["confidence"],
        results["DistilBERT"]["highlighted"],
        # RoBERTa
        results["RoBERTa"]["answer"],
        results["RoBERTa"]["confidence"],
        results["RoBERTa"]["highlighted"],
        # DeBERTa
        results["DeBERTa"]["answer"],
        results["DeBERTa"]["confidence"],
        results["DeBERTa"]["highlighted"],
    )


# Example questions
examples = [
    [
        "The Amazon rainforest, also known as Amazonia, covers 5.5 million square kilometers. It represents over half of the planet's remaining rainforests and comprises the largest and most biodiverse tract of tropical rainforest in the world.",
        "How large is the Amazon rainforest?",
    ],
    [
        "The Eiffel Tower was designed by Gustave Eiffel and completed in 1889. It stands 324 meters tall and was the world's tallest structure until the Chrysler Building was built in 1930.",
        "When was the Eiffel Tower completed?",
    ],
    [
        "Python was created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes code readability with its notable use of significant indentation.",
        "Who created Python?",
    ],
    [
        "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions. It extends from the Arctic Ocean in the north to the Southern Ocean in the south and is bounded by the continents of Asia and Australia in the west and the Americas in the east.",
        "Which ocean is the largest?",
    ],
]

# Create Gradio interface
with gr.Blocks(title="SQuAD QA - Model Comparison", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🤖 Question Answering: Model Comparison
        ## Compare DistilBERT, RoBERTa, and DeBERTa on SQuAD

        Ask the same question to all three models and compare their answers, confidence scores, and highlighted predictions.
        """
    )

    # Performance comparison table
    gr.Markdown(
        f"""
        ### 📊 Model Performance on SQuAD v1.1

        | Model | Parameters | F1 Score | Exact Match |
        |-------|-----------|----------|-------------|
        | **DistilBERT** | {MODELS['DistilBERT']['params']} | {MODELS['DistilBERT']['f1']} | {MODELS['DistilBERT']['em']} |
        | **RoBERTa** | {MODELS['RoBERTa']['params']} | {MODELS['RoBERTa']['f1']} | {MODELS['RoBERTa']['em']} |
        | **DeBERTa** | {MODELS['DeBERTa']['params']} | {MODELS['DeBERTa']['f1']} | {MODELS['DeBERTa']['em']} |
        """
    )

    # Input section
    with gr.Row():
        with gr.Column():
            context_input = gr.Textbox(
                label="📝 Context",
                placeholder="Enter the text containing the answer...",
                lines=8,
            )
            question_input = gr.Textbox(
                label="❓ Question",
                placeholder="Ask a question about the context...",
                lines=2,
            )
            submit_btn = gr.Button(
                "🚀 Get Answers from All Models", variant="primary", size="lg"
            )

    # Output section - Three columns
    with gr.Row():
        # DistilBERT Column
        with gr.Column():
            gr.Markdown(f"### 🔴 DistilBERT ({MODELS['DistilBERT']['params']})")
            distilbert_answer = gr.Textbox(label="Answer", lines=2)
            distilbert_confidence = gr.Textbox(label="Confidence")
            distilbert_highlight = gr.HTML(label="Highlighted Context")

        # RoBERTa Column
        with gr.Column():
            gr.Markdown(f"### 🔵 RoBERTa ({MODELS['RoBERTa']['params']})")
            roberta_answer = gr.Textbox(label="Answer", lines=2)
            roberta_confidence = gr.Textbox(label="Confidence")
            roberta_highlight = gr.HTML(label="Highlighted Context")

        # DeBERTa Column
        with gr.Column():
            gr.Markdown(f"### 🟢 DeBERTa ({MODELS['DeBERTa']['params']})")
            deberta_answer = gr.Textbox(label="Answer", lines=2)
            deberta_confidence = gr.Textbox(label="Confidence")
            deberta_highlight = gr.HTML(label="Highlighted Context")

    # Examples
    gr.Markdown("### 📝 Try these examples:")
    gr.Examples(
        examples=examples,
        inputs=[context_input, question_input],
    )

    # Footer
    gr.Markdown(
        """
        ---
        **About:** All models fine-tuned on SQuAD v1.1 (87,599 training examples) for 3 epochs with batch size 64.

        **Confidence colors** show where each model found the answer (color intensity = confidence level).

        **Training Details:**
        - Learning Rate: 3e-5
        - Warmup Ratio: 0.1
        - Max Length: 384 tokens
        - Document Stride: 128 tokens
        """
    )

    # Connect button
    submit_btn.click(
        fn=answer_question,
        inputs=[context_input, question_input],
        outputs=[
            distilbert_answer,
            distilbert_confidence,
            distilbert_highlight,
            roberta_answer,
            roberta_confidence,
            roberta_highlight,
            deberta_answer,
            deberta_confidence,
            deberta_highlight,
        ],
    )

if __name__ == "__main__":
    demo.launch()
