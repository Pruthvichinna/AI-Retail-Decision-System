# src/insights.py

from transformers import pipeline

def generate_ai_summary(optimization_results, model_name_or_path="google/flan-t5-small"):
    prompt = f"""
    You are a senior retail analyst writing a weekly brief for a store manager.
    Based on the results from our promotion optimization model, please write a concise, professional, and actionable summary.

    The data is as follows:
    - Products to Promote (by ID): {optimization_results['promoted_products']}
    - Maximum Expected Weekly Revenue: ${optimization_results['expected_revenue']:,.2f}
    - Total Cost of Discounts: ${optimization_results['discount_cost']:,.2f}

    Please structure your response with a clear heading, a brief summary, and a list of actionable next steps for the store manager.
    """
    
    # Change 'model=model_path' to 'model=model_name_or_path'
    summarizer = pipeline("text2text-generation", model=model_name_or_path)
    ai_report = summarizer(prompt, max_length=200)[0]['generated_text']
    return ai_report