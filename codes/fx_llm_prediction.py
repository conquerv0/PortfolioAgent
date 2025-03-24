import pandas as pd
from openai import OpenAI
from datetime import datetime
import json
import time
from typing import Dict, List
import sys
import os

# Add the parent directory to the path so we can import the config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

# Configure OpenAI API
client = OpenAI(api_key=OPENAI_API_KEY)

# Define the JSON schema for structured output
PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "predicted_return": {
            "type": "number",
            "description": "Predicted return for UUP tomorrow (as a decimal)"
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "Confidence level in the prediction"
        },
        "rationale": {
            "type": "string",
            "description": "Brief reasoning behind the prediction"
        }
    },
    "required": ["predicted_return", "confidence", "rationale"]
}

def prepare_prompt(row: pd.Series) -> str:
    """
    Prepare a prompt for GPT-4o based on the features for a given day
    """
    prompt = """Based on the following financial market data, predict the next day's return for the US Dollar Index (UUP ETF).

Today's market data:
1. Currency ETF:
- UUP price: {UUP}
- UUP 1-month momentum: {UUP_mom_1m:.4f}
- UUP 3-month momentum: {UUP_mom_3m:.4f}
- UUP 12-month momentum: {UUP_mom_12m:.4f}

2. Exchange Rates:
- EUR/USD: {EURUSD=X:.4f}
- JPY/USD: {JPYUSD=X:.4f}
- GBP/USD: {GBPUSD=X:.4f}

3. Interest Rates:
- US 10Y Treasury: {US_T10Y:.2f}%
- Eurozone 10Y: {EUR_T10Y:.2f}%
- Japan 10Y: {JPY_T10Y:.2f}%
- UK 10Y: {GBP_T10Y:.2f}%

4. Rate Changes:
- US 10Y Change: {US_T10Y_change:.3f}%
- EUR 10Y Change: {EUR_T10Y_change:.3f}%
- JPY 10Y Change: {JPY_T10Y_change:.3f}%
- GBP 10Y Change: {GBP_T10Y_change:.3f}%

5. Risk Metrics:
- VIX Index: {VIX:.2f}
- S&P 500 Return: {SPY_returns:.4f}

6. Macro Indicators:
- Fed Funds Rate: {FedRate:.2f}%
- 10Y-2Y Spread: {YieldSpread:.2f}%
- Fed Rate Change: {FedRate_change:.3f}%
- Yield Spread Change: {YieldSpread_change:.3f}%

Predict tomorrow's return for UUP and provide a brief rationale."""

    return prompt.format(**row.to_dict())

def get_gpt4o_prediction(prompt: str) -> Dict:
    """
    Get prediction from GPT-4o API with structured output
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using the latest GPT-4o model
            messages=[
                {"role": "system", "content": "You are a financial market expert providing accurate predictions for currency movements."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object", "schema": PREDICTION_SCHEMA, "strict": True},
            temperature=0.7,
            max_tokens=300
        )
        
        # Extract the JSON response
        prediction_str = response.choices[0].message.content.strip()
        prediction = json.loads(prediction_str)
        
        return prediction
    
    except Exception as e:
        print(f"Error getting prediction: {e}")
        return {"predicted_return": None, "confidence": None, "rationale": f"Error: {str(e)}"}

def main():
    # Load the feature data
    data = pd.read_csv('data/fx_combined_features.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    
    # Initialize lists to store predictions
    predictions = []
    dates = []
    
    # Get predictions for each day
    for date, row in data.iterrows():
        print(f"Processing {date}...")
        
        # Prepare prompt with current day's data
        prompt = prepare_prompt(row)
        
        # Get prediction from GPT-4o
        prediction = get_gpt4o_prediction(prompt)
        
        # Store results
        predictions.append(prediction)
        dates.append(date)
        
        # Sleep to respect API rate limits
        time.sleep(1)
    
    # Create predictions DataFrame
    pred_df = pd.DataFrame(predictions, index=dates)
    
    # Save predictions
    pred_df.to_csv('data/fx_llm_predictions.csv')
    print("Predictions completed and saved to 'data/fx_llm_predictions.csv'")

if __name__ == "__main__":
    main() 