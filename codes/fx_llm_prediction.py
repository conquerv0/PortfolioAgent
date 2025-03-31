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
    "name": "fx_prediction",
    "schema": {
        "type": "object",
        "properties": {
            "currency_pairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pair": {
                            "type": "string",
                            "enum": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "USD/CAD"],
                            "description": "The currency pair"
                        },
                        "predicted_return": {
                            "type": "number",
                            "description": "Predicted return for the currency pair (as a decimal)"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence level in the prediction (0-1)"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Brief reasoning for this specific currency pair"
                        }
                    },
                    "required": ["pair", "predicted_return", "confidence", "rationale"],
                    "additionalProperties": False
                }
            },
            "overall_analysis": {
                "type": "string",
                "description": "Overall market analysis and cross-currency factors"
            }
        },
        "required": ["currency_pairs", "overall_analysis"],
        "additionalProperties": False
    },
    "strict": True
}

def prepare_prompt(row: pd.Series) -> str:
    """
    Prepare a prompt for GPT-4o based on the weekly features
    """
    prompt = """Based on the following weekly financial market data, predict the next week's returns for the G5 currency pairs.

This week's market data:
1. Currency ETFs:
- Euro (FXE): {FXE:.4f} (mom_1m: {FXE_mom_1m:.4f}, mom_3m: {FXE_mom_3m:.4f}, mom_12m: {FXE_mom_12m:.4f})
- British Pound (FXB): {FXB:.4f} (mom_1m: {FXB_mom_1m:.4f}, mom_3m: {FXB_mom_3m:.4f}, mom_12m: {FXB_mom_12m:.4f})
- Japanese Yen (FXY): {FXY:.4f} (mom_1m: {FXY_mom_1m:.4f}, mom_3m: {FXY_mom_3m:.4f}, mom_12m: {FXY_mom_12m:.4f})
- Swiss Franc (FXF): {FXF:.4f} (mom_1m: {FXF_mom_1m:.4f}, mom_3m: {FXF_mom_3m:.4f}, mom_12m: {FXF_mom_12m:.4f})
- Canadian Dollar (FXC): {FXC:.4f} (mom_1m: {FXC_mom_1m:.4f}, mom_3m: {FXC_mom_3m:.4f}, mom_12m: {FXC_mom_12m:.4f})

2. Interest Rates and Changes:
- US 10Y: {US_T10Y:.2f}% (Weekly Δ: {US_T10Y_weekly_change:.3f}%)
- EUR 10Y: {EUR_T10Y:.2f}% (Weekly Δ: {EUR_T10Y_weekly_change:.3f}%)
- GBP 10Y: {GBP_T10Y:.2f}% (Weekly Δ: {GBP_T10Y_weekly_change:.3f}%)
- JPY 10Y: {JPY_T10Y:.2f}% (Weekly Δ: {JPY_T10Y_weekly_change:.3f}%)
- CHF 10Y: {CHF_T10Y:.2f}% (Weekly Δ: {CHF_T10Y_weekly_change:.3f}%)
- CAD 10Y: {CAD_T10Y:.2f}% (Weekly Δ: {CAD_T10Y_weekly_change:.3f}%)

3. Risk Sentiment:
- VIX Index: {VIX:.2f} (Weekly Δ: {VIX_weekly_change:.3f})
- MOVE Index: {MOVE:.2f} (Weekly Δ: {MOVE_weekly_change:.3f})

For each of the G5 currency pairs (EUR/USD, GBP/USD, USD/JPY, USD/CHF, USD/CAD):
1. Predict next week's return as a decimal (e.g., 0.0025 for a 0.25% increase)
2. Provide a confidence score between 0 and 1 (where 0 is no confidence and 1 is complete certainty)
3. Give a brief rationale specific to each currency pair
4. Provide an overall market analysis

Your response must be structured in the required JSON format."""

    # Handle potential missing values
    formatted_prompt = ""
    try:
        formatted_prompt = prompt.format(**row.to_dict())
    except KeyError as e:
        print(f"Warning: Missing data for key {e}. Using default values.")
        # Create a copy with missing values filled
        row_copy = row.copy()
        for col in ['FXE_mom_1m', 'FXE_mom_3m', 'FXE_mom_12m', 'FXB_mom_1m', 'FXB_mom_3m', 'FXB_mom_12m',
                   'FXY_mom_1m', 'FXY_mom_3m', 'FXY_mom_12m', 'FXF_mom_1m', 'FXF_mom_3m', 'FXF_mom_12m',
                   'FXC_mom_1m', 'FXC_mom_3m', 'FXC_mom_12m', 'US_T10Y_weekly_change', 'EUR_T10Y_weekly_change',
                   'GBP_T10Y_weekly_change', 'JPY_T10Y_weekly_change', 'CHF_T10Y_weekly_change', 'CAD_T10Y_weekly_change',
                   'VIX_weekly_change', 'MOVE_weekly_change']:
            if col not in row_copy or pd.isna(row_copy[col]):
                row_copy[col] = 0.0
        formatted_prompt = prompt.format(**row_copy.to_dict())
    
    return formatted_prompt

def get_gpt4o_prediction(prompt: str) -> Dict:
    """
    Get prediction from GPT-4o API with structured output
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4o mini
            messages=[
                {"role": "system", "content": "You are a financial market expert providing accurate predictions for G5 currency movements. Your analysis should be based on fundamental and technical factors."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_schema", "json_schema": PREDICTION_SCHEMA},
            temperature=0
        )
        
        # Extract the JSON response
        prediction_str = response.choices[0].message.content.strip()
        prediction = json.loads(prediction_str)
        
        return prediction
    
    except Exception as e:
        print(f"Error getting prediction: {e}")
        # Return a structured error response
        return {
            "currency_pairs": [
                {"pair": "EUR/USD", "predicted_return": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                {"pair": "GBP/USD", "predicted_return": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                {"pair": "USD/JPY", "predicted_return": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                {"pair": "USD/CHF", "predicted_return": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                {"pair": "USD/CAD", "predicted_return": None, "confidence": 0, "rationale": f"Error: {str(e)}"}
            ],
            "overall_analysis": f"Failed to generate predictions due to error: {str(e)}"
        }

def main():
    # Load the weekly feature data
    print("Loading weekly data...")
    data = pd.read_csv('data/fx_combined_features_weekly.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    
    # Initialize list to store predictions
    predictions = []
    dates = []
    
    # Get predictions for each week
    total_weeks = len(data)
    for i, (date, row) in enumerate(data.iterrows(), 1):
        print(f"Processing week ending {date} ({i}/{total_weeks})...")
        
        # Prepare prompt with current week's data
        prompt = prepare_prompt(row)
        
        # Get prediction from GPT-4o
        prediction = get_gpt4o_prediction(prompt)
        
        # Store results
        predictions.append(prediction)
        dates.append(date)
        
        # Sleep to respect API rate limits
        time.sleep(1)
        
        # Save progress every 5 predictions
        if i % 5 == 0 or i == total_weeks:
            # Process and save the predictions in a more analysis-friendly format
            processed_data = {
                'date': [],
                'etf': [],  # New ETF column
                'currency_pair': [],
                'predicted_return': [],
                'confidence': [],
                'rationale': [],
                'overall_analysis': []
            }
            
            # Map currency pairs to their corresponding ETFs
            pair_to_etf = {
                "EUR/USD": "FXE",
                "GBP/USD": "FXB",
                "USD/JPY": "FXY",
                "USD/CHF": "FXF",
                "USD/CAD": "FXC"
            }
            
            for j, pred in enumerate(predictions):
                pred_date = dates[j]
                overall = pred.get('overall_analysis', '')
                
                for pair_data in pred.get('currency_pairs', []):
                    pair = pair_data.get('pair', '')
                    processed_data['date'].append(pred_date)
                    processed_data['etf'].append(pair_to_etf.get(pair, ""))  # Add the corresponding ETF
                    processed_data['currency_pair'].append(pair)
                    processed_data['predicted_return'].append(pair_data.get('predicted_return'))
                    processed_data['confidence'].append(pair_data.get('confidence'))
                    processed_data['rationale'].append(pair_data.get('rationale', ''))
                    processed_data['overall_analysis'].append(overall)
            
            # Create DataFrame and save
            pred_df = pd.DataFrame(processed_data)
            pred_df.to_csv(f'data/temp/fx_weekly_predictions_temp_{i}.csv', index=False)
            print(f"Progress saved: {i}/{total_weeks} predictions")
    
    # Process all predictions for final output
    processed_data = {
        'date': [],
        'etf': [],  # New ETF column
        'currency_pair': [],
        'predicted_return': [],
        'confidence': [],
        'rationale': [],
        'overall_analysis': []
    }
    
    # Map currency pairs to their corresponding ETFs
    pair_to_etf = {
        "EUR/USD": "FXE",
        "GBP/USD": "FXB",
        "USD/JPY": "FXY",
        "USD/CHF": "FXF",
        "USD/CAD": "FXC"
    }
    
    for j, pred in enumerate(predictions):
        pred_date = dates[j]
        overall = pred.get('overall_analysis', '')
        
        for pair_data in pred.get('currency_pairs', []):
            pair = pair_data.get('pair', '')
            processed_data['date'].append(pred_date)
            processed_data['etf'].append(pair_to_etf.get(pair, ""))  # Add the corresponding ETF
            processed_data['currency_pair'].append(pair)
            processed_data['predicted_return'].append(pair_data.get('predicted_return'))
            processed_data['confidence'].append(pair_data.get('confidence'))
            processed_data['rationale'].append(pair_data.get('rationale', ''))
            processed_data['overall_analysis'].append(overall)
    
    # Create final DataFrame
    final_df = pd.DataFrame(processed_data)
    
    # Save predictions
    final_df.to_csv('data/fx_weekly_predictions.csv', index=False)
    print("Weekly predictions completed and saved to 'data/fx_weekly_predictions.csv'")

    # Also save the raw predictions
    raw_df = pd.DataFrame({'date': dates, 'predictions': predictions})
    raw_df.to_json('data/fx_weekly_predictions_raw.json', orient='records')
    print("Raw predictions saved to 'data/fx_weekly_predictions_raw.json'")

if __name__ == "__main__":
    main() 