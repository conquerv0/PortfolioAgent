import pandas as pd
from openai import OpenAI
import json
import time
from typing import Dict
import sys
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure OpenAI API
client = OpenAI(api_key=OPENAI_API_KEY)

# Define the JSON schema for the fixed income prediction output (now predicting ETF returns)
PREDICTION_SCHEMA = {
    "name": "fi_prediction",
    "schema": {
        "type": "object",
        "properties": {
            "instruments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "instrument": {
                            "type": "string",
                            "enum": [
                                "Short-Term Treasury",
                                "1-3 Year Treasury",
                                "3-7 Year Treasury",
                                "7-10 Year Treasury",
                                "10-20 Year Treasury",
                                "20+ Year Treasury"
                            ],
                            "description": "The treasury ETF instrument"
                        },
                        "predicted_return": {
                            "type": "number",
                            "description": "Predicted return for the instrument (as a decimal)"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence level in the prediction (0-1)"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Brief reasoning for this specific instrument"
                        }
                    },
                    "required": ["instrument", "predicted_return", "confidence", "rationale"],
                    "additionalProperties": False
                }
            },
            "overall_analysis": {
                "type": "string",
                "description": "Overall fixed income market analysis and related macro insights"
            }
        },
        "required": ["instruments", "overall_analysis"],
        "additionalProperties": False
    },
    "strict": True
}

def prepare_prompt(row: pd.Series) -> str:
    """
    Prepare a prompt for GPT-4 based on the weekly fixed income market data.
    """
    prompt = """Based on the following weekly fixed income market data, predict the next week's yield changes for the fixed income instruments.

Macro Indicators:
- Effective Federal Funds Rate (EFFR): {EFFR:.4f}
- Headline PCE: {Headline_PCE:.4f}
- Core PCE: {Core_PCE:.4f}

US Treasury Yields and Momentum:
- 3-Month Yield: {3M_Yield:.2f}% (Momentum: 1M: {3M_Yield_mom_1m:.4f}, 3M: {3M_Yield_mom_3m:.4f}, 12M: {3M_Yield_mom_12m:.4f})
- 6-Month Yield: {6M_Yield:.2f}% (Momentum: 1M: {6M_Yield_mom_1m:.4f}, 3M: {6M_Yield_mom_3m:.4f}, 12M: {6M_Yield_mom_12m:.4f})
- 1-Year Yield: {1Y_Yield:.2f}% (Momentum: 1M: {1Y_Yield_mom_1m:.4f}, 3M: {1Y_Yield_mom_3m:.4f}, 12M: {1Y_Yield_mom_12m:.4f})
- 2-Year Yield: {2Y_Yield:.2f}% (Momentum: 1M: {2Y_Yield_mom_1m:.4f}, 3M: {2Y_Yield_mom_3m:.4f}, 12M: {2Y_Yield_mom_12m:.4f})
- 5-Year Yield: {5Y_Yield:.2f}% (Momentum: 1M: {5Y_Yield_mom_1m:.4f}, 3M: {5Y_Yield_mom_3m:.4f}, 12M: {5Y_Yield_mom_12m:.4f})
- 10-Year Yield: {10Y_Yield:.2f}% (Momentum: 1M: {10Y_Yield_mom_1m:.4f}, 3M: {10Y_Yield_mom_3m:.4f}, 12M: {10Y_Yield_mom_12m:.4f})

Risk Sentiment:
- VIX Index: {VIX:.2f} (Weekly change: {VIX_weekly_change:.3f})
- MOVE Index: {MOVE:.2f} (Weekly change: {MOVE_weekly_change:.3f})

Based on the above data, please predict next week's yield changes for the following fixed income instruments:
- US 10-Year (based on 10Y_Yield)
- US 5-Year (based on 5Y_Yield)
- US 2-Year (based on 2Y_Yield)
- EUR 10-Year (based on EUR_T10Y)
- JPY 10-Year (based on JPY_T10Y)
- UK 10-Year (based on GBP_T10Y)

For each instrument, provide:
1. The predicted yield change as a decimal (e.g., -0.0025 for a -0.25% change),
2. A confidence score between 0 and 1,
3. A brief rationale for the prediction.

Also, include an overall fixed income market analysis that incorporates these macro indicators, yield levels, momentum signals, and risk sentiment.

Your response must be structured in the required JSON format.
"""
    try:
        formatted_prompt = prompt.format(**row.to_dict())
    except KeyError as e:
        print(f"Warning: Missing data for key {e}. Filling with default 0.0 values.")
        row_copy = row.copy()
        default_cols = [
            '3M_Yield_mom_1m', '3M_Yield_mom_3m', '3M_Yield_mom_12m',
            '6M_Yield_mom_1m', '6M_Yield_mom_3m', '6M_Yield_mom_12m',
            '1Y_Yield_mom_1m', '1Y_Yield_mom_3m', '1Y_Yield_mom_12m',
            '2Y_Yield_mom_1m', '2Y_Yield_mom_3m', '2Y_Yield_mom_12m',
            '5Y_Yield_mom_1m', '5Y_Yield_mom_3m', '5Y_Yield_mom_12m',
            '10Y_Yield_mom_1m', '10Y_Yield_mom_3m', '10Y_Yield_mom_12m',
            'EFFR', 'Headline_PCE', 'Core_PCE',
            '3M_Yield', '6M_Yield', '1Y_Yield', '2Y_Yield', '5Y_Yield', '10Y_Yield',
            'JPY_T10Y', 'GBP_T10Y',
            'VIX', 'VIX_weekly_change', 'MOVE', 'MOVE_weekly_change'
        ]
        for col in default_cols:
            if col not in row_copy or pd.isna(row_copy[col]):
                row_copy[col] = 0.0
        formatted_prompt = prompt.format(**row_copy.to_dict())
    
    return formatted_prompt

def get_gpt4o_prediction(prompt: str) -> Dict:
    """
    Get prediction from GPT-4 API with structured output for fixed income.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4o mini
            messages=[
                {"role": "system", "content": "You are a fixed income market expert. Provide yield change predictions based on macroeconomic, treasury, and risk sentiment data."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_schema", "json_schema": PREDICTION_SCHEMA},
            temperature=0
        )
        prediction_str = response.choices[0].message.content.strip()
        prediction = json.loads(prediction_str)
        return prediction
    except Exception as e:
        print(f"Error getting prediction: {e}")
        return {
            "instruments": [
                {"instrument": "Short-Term Treasury", "predicted_yield_change": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                {"instrument": "1-3 Year Treasury", "predicted_yield_change": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                {"instrument": "3-7 Year Treasury", "predicted_yield_change": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                {"instrument": "7-10 Year Treasury", "predicted_yield_change": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                {"instrument": "10-20 Year Treasury", "predicted_yield_change": None, "confidence": 0, "rationale": f"Error: {str(e)}"},
                {"instrument": "20+ Year Treasury", "predicted_yield_change": None, "confidence": 0, "rationale": f"Error: {str(e)}"}
            ],
            "overall_analysis": f"Failed to generate predictions due to error: {str(e)}"
        }

def main():
    # Load the weekly fixed income features data produced by the collector
    print("Loading weekly fixed income data...")
    data = pd.read_csv('data/fi_combined_features_weekly.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    
    predictions = []
    dates = []
    total_weeks = len(data)
    
    # Mapping from instrument names to ETF tickers
    instrument_to_etf = {
        "Short-Term Treasury": "SHV",
        "1-3 Year Treasury": "SHY",
        "3-7 Year Treasury": "IEI",
        "7-10 Year Treasury": "IEF",
        "10-20 Year Treasury": "TLH",
        "20+ Year Treasury": "TLT"
    }
    
    for i, (date, row) in enumerate(data.iterrows(), 1):
        print(f"Processing week ending {date} ({i}/{total_weeks})...")
        prompt = prepare_prompt(row)
        prediction = get_gpt4o_prediction(prompt)
        predictions.append(prediction)
        dates.append(date)
        time.sleep(1)  # Respect API rate limits
        
        if i % 5 == 0 or i == total_weeks:
            # Process and save the predictions in a more analysis-friendly format
            processed_data = {
                'date': [],
                'etf': [],  # New ETF column based on instrument mapping
                'instrument': [],
                'predicted_return': [],
                'confidence': [],
                'rationale': [],
                'overall_analysis': []
            }
            for j, pred in enumerate(predictions):
                pred_date = dates[j]
                overall = pred.get('overall_analysis', '')
                for inst in pred.get('instruments', []):
                    instrument = inst.get('instrument', '')
                    processed_data['date'].append(pred_date)
                    # Map instrument to ETF ticker
                    processed_data['etf'].append(instrument_to_etf.get(instrument, ""))
                    processed_data['instrument'].append(instrument)
                    processed_data['predicted_return'].append(inst.get('predicted_return'))
                    processed_data['confidence'].append(inst.get('confidence'))
                    processed_data['rationale'].append(inst.get('rationale', ''))
                    processed_data['overall_analysis'].append(overall)
            
            temp_filename = f'data/temp/fi_weekly_predictions_temp_{i}.csv'
            os.makedirs(os.path.dirname(temp_filename), exist_ok=True)
            temp_df = pd.DataFrame(processed_data)
            temp_df.to_csv(temp_filename, index=False)
            print(f"Progress saved: {i}/{total_weeks} predictions")
    
    # Final processing of predictions
    processed_data = {
        'date': [],
        'etf': [],
        'instrument': [],
        'predicted_return': [],
        'confidence': [],
        'rationale': [],
        'overall_analysis': []
    }
    for j, pred in enumerate(predictions):
        pred_date = dates[j]
        overall = pred.get('overall_analysis', '')
        for inst in pred.get('instruments', []):
            instrument = inst.get('instrument', '')
            processed_data['date'].append(pred_date)
            processed_data['etf'].append(instrument_to_etf.get(instrument, ""))
            processed_data['instrument'].append(instrument)
            processed_data['predicted_return'].append(inst.get('predicted_return'))
            processed_data['confidence'].append(inst.get('confidence'))
            processed_data['rationale'].append(inst.get('rationale', ''))
            processed_data['overall_analysis'].append(overall)
    
    final_df = pd.DataFrame(processed_data)
    final_csv = 'data/fi_weekly_predictions.csv'
    final_df.to_csv(final_csv, index=False)
    print(f"Weekly fixed income predictions completed and saved to '{final_csv}'")
    
    raw_df = pd.DataFrame({'date': dates, 'predictions': predictions})
    raw_json = 'data/fi_weekly_predictions_raw.json'
    raw_df.to_json(raw_json, orient='records')
    print(f"Raw predictions saved to '{raw_json}'")

if __name__ == "__main__":
    main()
