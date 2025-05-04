# PortfolioAgent

A multi-asset portfolio management system powered by LLMs (Large Language Models) that provides intelligent asset allocation recommendations across multiple asset classes.

## Overview

PortfolioAgent is a framework that combines traditional financial models with AI-powered insights to optimize investment portfolios. The system integrates data from various financial markets and uses LLMs to analyze this data and produce investment recommendations.

## Features

- **Multi-asset Class Coverage**:
  - Equities (Sector ETFs)
  - Fixed Income (Treasury and Credit ETFs)
  - Commodities
  - Foreign Exchange

- **Intelligent Analysis**:
  - Uses OpenAI's GPT models to analyze market data
  - Combines traditional financial metrics with AI insights
  - Produces structured recommendations with confidence scores

- **Advanced Backtesting**:
  - Black-Litterman model integration
  - Performance evaluation over historical data

## Project Structure

```
├── src/                     # Source code
│   ├── agent/               # Agent implementations
│   │   ├── PortfolioAgent.py    # Base agent class
│   │   ├── equity_agent.py      # Equity-specific agent
│   │   ├── fixedIncome_agent.py # Fixed income agent
│   │   ├── fx_agent.py          # Foreign exchange agent
│   │   ├── commodity_agent.py   # Commodity agent
│   │   ├── manager_agent.py     # Manager agent coordinating other agents
│   │   └── DataCollector.py     # Data collection utilities
│   ├── backtest/            # Backtesting framework
│   │   └── blackLitterman_test.py # Black-Litterman model implementation
│   ├── config/              # Configuration settings
│   │   └── settings.py      # Portfolio definitions and settings
│   ├── evaluator/           # Performance evaluation tools
│   └── util/                # Utility functions
├── data/                    # Data storage directory
├── report/                  # Analysis reports
├── slides/                  # Presentation materials
└── literature/              # Research papers and references
```

## Requirements

- Python 3.8+
- OpenAI API Key
- FRED API Key (for macroeconomic data)
- WRDS Login
- Dependencies listed in requirements.txt

## Setup

1. Clone the repository
2. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   FRED_API_KEY=your_fred_api_key
   WRDS_USERNAME=your_wrds_username 
   ```
3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

The system operates through specialized agents for each asset class that:

1. Collect relevant market data via `DataCollector` classes
2. Process and prepare data for LLM analysis
3. Generate investment recommendations using structured LLM prompts
4. Combine recommendations across asset classes via the `manager_agent`


## Acknowledgements

This project builds on research in portfolio optimization, specifically the Black-Litterman model, and leverages large language models for financial analysis. 