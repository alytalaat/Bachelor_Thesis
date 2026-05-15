# LLM-Coordinated Multi-Agent Systems for Database Management

Bachelor thesis implementation — GUC, 2026.

## Overview
A multi-agent pipeline that accepts natural language questions, 
generates role-appropriate SQL, verifies it, and executes it 
against a database — with role-based access control and 
row-level security.

## System Components
- **Coordinator** — orchestrates the full request lifecycle
- **Coder** — generates SQL candidates from a structured plan
- **Verifier** — runs three sequential checks before execution

## How to Run

### Install dependencies
pip install -r requirements.txt

### Set up environment variables
Create a `.env` file with your Groq API key:
GROQ_API_KEY=your_key_here

### Run a query
python query_agent.py

### Run benchmark
python spider_runner.py --difficulty hard
python spider_runner.py --no_plan --difficulty hard

## Thesis
This repository accompanies the thesis:
"LLM-Coordinated Multi-Agent Systems for Database Management Tasks"
